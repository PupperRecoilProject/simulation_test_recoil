# hardware_controller.py
import serial
import serial.tools.list_ports
import threading
import time
import re
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

# 為了型別提示，避免迴圈匯入
if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager # 【修改】為了調用新函式，將 ONNXPolicy 改為 PolicyManager
    from state import SimulationState

class RobotStateHardware:
    """一個專門用來儲存從實體機器人獲取的即時狀態的數據類別。"""
    def __init__(self):
        self.imu_gyro_radps = np.zeros(3, dtype=np.float32) # IMU角速度 (rad/s)
        self.imu_acc_g = np.zeros(3, dtype=np.float32) # IMU加速度 (g)
        self.joint_positions_rad = np.zeros(12, dtype=np.float32) # 關節角度 (rad)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32) # 關節角速度 (rad/s)
        self.lin_vel_local = np.zeros(3, dtype=np.float32) # 估算的機身局部座標系線速度 (m/s)
        self.gravity_vector_local = np.zeros(3, dtype=np.float32) # 估算的機身局部座標系重力向量
        self.last_action = np.zeros(12, dtype=np.float32) # 上一次AI輸出的動作
        self.command = np.zeros(3, dtype=np.float32) # 使用者下達的指令
        self.last_update_time = 0.0 # 上次收到硬體數據的時間戳

class HardwareController:
    """管理與實體硬體(例如Teensy)的連接和高頻控制迴圈。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', global_state: 'SimulationState'):
        """
        初始化硬體控制器。
        Args:
            config (AppConfig): 應用程式的全域設定。
            policy (PolicyManager): 策略管理器，用於獲取AI動作。
            global_state (SimulationState): 全域狀態物件，用於讀取使用者指令等。
        """
        self.config = config
        self.policy = policy
        self.global_state = global_state
        
        self.ser = None # 序列埠物件
        self.is_running = False # 控制執行緒是否繼續運行的旗標
        self.read_thread = None # 讀取序列埠的執行緒
        self.control_thread = None # 執行AI控制迴圈的執行緒
        
        self.hw_state = RobotStateHardware() # 儲存硬體狀態的實例
        self.lock = threading.Lock() # 執行緒鎖，用於保護對 hw_state 的同時讀寫
        self.ai_control_enabled = threading.Event() # 事件旗標，用於暫停/恢復AI控制迴圈

        # 【重要】: 機器人運動學參數
        # 預設站姿下，四個腳尖相對於身體中心(質心或IMU位置)的座標 (單位: 米)
        # 注意: 這個值需要根據您的實體機器人精確測量或從 URDF/CAD 模型中導出。
        # 這個值是根據您專案中的 `pupper.xml` 推算的，您可能需要微調。
        self.foot_positions_in_body = np.array([
            [-0.0804, -0.1759, -0.1964],  # FR (Front Right)
            [ 0.0806, -0.1759, -0.1964],  # FL (Front Left)
            [-0.0804,  0.0239, -0.1964],  # RR (Rear Right)
            [ 0.0806,  0.0239, -0.1964],  # RL (Rear Left)
        ], dtype=np.float32)

        print("✅ 硬體控制器已初始化。")

    def connect_and_start(self) -> bool:
        """掃描並連接到序列埠，如果成功，則啟動所有背景執行緒。"""
        if self.is_running:
            print("硬體控制器已在運行中。")
            return True
            
        print("\n" + "="*20 + " 正在掃描可用序列埠 " + "="*20)
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("❌ 錯誤: 未找到任何序列埠。")
            return False
        
        # 這裡我們自動選擇第一個找到的埠，您可以根據需要修改為手動選擇
        port_name = ports[0].device
        print(f"自動選擇埠: {port_name} (波特率: 115200)")

        try:
            self.ser = serial.Serial(port_name, 115200, timeout=1)
            time.sleep(1.0) # 等待序列埠穩定
            self.ser.flushInput() # 清空輸入緩衝區
            print(f"✅ 成功連接到 {port_name}")
            
            self.is_running = True
            # 建立並啟動讀取和控制的背景執行緒
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            print("✅ 硬體控制執行緒已啟動。")
            return True
        except serial.SerialException as e:
            print(f"❌ 連接失敗: {e}")
            self.ser = None
            return False

    def stop(self):
        """安全地停止所有執行緒和序列埠連接。"""
        if not self.is_running: return
        
        print("正在停止硬體控制器...")
        self.is_running = False # 設定旗標，讓執行緒的 while 循環退出
        self.disable_ai() # 確保AI已暫停
        self.ai_control_enabled.set() # 喚醒可能正在等待的控制執行緒，以便它能檢查 is_running 旗標並退出
        
        # 等待執行緒結束
        if self.control_thread and self.control_thread.is_alive(): self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive(): self.read_thread.join(timeout=1)
        
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"序列埠 {self.ser.port} 已關閉。")
        
        self.ser = None
        print("硬體控制器已完全停止。")
        
    def enable_ai(self):
        """啟用 AI 控制。"""
        if not self.is_running:
            print("無法啟用 AI：硬體控制器未運行。")
            return
        print("🤖 AI 控制已啟用。")
        self.policy.reset() # 重置AI策略的內部狀態（例如歷史觀察）
        self.ai_control_enabled.set() # 設定事件，讓控制迴圈開始運行
        self.global_state.hardware_ai_is_active = True

    def disable_ai(self):
        """禁用 AI 控制。"""
        print("⏸️ AI 控制已暫停。")
        self.ai_control_enabled.clear() # 清除事件，讓控制迴圈暫停在 wait()
        self.global_state.hardware_ai_is_active = False
        # 向硬體發送一個停止指令，讓其恢復到預設站姿或安全狀態
        if self.ser and self.ser.is_open:
            try: self.ser.write(b"stop\n")
            except serial.SerialException as e: print(f"發送停止指令失敗: {e}")

    def parse_teensy_data(self, line: str):
        """使用正則表達式解析來自 Teensy 的單行字串數據。"""
        # 匹配加速度數據的格式
        acc_match = re.search(r"IMU Acc\(g\) -> X: ([+-]?[\d.]+)\s+Y: ([+-]?[\d.]+)\s+Z: ([+-]?[\d.]+)", line)
        # 匹配角速度數據的格式
        gyro_match = re.search(r"IMU Gyro\(dps\)-> X: ([+-]?[\d.]+)\s+Y: ([+-]?[\d.]+)\s+Z: ([+-]?[\d.]+)", line)
        # 匹配馬達數據的格式
        motor_match = re.search(r"Motor\s+(\d+)\s*\|\s*Pos:\s+([+-]?[\d.]+)\s*\|\s*Vel:\s+([+-]?[\d.]+)", line)

        with self.lock: # 使用鎖保護對共享資源 hw_state 的寫入
            if acc_match:
                self.hw_state.imu_acc_g = np.array([float(g) for g in acc_match.groups()], dtype=np.float32)
            elif gyro_match:
                dps = np.array([float(g) for g in gyro_match.groups()], dtype=np.float32)
                self.hw_state.imu_gyro_radps = dps * (np.pi / 180.0) # 將 dps (度/秒) 轉換為 rad/s
            elif motor_match:
                motor_id = int(motor_match.group(1))
                if 0 <= motor_id < self.config.num_motors:
                    self.hw_state.joint_positions_rad[motor_id] = float(motor_match.group(2))
                    self.hw_state.joint_velocities_radps[motor_id] = float(motor_match.group(3))
            self.hw_state.last_update_time = time.time() # 更新收到數據的時間戳

    def estimate_linear_velocity(self):
        """[核心演算法] 根據 IMU 和運動學模型，估算機身的線速度。"""
        # 假設：當機器人移動時，總有一隻腳是支撐腳（stance foot），其與地面的相對速度為零。
        with self.lock: # 複製數據以避免長時間鎖定
            acc_g = self.hw_state.imu_acc_g.copy()
            w_body = self.hw_state.imu_gyro_radps.copy()
        
        # 靜止時，加速度計讀數主要反映重力
        body_gravity_vec = acc_g * 9.81
        world_gravity_vec = np.array([0, 0, -9.81])
        
        if np.linalg.norm(body_gravity_vec) < 1e-6: return # 避免除以零

        try:
            # 計算從機身座標系到世界座標系的旋轉矩陣
            rot_body_to_world, _ = Rotation.align_vectors(world_gravity_vec.reshape(1, -1), body_gravity_vec.reshape(1, -1))
        except (ValueError, np.linalg.LinAlgError): return

        # 將機身角速度轉換到世界座標系
        w_world = rot_body_to_world.apply(w_body)

        # 估算每隻腳因機身旋轉而產生的速度，並找到速度最小的腳作為支撐腳
        foot_velocities_world = []
        for i in range(4):
            r_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[i])
            v_foot_world = np.cross(w_world, r_foot_in_world)
            foot_velocities_world.append(np.linalg.norm(v_foot_world))

        stance_foot_idx = np.argmin(foot_velocities_world)
        r_stance_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[stance_foot_idx])
        # 根據公式 V_body = - (w x r_stance_foot)，估算機身速度
        v_body_world_est = -np.cross(w_world, r_stance_foot_in_world)

        with self.lock: # 更新估算結果
            self.hw_state.gravity_vector_local = body_gravity_vec
            self.hw_state.lin_vel_local = rot_body_to_world.inv().apply(v_body_world_est)

    def construct_observation(self) -> np.ndarray:
        """建立提供給 ONNX 模型的觀察向量。"""
        self.estimate_linear_velocity() # 首先更新速度估算
        
        with self.lock:
            # 根據設定檔縮放使用者指令
            self.hw_state.command = self.global_state.command * np.array(self.config.command_scaling_factors)
            
            # 建立一個字典，包含所有可能的觀察分量
            obs_list = {
                'linear_velocity': self.hw_state.lin_vel_local,
                'angular_velocity': self.hw_state.imu_gyro_radps,
                'gravity_vector': self.hw_state.gravity_vector_local / 9.81, # 模型通常需要歸一化的重力向量
                'accelerometer': self.hw_state.imu_acc_g,
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command,
            }
            
            # 【核心修正】從 PolicyManager 動態獲取當前啟用模型的觀察配方
            recipe = self.policy.get_active_recipe()
            if not recipe:
                print("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
                return np.array([])

            # 根據配方要求的順序，從字典中取出對應的數據分量
            final_obs_list = [obs_list[key] for key in recipe if key in obs_list]

            # 拼接成最終的觀察向量
            return np.concatenate(final_obs_list).astype(np.float32)

    def _read_from_port(self):
        """[背景執行緒] 持續從序列埠讀取數據並調用解析器。"""
        print("[硬體讀取線程已啟動] 等待來自 Teensy 的數據...")
        while self.is_running:
            if not self.ser or not self.ser.is_open:
                self.stop() # 如果序列埠異常，停止所有服務
                break
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line: self.parse_teensy_data(line)
            except (serial.SerialException, OSError):
                print("❌ 錯誤：序列埠斷開連接或讀取錯誤。")
                self.stop()
                break
            except Exception as e: print(f"❌ _read_from_port 發生未知錯誤: {e}")
                
    def _control_loop(self):
        """[背景執行緒] 以固定頻率執行 AI 控制。"""
        print("\n--- 硬體控制線程已就緒，等待 AI 啟用 ---")
        # 假設硬體的預設站姿與模擬中的 `default_pose` 相同
        default_pose_hardware = self.global_state.sim.default_pose

        while self.is_running:
            self.ai_control_enabled.wait() # 等待 enable_ai() 被呼叫
            if not self.is_running: break # 在喚醒後再次檢查運行旗標

            loop_start_time = time.perf_counter()
            
            # 1. 建立觀察向量
            observation = self.construct_observation()
            if observation.size == 0: # 如果無法建立觀察，則跳過本輪
                time.sleep(0.02)
                continue
            
            # 2. 執行AI推論，獲取原始動作
            _, action_raw = self.policy.get_action_for_hardware(observation)
            
            with self.lock:
                self.hw_state.last_action[:] = action_raw
            
            # 3. 計算最終發送到馬達的目標角度
            # 這裡的邏輯與模擬中完全一致
            final_command = default_pose_hardware + action_raw * self.global_state.tuning_params.action_scale

            # 4. 將指令格式化為字串並發送
            action_str = ' '.join(f"{a:.4f}" for a in final_command)
            command_to_send = f"jpos {action_str}\n"

            if self.ser and self.ser.is_open:
                try: self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException: self.stop()
            
            # 5. 精確控制迴圈頻率
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)