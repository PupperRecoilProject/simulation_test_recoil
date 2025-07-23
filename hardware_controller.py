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
    from policy import PolicyManager
    from state import SimulationState

class RobotStateHardware:
    """
    一個專門用來儲存從實體機器人獲取的即時狀態的數據容器(Data Class)。
    這個類別的設計是為了在多執行緒環境下安全地儲存一份硬體狀態的快照。
    【新版】增加了更多欄位以匹配新的CSV通訊協定。
    """
    def __init__(self):
        # --- AI模型直接需要的核心數據 ---
        self.imu_gyro_radps = np.zeros(3, dtype=np.float32)       # IMU角速度 (rad/s) - 【注意】這是從RPY估算得來的，因為硬體直接提供RPY
        self.imu_acc_g = np.zeros(3, dtype=np.float32)            # IMU加速度 (單位: g)
        self.joint_positions_rad = np.zeros(12, dtype=np.float32) # 所有關節的當前角度 (rad)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32)# 所有關節的當前角速度 (rad/s)
        self.lin_vel_local = np.zeros(3, dtype=np.float32)        # 估算的機身局部座標系線速度 (m/s)
        self.gravity_vector_local = np.zeros(3, dtype=np.float32) # 估算的機身局部座標系重力向量
        self.last_action = np.zeros(12, dtype=np.float32)         # 上一幀AI輸出的動作指令
        self.command = np.zeros(3, dtype=np.float32)              # 使用者下達的目標速度指令
        
        # --- 從硬體接收的原始數據和狀態 (用於除錯和未來擴展) ---
        self.timestamp_ms = 0                                     # 硬體回傳的時間戳 (ms)
        self.robot_mode = "N/A"                                   # 硬體回傳的機器人模式 (e.g., IDLE)
        self.is_calibrated = False                                # 硬體回傳的是否已校準狀態
        self.rpy_rad = np.zeros(3, dtype=np.float32)              # 姿態角 Roll, Pitch, Yaw (rad)
        self.target_current_ma = np.zeros(12, dtype=np.float32)   # 目標電流 (mA)
        self.actual_current_ma = np.zeros(12, dtype=np.float32)   # 實際電流 (mA)
        
        # --- 用於角速度估算的內部變數 ---
        self.prev_rpy_rad = np.zeros(3, dtype=np.float32)         # 儲存上一幀的姿態角，用於計算差值
        self.prev_rpy_time = 0.0                                  # 儲存上一幀的時間戳，用於計算時間差
        
        self.last_update_time = 0.0                               # PC端上次成功解析硬體數據的時間戳

class HardwareController:
    """
    管理與實體硬體(例如Teensy)的通訊和高頻控制迴圈。
    主要職責：
    1. 建立和管理序列埠連接。
    2. 在背景執行緒中讀取和解析來自硬體的感測器數據。
    3. 根據感測器數據建立AI模型所需的觀察向量(Observation)。
    4. 在背景執行緒中以固定頻率運行AI模型推論。
    5. 將AI模型的輸出格式化為硬體可接收的指令並發送。
    """
    
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
        
        self.ser = None # serial.Serial 物件
        self.is_running = False # 控制背景執行緒是否繼續運行的旗標
        self.read_thread = None # 讀取序列埠的執行緒
        self.control_thread = None # 執行AI控制迴圈的執行緒
        
        self.hw_state = RobotStateHardware() # 儲存硬體狀態的實例
        self.lock = threading.Lock() # 執行緒鎖，用於保護對 hw_state 的同時讀寫，避免數據競爭
        self.ai_control_enabled = threading.Event() # 事件旗標，用於優雅地暫停/恢復AI控制迴圈

        # 【重要】: 機器人運動學參數
        # 預設站姿下，四個腳尖相對於身體中心(質心或IMU位置)的座標 (單位: 米)
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
        
        # 簡化處理，自動選擇找到的第一個埠
        port_name = ports[0].device
        print(f"自動選擇埠: {port_name} (波特率: 115200)")

        try:
            self.ser = serial.Serial(port_name, 115200, timeout=1)
            time.sleep(1.0) # 等待序列埠穩定
            self.ser.flushInput() # 清空可能殘留的舊數據
            print(f"✅ 成功連接到 {port_name}")
            
            self.is_running = True
            # 建立並啟動讀取和控制的背景執行緒，daemon=True 表示主程式退出時它們也會跟著退出
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
        self.is_running = False # 1. 設定旗標，讓執行緒的 while 循環在下一輪退出
        self.disable_ai()       # 2. 確保AI已暫停
        self.ai_control_enabled.set() # 3. 喚醒可能正在 wait() 的控制執行緒，以便它能檢查 is_running 旗標並退出
        
        # 4. 等待執行緒真正結束
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
        self.ai_control_enabled.set() # 設定事件，讓控制迴圈的 wait() 通過，開始運行
        self.global_state.hardware_ai_is_active = True

    def disable_ai(self):
        """禁用 AI 控制。"""
        print("⏸️ AI 控制已暫停。")
        self.ai_control_enabled.clear() # 清除事件，讓控制迴圈在下一輪暫停在 wait()
        self.global_state.hardware_ai_is_active = False
        # 向硬體發送一個停止指令，讓其恢復到預設站姿或安全狀態
        if self.ser and self.ser.is_open:
            try: self.ser.write(b"stop\n")
            except serial.SerialException as e: print(f"發送停止指令失敗: {e}")

    def parse_teensy_data(self, line: str):
        """
        【核心重構】解析來自硬體的單行CSV格式數據。
        """
        try:
            # 1. 將CSV字串按逗號分割成一個字串列表
            parts = line.split(',')
            
            # 2. 健全性檢查：確保欄位數量正確，防止因數據傳輸不完整而導致程式崩潰
            # 計算方式: 1(ts)+1(mode)+1(cal)+3(rpy)+3(acc)+12(pos)+12(vel)+12(targ_curr)+12(act_curr) = 57
            if len(parts) != 57:
                return # 如果數量不對，靜默丟棄這一幀數據，避免洗版

            with self.lock: # 使用鎖保護對共享資源 hw_state 的寫入
                current_time = time.time()
                
                # 3. 逐個解析欄位並儲存到 hw_state 物件中
                self.hw_state.timestamp_ms = int(parts[0])
                self.hw_state.robot_mode = parts[1]
                self.hw_state.is_calibrated = (parts[2] == '1')
                
                # 解析RPY (Roll, Pitch, Yaw)，並從角度轉換為弧度
                rpy_deg = np.array(parts[3:6], dtype=np.float32)
                self.hw_state.rpy_rad = rpy_deg * (np.pi / 180.0)
                
                # 解析加速度
                self.hw_state.imu_acc_g = np.array(parts[6:9], dtype=np.float32)
                
                # 解析12個關節的角度 (假設硬體已提供弧度)
                self.hw_state.joint_positions_rad = np.array(parts[9:21], dtype=np.float32)
                
                # 解析12個關節的速度 (假設硬體已提供 rad/s)
                self.hw_state.joint_velocities_radps = np.array(parts[21:33], dtype=np.float32)
                
                # 解析目標和實際電流 (即使AI不用，也解析出來以備後用)
                self.hw_state.target_current_ma = np.array(parts[33:45], dtype=np.float32)
                self.hw_state.actual_current_ma = np.array(parts[45:57], dtype=np.float32)
                
                # 更新PC端的時間戳
                self.hw_state.last_update_time = current_time

        except (ValueError, IndexError) as e:
            # 如果在解析過程中發生錯誤 (例如某個欄位不是數字)，則打印錯誤並跳過
            print(f"❌ 解析硬體數據時出錯: {e} | 原始數據: {line}")

    def estimate_linear_velocity(self):
        """[核心演算法] 根據 IMU 和運動學模型，估算機身的線速度。"""
        with self.lock:
            acc_g = self.hw_state.imu_acc_g.copy()
            w_body = self.hw_state.imu_gyro_radps.copy() # 使用我們估算出的角速度
        
        body_gravity_vec = acc_g * 9.81
        world_gravity_vec = np.array([0, 0, -9.81])
        
        if np.linalg.norm(body_gravity_vec) < 1e-6: return

        try:
            rot_body_to_world, _ = Rotation.align_vectors(world_gravity_vec.reshape(1, -1), body_gravity_vec.reshape(1, -1))
        except (ValueError, np.linalg.LinAlgError): return

        w_world = rot_body_to_world.apply(w_body)

        foot_velocities_world = []
        for i in range(4):
            r_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[i])
            v_foot_world = np.cross(w_world, r_foot_in_world)
            foot_velocities_world.append(np.linalg.norm(v_foot_world))

        stance_foot_idx = np.argmin(foot_velocities_world)
        r_stance_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[stance_foot_idx])
        v_body_world_est = -np.cross(w_world, r_stance_foot_in_world)

        with self.lock:
            self.hw_state.gravity_vector_local = body_gravity_vec
            self.hw_state.lin_vel_local = rot_body_to_world.inv().apply(v_body_world_est)

    def construct_observation(self) -> np.ndarray:
        """建立提供給 ONNX 模型的觀察向量，包含從姿態角估算角速度的過程。"""
        
        # --- 【核心邏輯】從 RPY 估算角速度 (imu_gyro_radps) ---
        with self.lock:
            current_time = self.hw_state.last_update_time
            dt = current_time - self.hw_state.prev_rpy_time
            
            # 只有在時間間隔有效(>0)且這不是第一幀數據時才進行估算
            if dt > 1e-6 and self.hw_state.prev_rpy_time > 0:
                # 計算姿態角的變化量 (當前角度 - 上一幀角度)
                delta_rpy = self.hw_state.rpy_rad - self.hw_state.prev_rpy_rad
                
                # 處理 Yaw (偏航角) 的 2*pi 跳變問題。
                # 例如，從 3.14 變為 -3.14，真實變化是-0.003，但直接相減是-6.28。
                if delta_rpy[2] > np.pi: delta_rpy[2] -= 2 * np.pi
                if delta_rpy[2] < -np.pi: delta_rpy[2] += 2 * np.pi
                
                # 角速度 = 角度變化 / 時間變化
                estimated_gyro = delta_rpy / dt
                self.hw_state.imu_gyro_radps = estimated_gyro
            
            # 更新歷史數據以供下一幀計算
            self.hw_state.prev_rpy_rad = self.hw_state.rpy_rad
            self.hw_state.prev_rpy_time = current_time
        # --- 估算結束 ---

        self.estimate_linear_velocity() # 基於新的角速度估算，更新線速度
        
        with self.lock:
            self.hw_state.command = self.global_state.command * np.array(self.config.command_scaling_factors)
            
            # 建立一個字典，包含所有可能的觀察分量
            obs_list = {
                'linear_velocity': self.hw_state.lin_vel_local,
                'angular_velocity': self.hw_state.imu_gyro_radps, # 使用我們估算出的角速度
                'gravity_vector': self.hw_state.gravity_vector_local / 9.81, # 模型通常需要歸一化的重力向量
                'accelerometer': self.hw_state.imu_acc_g,
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command,
            }
            
            # 從 PolicyManager 動態獲取當前啟用模型的觀察配方
            recipe = self.policy.get_active_recipe()
            if not recipe:
                print("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
                return np.array([])

            # 根據配方要求的順序，從字典中取出對應的數據分量
            final_obs_list = [obs_list[key] for key in recipe if key in obs_list]

            # 拼接成最終的、符合模型輸入順序的觀察向量
            return np.concatenate(final_obs_list).astype(np.float32)

    def _read_from_port(self):
        """[背景執行緒] 持續從序列埠讀取數據並調用解析器。"""
        print("[硬體讀取線程已啟動] 等待來自 Teensy 的數據...")
        while self.is_running:
            if not self.ser or not self.ser.is_open:
                self.stop()
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
        default_pose_hardware = self.global_state.sim.default_pose

        while self.is_running:
            self.ai_control_enabled.wait() # 在這裡暫停，直到 enable_ai() 被呼叫
            if not self.is_running: break # 在喚醒後再次檢查，確保不是因為 stop() 而被喚醒

            loop_start_time = time.perf_counter()
            
            observation = self.construct_observation()
            if observation.size == 0:
                time.sleep(0.02)
                continue
            
            _, action_raw = self.policy.get_action_for_hardware(observation)
            
            with self.lock:
                self.hw_state.last_action[:] = action_raw
            
            final_command = default_pose_hardware + action_raw * self.global_state.tuning_params.action_scale

            # --- 將指令格式化為 "move all <12個浮點數>" ---
            action_str = ' '.join(f"{a:.4f}" for a in final_command)
            command_to_send = f"move all {action_str}\n"

            if self.ser and self.ser.is_open:
                try: 
                    self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException: 
                    self.stop()
            
            # 精確控制迴圈頻率
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)