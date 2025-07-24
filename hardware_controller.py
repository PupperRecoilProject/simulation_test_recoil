# hardware_controller.py
import serial
import serial.tools.list_ports
import threading
import time
import re
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager
    from state import SimulationState
    from serial_communicator import SerialCommunicator

class RobotStateHardware:
    """儲存從實體機器人收到的所有狀態數據。"""
    def __init__(self):
        self.imu_gyro_radps = np.zeros(3, dtype=np.float32) # IMU 角速度 (rad/s)
        self.imu_acc_g = np.zeros(3, dtype=np.float32) # IMU 加速度 (g)
        self.joint_positions_rad = np.zeros(12, dtype=np.float32) # 關節角度 (rad)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32) # 關節角速度 (rad/s)
        self.lin_vel_local = np.zeros(3, dtype=np.float32) # 估算的局部座標系線速度
        self.gravity_vector_local = np.zeros(3, dtype=np.float32) # 估算的局部座標系重力向量
        self.last_action = np.zeros(12, dtype=np.float32) # 上一次的動作指令
        self.command = np.zeros(3, dtype=np.float32) # 使用者下達的指令 [vy, vx, wz]
        self.timestamp_ms = 0 # 硬體時間戳 (ms)
        self.robot_mode = "N/A" # 機器人模式 (來自硬體)
        self.is_calibrated = False # IMU 是否校準
        self.rpy_rad = np.zeros(3, dtype=np.float32) # 姿態角 Roll, Pitch, Yaw (rad)
        self.target_current_ma = np.zeros(12, dtype=np.float32) # 目標電流 (mA)
        self.actual_current_ma = np.zeros(12, dtype=np.float32) # 實際電流 (mA)
        self.prev_rpy_rad = np.zeros(3, dtype=np.float32) # 上一次的 RPY，用於計算角速度
        self.prev_rpy_time = 0.0 # 上一次收到 RPY 的時間
        self.last_update_time = 0.0 # 最後一次成功更新數據的時間

class HardwareController:
    """【修改版】管理與實體硬體的AI控制迴圈，從SerialCommunicator借用連接。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', global_state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        """【修改】初始化時接收 SerialCommunicator 的參考。"""
        self.config = config # 儲存應用程式設定
        self.policy = policy # 儲存策略管理器
        self.global_state = global_state # 儲存全域狀態
        self.serial_comm = serial_comm # 【新增】儲存序列埠通訊器的參考
        
        self.ser = None #序列埠物件
        self.is_running = False # 控制執行緒是否運行的旗標
        self.read_thread = None # 讀取執行緒
        self.control_thread = None # 控制執行緒
        
        self.hw_state = RobotStateHardware() # 實例化硬體狀態物件
        self.lock = threading.Lock() # 建立執行緒鎖，保護 hw_state 的讀寫
        self.ai_control_enabled = threading.Event() # 使用 Event 物件來控制 AI 是否啟用

        # 機器人本體座標系下四個足端的預設位置
        self.foot_positions_in_body = np.array([
            [-0.0804, -0.1759, -0.1964],
            [ 0.0806, -0.1759, -0.1964],
            [-0.0804,  0.0239, -0.1964],
            [ 0.0806,  0.0239, -0.1964],
        ], dtype=np.float32)
        print("✅ 硬體控制器已初始化。")

    def connect_and_start(self) -> bool:
        """【核心重構】不再自己建立連接，而是從 SerialCommunicator 獲取已建立的連接。"""
        if self.is_running: # 如果已經在運行
            print("硬體控制器已在運行中。")
            return True
            
        if not self.serial_comm.is_connected: # 如果序列埠未連接
            print("❌ 硬體模式錯誤：請先按 'U' 鍵連接序列埠。")
            return False
        
        self.ser = self.serial_comm.get_serial_connection() # 從通訊器獲取序列埠物件
        if not self.ser: # 如果獲取失敗
            print("❌ 硬體模式錯誤：無法從 SerialCommunicator 獲取有效的序列埠連接。")
            return False
            
        print(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True # 通知通訊器，控制權已移交
        
        self.is_running = True # 設定運行旗標
        self.read_thread = threading.Thread(target=self._read_from_port, daemon=True) # 建立讀取執行緒
        self.read_thread.start() # 啟動讀取執行緒
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True) # 建立控制執行緒
        self.control_thread.start() # 啟動控制執行緒
        
        print("✅ 硬體控制執行緒已啟動。")
        return True

    def stop(self):
        """【修改】停止時，將序列埠的控制權交還，但不關閉連接。"""
        if not self.is_running: return # 如果未運行，直接返回
        
        print("正在停止硬體控制器...")
        self.is_running = False # 清除運行旗標
        self.disable_ai() # 確保 AI 已禁用
        self.ai_control_enabled.set() # 喚醒可能在等待中的控制執行緒，讓它能夠退出
        
        if self.control_thread and self.control_thread.is_alive(): self.control_thread.join(timeout=1) # 等待控制執行緒結束
        if self.read_thread and self.read_thread.is_alive(): self.read_thread.join(timeout=1) # 等待讀取執行緒結束
        
        if self.serial_comm: # 如果通訊器存在
            self.serial_comm.is_managed_by_hardware_controller = False # 將序列埠控制權交還
            print("序列埠控制權已交還。")
        
        self.ser = None # 清空序列埠物件
        print("硬體控制器已完全停止。")
        
    def enable_ai(self):
        """啟用 AI 控制。"""
        if not self.is_running: # 如果控制器未運行
            print("無法啟用 AI：硬體控制器未運行。")
            return
        print("🤖 AI 控制已啟用。")
        self.policy.reset() # 重置 AI 策略的內部狀態（如歷史觀測）
        self.ai_control_enabled.set() # 設定 Event，允許控制迴圈運行
        self.global_state.hardware_ai_is_active = True # 更新全域狀態

    def disable_ai(self):
        """禁用 AI 控制。"""
        print("⏸️ AI 控制已暫停。")
        self.ai_control_enabled.clear() # 清除 Event，使控制迴圈暫停
        self.global_state.hardware_ai_is_active = False # 更新全域狀態
        if self.ser and self.ser.is_open: # 如果序列埠可用
            try: self.ser.write(b"stop\n") # 發送 "stop" 指令讓機器人停止運動
            except serial.SerialException as e: print(f"發送停止指令失敗: {e}")

    def parse_teensy_data(self, line: str):
        """【核心修正】重構此函式，使其更具彈性，並能提供有用的除錯資訊。"""
        try:
            parts = line.split(',') # 使用逗號分割字串
            
            # 【核心修正】檢查欄位數，如果數量不符，則在終端機印出提示，而不是默默忽略
            if len(parts) != 57:
                # 這個 print 非常重要，它會告訴您硬體傳來的數據格式到底是什麼樣的
                print(f"[硬體數據除錯] 忽略格式不符的行 (欄位數: {len(parts)}): {line}")
                return # 忽略此行數據

            with self.lock: # 鎖定狀態物件，防止多執行緒衝突
                current_time = time.time() # 獲取當前時間
                self.hw_state.timestamp_ms = int(parts[0]) # 解析時間戳
                self.hw_state.robot_mode = parts[1] # 解析機器人模式
                self.hw_state.is_calibrated = (parts[2] == '1') # 解析校準狀態
                rpy_deg = np.array(parts[3:6], dtype=np.float32) # 解析 RPY 角度 (度)
                self.hw_state.rpy_rad = rpy_deg * (np.pi / 180.0) # 轉換為弧度
                self.hw_state.imu_acc_g = np.array(parts[6:9], dtype=np.float32) # 解析加速度
                self.hw_state.joint_positions_rad = np.array(parts[9:21], dtype=np.float32) # 解析關節角度
                self.hw_state.joint_velocities_radps = np.array(parts[21:33], dtype=np.float32) # 解析關節角速度
                self.hw_state.target_current_ma = np.array(parts[33:45], dtype=np.float32) # 解析目標電流
                self.hw_state.actual_current_ma = np.array(parts[45:57], dtype=np.float32) # 解析實際電流
                
                # 在成功解析後，立即更新最後更新時間
                self.hw_state.last_update_time = current_time
                
        except (ValueError, IndexError) as e: # 捕捉解析時可能發生的錯誤
            print(f"❌ 解析硬體數據時出錯: {e} | 原始數據: {line}")

    def estimate_linear_velocity(self):
        """根據 IMU 和足端位置估算機器人的線速度。"""
        with self.lock: # 鎖定，安全地複製需要的狀態
            acc_g = self.hw_state.imu_acc_g.copy()
            w_body = self.hw_state.imu_gyro_radps.copy()
            
        body_gravity_vec = acc_g * 9.81 # 將 g 單位下的加速度轉為 m/s^2
        world_gravity_vec = np.array([0, 0, -9.81]) # 定義世界座標系下的重力向量
        
        if np.linalg.norm(body_gravity_vec) < 1e-6: return # 避免除以零
        
        try:
            # 計算從身體座標系到世界座標系的旋轉矩陣
            rot_body_to_world, _ = Rotation.align_vectors(world_gravity_vec.reshape(1, -1), body_gravity_vec.reshape(1, -1))
        except (ValueError, np.linalg.LinAlgError): return # 處理對齊失敗的情況
        
        w_world = rot_body_to_world.apply(w_body) # 將局部角速度轉換到世界座標系
        
        foot_velocities_world = [] # 儲存每個腳在世界座標系下的速度大小
        for i in range(4): # 遍歷四隻腳
            r_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[i]) # 將腳的位置向量轉到世界系
            v_foot_world = np.cross(w_world, r_foot_in_world) # 計算因旋轉產生的速度 v = ω x r
            foot_velocities_world.append(np.linalg.norm(v_foot_world)) # 計算速度大小並儲存
            
        stance_foot_idx = np.argmin(foot_velocities_world) # 假設速度最小的腳是支撐腳
        r_stance_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[stance_foot_idx]) # 獲取支撐腳的位置向量
        
        # 假設支撐腳速度為0，反推出身體中心的速度
        v_body_world_est = -np.cross(w_world, r_stance_foot_in_world)
        
        with self.lock: # 鎖定，更新估算出的狀態
            self.hw_state.gravity_vector_local = body_gravity_vec # 更新局部重力向量
            self.hw_state.lin_vel_local = rot_body_to_world.inv().apply(v_body_world_est) # 將估算出的世界線速度轉回局部座標系並更新

    def construct_observation(self) -> np.ndarray:
        """根據策略配方，組合所有需要的感測器數據以建立 ONNX 模型的輸入向量。"""
        with self.lock: # 鎖定狀態
            current_time = self.hw_state.last_update_time # 使用最新的成功更新時間
            dt = current_time - self.hw_state.prev_rpy_time # 計算時間差
            
            # 根據 RPY 姿態角的變化來估算角速度 (作為陀螺儀的備份或補充)
            if dt > 1e-6 and self.hw_state.prev_rpy_time > 0:
                delta_rpy = self.hw_state.rpy_rad - self.hw_state.prev_rpy_rad # 計算姿態角變化
                # 處理 yaw 角的 2π 跳變問題
                if delta_rpy[2] > np.pi: delta_rpy[2] -= 2 * np.pi
                if delta_rpy[2] < -np.pi: delta_rpy[2] += 2 * np.pi
                estimated_gyro = delta_rpy / dt # 角速度 = 角度變化 / 時間差
                self.hw_state.imu_gyro_radps = estimated_gyro # 更新角速度狀態
                
            self.hw_state.prev_rpy_rad = self.hw_state.rpy_rad # 更新上一次的姿態角
            self.hw_state.prev_rpy_time = current_time # 更新上一次的時間
            
        self.estimate_linear_velocity() # 呼叫線速度估算函式
        
        with self.lock: # 再次鎖定
            # 組合使用者指令
            self.hw_state.command = self.global_state.command * np.array(self.config.command_scaling_factors)
            # 定義一個字典，包含所有可能的觀察元件
            obs_list = {
                'linear_velocity': self.hw_state.lin_vel_local,
                'angular_velocity': self.hw_state.imu_gyro_radps,
                'gravity_vector': self.hw_state.gravity_vector_local / 9.81, # 標準化重力向量
                'accelerometer': self.hw_state.imu_acc_g,
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command,
            }
            recipe = self.policy.get_active_recipe() # 從策略管理器獲取當前模型需要的配方
            if not recipe: # 如果配方不存在
                print("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
                return np.array([]) # 返回空陣列
            # 根據配方，從字典中挑選出需要的數據並拼接
            final_obs_list = [obs_list[key] for key in recipe if key in obs_list]
            return np.concatenate(final_obs_list).astype(np.float32) # 拼接成單一的 numpy 陣列

    def _read_from_port(self):
        """[背景讀取執行緒] 持續從序列埠讀取數據並呼叫解析函式。"""
        print("[硬體讀取線程已啟動] 等待來自 Teensy 的數據...")
        while self.is_running: # 只要控制器在運行
            if not self.ser or not self.ser.is_open: # 如果序列埠失效
                self.stop(); break # 停止控制器並退出迴圈
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip() # 讀取一行數據
                if line: # 如果讀取到內容
                    self.parse_teensy_data(line) # 無論AI是否啟用，都持續解析數據
            except (serial.SerialException, OSError): # 捕捉序列埠錯誤
                print("❌ 錯誤：序列埠斷開連接或讀取錯誤。"); self.stop(); break
            except Exception as e: # 捕捉未知錯誤
                print(f"❌ _read_from_port 發生未知錯誤: {e}")
                
    def _control_loop(self):
        """[背景控制執行緒] AI 控制的核心迴圈。"""
        print("\n--- 硬體控制線程已就緒，等待 AI 啟用 ---")
        default_pose_hardware = self.global_state.sim.default_pose # 獲取預設站立姿態
        while self.is_running: # 只要控制器在運行
            self.ai_control_enabled.wait() # 等待 AI 啟用信號，如果未啟用，會在此阻塞
            if not self.is_running: break # 如果在等待時控制器被停止，則退出
            
            loop_start_time = time.perf_counter() # 記錄迴圈開始時間
            
            observation = self.construct_observation() # 建立觀察向量
            if observation.size == 0: # 如果觀察向量為空
                time.sleep(0.02); continue # 短暫休眠後繼續
                
            _, action_raw = self.policy.get_action_for_hardware(observation) # 傳入觀察，獲取 AI 模型的原始動作輸出
            
            with self.lock: # 鎖定
                self.hw_state.last_action[:] = action_raw # 更新上一次的動作
            
            # 計算最終要發送給馬達的角度 = 預設姿態 + (模型輸出 * 動作縮放比例)
            final_command = default_pose_hardware + action_raw * self.global_state.tuning_params.action_scale
            action_str = ' '.join(f"{a:.4f}" for a in final_command) # 將角度陣列轉換為字串
            command_to_send = f"move all {action_str}\n" # 組合最終的序列埠指令
            
            if self.ser and self.ser.is_open: # 如果序列埠可用
                try: self.ser.write(command_to_send.encode('utf-8')) # 發送指令
                except serial.SerialException: self.stop() # 如果發送失敗，則停止控制器
            
            loop_duration = time.perf_counter() - loop_start_time # 計算迴圈執行時間
            sleep_time = (1.0 / self.config.control_freq) - loop_duration # 計算需要休眠的時間以維持固定的控制頻率
            if sleep_time > 0: time.sleep(sleep_time) # 如果還有剩餘時間，則休眠