# hardware_controller.py
import serial
import serial.tools.list_ports
import threading
import time
from logger import log
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
    """【修改】只儲存AI決策所需的、單位正確的數據。"""
    def __init__(self):
        # 這些是直接從Teensy的POLICY_STREAM解析來的數據
        self.angular_velocity_radps = np.zeros(3, dtype=np.float32)
        self.gravity_vector_norm = np.zeros(3, dtype=np.float32)
        self.accelerometer_ms2 = np.zeros(3, dtype=np.float32)
        self.pitch_rad = 0.0
        self.joint_positions_rad = np.zeros(12, dtype=np.float32)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32)
        
        # 這些是PC端自己維護的狀態
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        
        self.last_update_time = 0.0

class HardwareController:
    """【修改版】管理與實體硬體的AI控制迴圈，從SerialCommunicator借用連接。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', global_state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        """【修改】初始化時接收 SerialCommunicator 的參考。"""
        self.config = config
        self.policy = policy
        self.global_state = global_state
        self.serial_comm = serial_comm 
        
        self.ser = None 
        self.is_running = False 
        self.read_thread = None 
        self.control_thread = None 
        
        self.hw_state = RobotStateHardware()
        self.lock = threading.Lock()
        self.ai_control_enabled = threading.Event()

        self.foot_positions_in_body = np.array([
            [-0.0804, -0.1759, -0.1964],
            [ 0.0806, -0.1759, -0.1964],
            [-0.0804,  0.0239, -0.1964],
            [ 0.0806,  0.0239, -0.1964],
        ], dtype=np.float32)
        log.info("✅ 硬體控制器已初始化。")

    def start_controller_threads(self):
        """在啟動背景執行緒前，自動命令Teensy切換到POLICY_STREAM模式。"""
        if self.is_running:
            log.info("硬體控制器已在運行中。")
            return

        # 原本的程式碼會在檢查失敗時，強制將模式設回 WALKING。
        # 我們將其移除，讓 UI 層來決定是否允許用戶進入此模式。
        if not self.serial_comm.is_connected:
            log.error("❌ 硬體模式錯誤：請先連接序列埠。")
            # self.global_state.set_control_mode("WALKING") # <--【刪除】
            return

        self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            log.error("❌ 硬體模式錯誤：無法取得有效序列埠連接。")
            # self.global_state.set_control_mode("WALKING") # <--【刪-除】
            return

        # 接管序列埠控制權
        log.info(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True

        # 自動切換 Teensy 到 AI 決策流模式
        try:
            log.info("  -> 正在命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) # 給Teensy一點時間切換模式並清空緩衝區
            self.ser.reset_input_buffer()
            log.info("  -> Teensy 模式切換指令已發送。")
        except serial.SerialException as e:
            log.error(f"❌ 發送模式切換指令失敗: {e}")
            self.stop_controller_threads() # 使用現有的停止函式來清理
            return

        # 後續的執行緒啟動邏輯
        self.ai_control_enabled.clear()
        with self.global_state.lock:
            self.global_state.hardware_ai_is_active = False

        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
        self.read_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        with self.global_state.lock:
            self.global_state.hardware_is_connected = True
        log.info("✅ 硬體控制執行緒已啟動。")

    def stop_controller_threads(self):
        """【修改】在停止執行緒後，自動命令Teensy恢復到安全的人類友好模式。"""
        if not self.is_running: return
        
        log.info("正在停止硬體控制器...")
        self.is_running = False
        self.disable_ai()
        self.ai_control_enabled.set() 
        
        # 在交還控制權前，命令Teensy恢復安全狀態
        if self.ser and self.ser.is_open:
            try:
                log.info("  -> 正在命令 Teensy 停止運動並恢復 HUMAN 遙測模式...")
                self.ser.write(b"stop\n")
                time.sleep(0.05)
                self.ser.write(b"monitor h\n")
                time.sleep(0.05)
            except serial.SerialException:
                log.warning("  -> 警告: 發送停止指令失敗，可能連接已斷開。")

        # 所有執行緒清理和交還控制權的邏輯
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            log.info("序列埠控制權已交還。")
        
        self.ser = None
        log.info("硬體控制器已完全停止。")
        
    def enable_ai(self):
        if not self.is_running:
            log.info("無法啟用 AI：硬體控制器未運行。")
            return
        log.info("🤖 AI 控制已啟用。")
        self.policy.reset()
        self.ai_control_enabled.set()
        self.global_state.hardware_ai_is_active = True

    def disable_ai(self):
        log.info("⏸️ AI 控制已暫停。")
        self.ai_control_enabled.clear()
        self.global_state.hardware_ai_is_active = False
        if self.is_running and self.ser and self.ser.is_open: # 增加 is_running 判斷
            try: self.ser.write(b"stop\n")
            except serial.SerialException as e:
                log.error(f"發送停止指令失敗: {e}")

    def parse_policy_stream(self, line: str):
        """
        專門解析來自 Teensy 'monitor p' 模式的 34 維數據流。
        """
        try:
            parts = line.split(',')
            if len(parts) != 34:
                return

            data_vec = np.array(parts, dtype=np.float32)

            with self.lock:
                self.hw_state.angular_velocity_radps[:] = data_vec[0:3]
                self.hw_state.gravity_vector_norm[:] = data_vec[3:6]
                self.hw_state.accelerometer_ms2[:] = data_vec[6:9]
                self.hw_state.pitch_rad = data_vec[9]
                self.hw_state.joint_positions_rad[:] = data_vec[10:22]
                self.hw_state.joint_velocities_radps[:] = data_vec[22:34]
                self.hw_state.last_update_time = time.time()

        except (ValueError, IndexError) as e:
            log.error(f"❌ 解析 POLICY_STREAM 時出錯: {e} | 原始數據長度: {len(parts)}")

    def construct_observation(self) -> np.ndarray:
        """
        [全新重構] 從 hw_state 中直接獲取數據，並拼接成最終的 ONNX 輸入向量。
        不再需要任何客戶端的估算。
        """
        with self.lock:
            # 【保留】從全域狀態獲取用戶指令
            self.hw_state.command = self.global_state.command * np.array(self.config.command_scaling_factors)
            
            # 【修改】建立一個全新的、數據源清晰的字典
            obs_list = {
                # --- 來自Teensy的高品質數據 ---
                'angular_velocity': self.hw_state.angular_velocity_radps,
                'gravity_vector': self.hw_state.gravity_vector_norm,
                'accelerometer': self.hw_state.accelerometer_ms2,
                'pitch': np.array([self.hw_state.pitch_rad]), # 確保是1維向量
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                
                # --- PC端自行維護的狀態 ---
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command,

                # --- 為了兼容性，保留舊的鍵，用零填充 ---
                'linear_velocity': np.zeros(3), 
            }
            
            # 【保留】由配方驅動的拼接邏輯
            recipe = self.policy.get_active_recipe()
            if not recipe:
                log.warning("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
                return np.array([])
            
            try:
                final_obs_list = [obs_list[key] for key in recipe]
                return np.concatenate(final_obs_list).astype(np.float32)
            except KeyError as e:
                log.error(f"❌ 觀察向量構建失敗：配方中需求的 '{e}' 不在 obs_list 中。")
                return np.array([])

    def _read_from_port(self):
        log.info("[硬體讀取線程已啟動] 等待來自 Teensy 的 POLICY_STREAM 數據...")
        while self.is_running:
            if not self.ser or not self.ser.is_open:
                # 【修改】確保呼叫正確的停止函式
                self.stop_controller_threads()
                break
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    # 【修改】呼叫新的解析器
                    self.parse_policy_stream(line) 
            except (serial.SerialException, OSError):
                log.error("❌ 錯誤：序列埠斷開連接或讀取錯誤。")
                self.stop_controller_threads()
                break
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}")
                
    def _control_loop(self):
        """【保留大部分邏輯】此迴圈現在是硬體指令的唯一來源，並能感知 JOINT_TEST 模式。"""
        log.info("\n--- 硬體控制執行緒已就緒 ---")
        # 【保留】 default_pose_hardware 保持不變
        default_pose_hardware = self.global_state.sim.default_pose
        while self.is_running:
            loop_start_time = time.perf_counter()
            command_to_send = None

            # 【保留】 JOINT_TEST 模式的指令生成邏輯不變
            if self.global_state.control_mode == "JOINT_TEST":
                final_command = default_pose_hardware + self.global_state.joint_test_offsets
                action_str = ' '.join(f"{a:.4f}" for a in final_command)
                command_to_send = f"move all {action_str}\n"

            # 【保留】 HARDWARE_MODE 的指令生成邏輯
            elif self.global_state.control_mode == "HARDWARE_MODE":
                # 【保留】等待AI啟用信號
                self.ai_control_enabled.wait()
                if not self.is_running: break

                # 【保留】呼叫新的 construct_observation 函式
                # 這個函式現在會使用 RobotStateHardware 中更新過的高品質數據
                observation = self.construct_observation()
                if observation.size == 0:
                    time.sleep(0.02); continue
                
                # 【保留】運行AI策略獲取動作
                _, action_raw = self.policy.get_action_for_hardware(observation)
                with self.lock:
                    # 【保留】更新 last_action
                    self.hw_state.last_action[:] = action_raw
                
                # 【保留】生成發送給Teensy的文字指令
                final_command = default_pose_hardware + action_raw * self.global_state.tuning_params.action_scale
                action_str = ' '.join(f"{a:.4f}" for a in final_command)
                command_to_send = f"move all {action_str}\n"

            # 【保留】發送指令邏輯
            if command_to_send and self.ser and self.ser.is_open:
                try:
                    self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException:
                    # 【修改】確保呼叫一致的停止函式
                    self.stop_controller_threads() # <--- 這裡從 self.stop() 改為 self.stop_controller_threads()
            
            # 【保留】迴圈時間管理
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
