# hardware_controller.py
import serial
import serial.tools.list_ports
import threading
import time
from logger import log
import numpy as np
from typing import TYPE_CHECKING

from event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED # 我們需要讓它能夠訂閱事件匯流排。


if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager
    from serial_communicator import SerialCommunicator


# 解釋：這是我們重構的核心。定義一個標準的數據容器，專門用來儲存從硬體
# (Teensy) 傳來的、經過單位和格式整理後的純淨數據。
# 這使得 HardwareController 內部的數據流變得極其清晰。
class RobotStateHardware:
    """【v3.3.1 新增】只儲存AI決策所需的、單位正確的數據。"""
    def __init__(self):
        # --- 直接來自Teensy的數據流 (POLICY_STREAM) ---
        self.angular_velocity_radps = np.zeros(3, dtype=np.float32)
        self.gravity_vector_norm = np.zeros(3, dtype=np.float32)
        self.accelerometer_ms2 = np.zeros(3, dtype=np.float32)
        self.pitch_rad = 0.0
        self.joint_positions_rad = np.zeros(12, dtype=np.float32)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32)
        
        # --- 由PC端維護或從其他來源獲得的狀態 ---
        self.last_action = np.zeros(12, dtype=np.float32)
        # 註：command 將在未來透過序列埠指令更新，暫時保留
        self.command = np.zeros(3, dtype=np.float32)
        
        self.last_update_time = 0.0

class HardwareController:
    """【修改版】管理與實體硬體的AI控制迴圈，從SerialCommunicator借用連接。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', serial_comm: 'SerialCommunicator'):
        """
        【v3.3.1 修改】初始化函式不再接收 global_state (SimulationState)。
        這使得此類別完全獨立，不再與主應用程式的狀態緊密耦合。
        """
        self.config = config
        self.policy = policy
        self.serial_comm = serial_comm 
        
        self.ser: serial.Serial | None = None 
        self.is_running = False 
        self.read_thread: threading.Thread | None = None 
        self.control_thread: threading.Thread | None = None 
        
        self.hw_state = RobotStateHardware() # 使用新的標準數據結構
        self.lock = threading.Lock() # 保護 self.hw_state 的讀寫安全
        
        # --- 安全的 AI 開關機制 ---
        self.ai_control_enabled = threading.Event() # 控制迴圈是否執行 AI 邏輯
        self._ai_toggle_pending = threading.Event() # 標記是否有來自外部的 AI 開關請求
        
        # 訂閱它關心的事件
        self._subscribe_to_events()
        
        log.info("✅ 硬體控制器 (v3.3.1) 已初始化，完全解耦。")


    def _subscribe_to_events(self):
        """【v3.3.1 新增】讓控制器自己訂閱關心的事件。"""
        event_bus.subscribe(EVENT_HARDWARE_AI_TOGGLE_REQUESTED, self.on_ai_toggle_requested)
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件。")

    def on_ai_toggle_requested(self):
        """
        【v3.3.1 新增】事件回呼函式。
        當收到 AI 開關請求時，它只做一件事：設定一個內部旗標。
        這是一個非阻塞操作，可以安全地被任何執行緒呼叫，從而避免 UI 卡頓。
        """
        if self.is_running:
            self._ai_toggle_pending.set()
            log.info("  -> 接收到 AI 切換請求，已設定待處理旗標。")


    def start_controller_threads(self):
        """
        【v3.3.1 修改】啟動控制器。
        移除了所有對 global_state.set_control_mode 的越權呼叫。
        現在它的職責非常單一：檢查前置條件並啟動執行緒。
        返回一個布林值來告知呼叫者是否成功啟動。
        """
        if self.is_running:
            log.info("硬體控制器已在運行中。")
            return True

        # --- 前置條件檢查 ---
        if not self.serial_comm.is_connected:
            log.error("❌ 硬體控制器啟動失敗：序列埠未連接。")
            return False

        self.ser = self.serial_comm.get_serial_connection()  # 取得序列連線實體
        if not self.ser:
            log.error("❌ 硬體控制器啟動失敗：無法從通訊器獲取有效連接。")
            return False

        # --- 接管與初始化 ---
        log.info(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True  # 告知 serial_comm 不再管理 serial

        try:
            log.info("  -> 正在命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            log.info("  -> Teensy 模式切換指令已發送。")
        except serial.SerialException as e:
            log.error(f"❌ 發送模式切換指令失敗: {e}")
            self.serial_comm.is_managed_by_hardware_controller = False # 歸還控制權
            return False

        # --- 啟動執行緒 ---
        self.ai_control_enabled.clear()
        self._ai_toggle_pending.clear()
        self.is_running = True
        
        self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
        self.read_thread.start()
        
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()  # 啟動 AI 控制執行緒

        log.info("✅ 硬體控制與讀取執行緒已成功啟動。")
        return True

    def stop_controller_threads(self):
        """
        【v3.3.1 修改】停止控制器。
        增加了更完整的清理邏輯，確保 Teensy 返回安全狀態。
        """
        if not self.is_running: 
            return
        
        log.info("正在停止硬體控制器...")
        self.is_running = False
        
        # 確保控制迴圈能退出等待
        if self.control_thread and self.control_thread.is_alive():
            self._ai_toggle_pending.set()
            self.ai_control_enabled.set() 
        
        # 等待執行緒結束
        if self.read_thread: self.read_thread.join(timeout=1)
        if self.control_thread: self.control_thread.join(timeout=1)
        
        # 在交還控制權前，命令 Teensy 恢復安全狀態
        if self.ser and self.ser.is_open:
            try:
                log.info("  -> 正在命令 Teensy 停止運動並恢復 HUMAN 遙測模式...")
                self.ser.write(b"stop\n")
                time.sleep(0.05)
                self.ser.write(b"monitor h\n")
                time.sleep(0.05)
            except serial.SerialException:
                log.warning("  -> 警告: 發送停止指令失敗，可能連接已斷開。")

        # 歸還序列埠控制權
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            log.info("  -> 序列埠控制權已交還。")
        
        self.ser = None
        log.info("✅ 硬體控制器已完全停止。")



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
        【v3.3.1 修改】專門解析來自 Teensy 的數據流，並填充到標準的 hw_state 物件中。
        """
        try:
            parts = line.split(',')
            if len(parts) != 34: return

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
        【v3.3.1 重構】從 hw_state 中直接獲取數據，並拼接成最終的 ONNX 輸入向量。
        數據來源清晰、統一。不再依賴任何外部狀態。
        """
        with self.lock:
            # 註：此處的 command 暫時為零。在 Phase 2 中，
            # 它將由從序列埠接收的指令來更新。
            # command_scaled = self.hw_state.command * np.array(self.config.command_scaling_factors)
            
            # 建立一個清晰的數據源字典
            obs_components = {
                'angular_velocity': self.hw_state.angular_velocity_radps,
                'gravity_vector': self.hw_state.gravity_vector_norm,
                'accelerometer': self.hw_state.accelerometer_ms2,
                'pitch': np.array([self.hw_state.pitch_rad]),
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command, # 使用 hw_state 內的 command
                # --- 為兼容舊模型，暫時保留的填充項 ---
                'linear_velocity': np.zeros(3), 
            }
            
        recipe = self.policy.get_active_recipe()
        if not recipe:
            log.warning("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
            return np.array([])
        
        try:
            final_obs_list = [obs_components[key] for key in recipe]
            return np.concatenate(final_obs_list).astype(np.float32)
        except KeyError as e:
            log.error(f"❌ 觀察向量構建失敗：配方中需求的 '{e}' 不在 obs_components 中。")
            return np.array([])



    def _read_from_port(self):
        """讀取執行緒，邏輯基本不變，但出錯時呼叫新的停止函式。"""
        log.info("[硬體讀取線程已啟動] 等待來自 Teensy 的 POLICY_STREAM 數據...")
        while self.is_running:
            if not self.ser or not self.ser.is_open:
                self.stop_controller_threads()
                break
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.parse_policy_stream(line) 
            except (serial.SerialException, OSError):
                log.error("❌ 錯誤：序列埠斷開連接或讀取錯誤。")
                self.stop_controller_threads()
                break
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}", exc_info=True)
                
    def _control_loop(self):
        """
        【v3.3.1 重構】控制迴圈。
        增加了對 AI 開關掛起請求的處理。
        """
        log.info("--- 硬體控制執行緒已就緒 ---")
        # 註：未來 default_pose 應從 config 讀取
        default_pose_hardware = np.zeros(12) 

        while self.is_running:
            loop_start_time = time.perf_counter()

            # --- 處理掛起的 AI 開關請求 ---
            if self._ai_toggle_pending.is_set():
                if self.ai_control_enabled.is_set():
                    log.info("⏸️ AI 控制已暫停。")
                    self.ai_control_enabled.clear()
                    try: self.ser.write(b"stop\n")
                    except serial.SerialException as e: log.error(f"發送停止指令失敗: {e}")
                else:
                    log.info("🤖 AI 控制已啟用。")
                    self.policy.reset() # 重置AI策略狀態
                    self.ai_control_enabled.set()
                self._ai_toggle_pending.clear() # 清除請求旗標

            # --- AI 決策邏輯 ---
            if self.ai_control_enabled.is_set():
                observation = self.construct_observation()
                
                if observation.size > 0:
                    # 運行AI策略獲取動作
                    _, action_raw = self.policy.get_action_for_hardware(observation)
                    
                    with self.lock:
                        self.hw_state.last_action[:] = action_raw
                    
                    # 應用 action_scale 並生成最終指令
                    # 注意：這裡的 tuning_params 應該是從 config 讀取，而不是 state
                    # 為了簡化，暫時使用 config 中的值
                    action_scale = self.config.initial_tuning_params.action_scale
                    final_command = default_pose_hardware + action_raw * action_scale
                    
                    action_str = ' '.join(f"{a:.4f}" for a in final_command)
                    command_to_send = f"move all {action_str}\n"

                    if self.ser and self.ser.is_open:
                        try:
                            self.ser.write(command_to_send.encode('utf-8'))
                        except serial.SerialException:
                            self.stop_controller_threads()
            
            # --- 迴圈時間管理 ---
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif not self.ai_control_enabled.is_set():
                # 如果 AI 未啟用，不需要高速輪詢，可以休息一下
                time.sleep(1.0 / self.config.control_freq)

