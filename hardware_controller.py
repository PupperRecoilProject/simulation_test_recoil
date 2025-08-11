# hardware_controller.py
import serial
import threading
import time
from logger import log
import numpy as np
from typing import TYPE_CHECKING
from queue import Queue, Empty # 【v4.0.2 新增】導入線程安全的隊列
from enum import Enum, auto     # 【v4.0.2 新增】導入枚舉，用於定義狀態機

from event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED # 我們需要讓它能夠訂閱事件匯流排。


if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager
    from serial_communicator import SerialCommunicator
    from state import SimulationState # 【v4.0.2 新增】需要 state 來更新狀態


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

# 【v4.0.2 新增】定義內部命令的類型
class HWCommand(Enum):
    START = auto()
    STOP = auto()
    TOGGLE_AI = auto()

# 【v4.0.2 新增】定義 HardwareController 的內部狀態機
class HWState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAILED = auto()


class HardwareController:
    """【修改版】管理與實體硬體的AI控制迴圈，從SerialCommunicator借用連接。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        """
        【v3.3.1 修改】初始化函式不再接收 global_state (SimulationState)。
        這使得此類別完全獨立，不再與主應用程式的狀態緊密耦合。
        """
        self.config = config
        self.policy = policy
        self.state = state # 【v4.0.2 新增】儲存 state 的參考，用於更新硬體狀態
        self.serial_comm = serial_comm 
        
        self.ser: serial.Serial | None = None 
        # 【v4.0.2 TO DELETE】is_running 布林旗標是舊架構的產物。
        # self.is_running = False 
        self.read_thread: threading.Thread | None = None 
        self.control_thread: threading.Thread | None = None 
        
        self.hw_state_data = RobotStateHardware() # 舊的數據容器，改名以區分
        self.lock = threading.Lock() # 保護 self.hw_state 的讀寫安全
        
        # --- 【v4.0.2 核心改造】 ---
        self._is_running_event = threading.Event() # 用於控制執行緒主迴圈是否繼續
        self.command_queue = Queue()               # 命令板 (線程安全)
        self.internal_state = HWState.STOPPED      # 狀態機，初始為停止狀態
        self.last_state_change_time = time.time()  # 用於超時判斷

        # 【v4.0.2 修改】AI控制現在由命令隊列管理，不再需要獨立的Event
        self.ai_control_active = False

        self._subscribe_to_events()
        log.info("✅ 硬體控制器 (v4.0.2 異步版) 已初始化。")


    def _subscribe_to_events(self):
        """訂閱 AI 切換請求事件。"""
        # 這個事件的回呼，現在只會向命令隊列發送一個命令
        event_bus.subscribe(EVENT_HARDWARE_AI_TOGGLE_REQUESTED, 
                            lambda: self.command_queue.put(HWCommand.TOGGLE_AI))
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件 (將發送至內部隊列)。")

    # 【v4.0.2 TO DELETE】on_ai_toggle_requested
    # 這個函式是舊架構的遺留物。它操作的 _ai_toggle_pending 已經在 __init__ 中被移除。
    # 它的功能已經被上面 _subscribe_to_events 中的 lambda 函式完全取代。

    # 把舊的 start/stop_controller_threads 完全替換掉了
    def request_start(self) -> None:
        """【v4.0.2 新增】外部呼叫的非阻塞啟動請求。"""
        if self.internal_state in [HWState.STOPPED, HWState.FAILED]:
            log.info("收到啟動請求，向控制執行緒發送 START 命令。")
            self._start_threads_if_not_alive() # 確保執行緒本身在運行
            self.command_queue.put(HWCommand.START)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略啟動請求。")

    def request_stop(self) -> None:
        """【v4.0.2 新增】外部呼叫的非阻塞停止請求。"""
        if self.internal_state == HWState.RUNNING:
            log.info("收到停止請求，向控制執行緒發送 STOP 命令。")
            self.command_queue.put(HWCommand.STOP)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略停止請求。")
    
    def shutdown(self):
        """【v4.0.2 新增】應用程式關閉時的強制清理。"""
        self._is_running_event.clear() # 請求執行緒退出
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        log.info("硬體控制器執行緒已關閉。")

    def _start_threads_if_not_alive(self):
        """【v4.0.2 新增】一個內部輔助函式，確保控制和讀取執行緒存在並運行。"""
        self._is_running_event.set()
        if not self.control_thread or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            log.info("控制執行緒已啟動。")
        
        if not self.read_thread or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            log.info("讀取執行緒已啟動。")

    # 【v4.0.2 TO DELETE】enable_ai / disable_ai
    # 這兩個函式是舊的、直接命令式的 API。它們的功能現在由統一的、
    # 異步的 _execute_toggle_ai 函式處理，該函式由 TOGGLE_AI 命令觸發。
    # 保留它們會造成 API 的混亂和潛在的錯誤。


    def parse_policy_stream(self, line: str):
        """
        【v3.3.1 修改】專門解析來自 Teensy 的數據流，並填充到標準的 hw_state 物件中。
        """
        try:
            parts = line.split(',')
            if len(parts) != 34: return

            data_vec = np.array(parts, dtype=np.float32)

            with self.lock:
                self.hw_state_data.angular_velocity_radps[:] = data_vec[0:3]
                self.hw_state_data.gravity_vector_norm[:] = data_vec[3:6]
                self.hw_state_data.accelerometer_ms2[:] = data_vec[6:9]
                self.hw_state_data.pitch_rad = data_vec[9]
                self.hw_state_data.joint_positions_rad[:] = data_vec[10:22]
                self.hw_state_data.joint_velocities_radps[:] = data_vec[22:34]
                self.hw_state_data.last_update_time = time.time()

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
                'angular_velocity': self.hw_state_data.angular_velocity_radps,
                'gravity_vector': self.hw_state_data.gravity_vector_norm,
                'accelerometer': self.hw_state_data.accelerometer_ms2,
                'pitch': np.array([self.hw_state_data.pitch_rad]),
                'joint_positions': self.hw_state_data.joint_positions_rad,
                'joint_velocities': self.hw_state_data.joint_velocities_radps,
                'last_action': self.hw_state_data.last_action,
                'commands': self.hw_state_data.command, # 使用 hw_state 內的 command
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
        """【v4.0.2 TO FIX】讀取執行緒需要使用新的執行緒控制和錯誤處理機制。"""
        log.info("[硬體讀取線程已啟動] 等待來自 Teensy 的 POLICY_STREAM 數據...")
        # 【FIX 1】迴圈條件必須使用新的 _is_running_event
        while self._is_running_event.is_set():
            # 【FIX 2】當控制器不在 RUNNING 狀態時，不應該嘗試讀取，以避免競爭
            if self.internal_state != HWState.RUNNING or not self.ser or not self.ser.is_open:
                time.sleep(0.1) # 處於非活動狀態時，低頻率檢查即可
                continue

            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.parse_policy_stream(line) 
            except (serial.SerialException, OSError):
                log.error("❌ 錯誤：序列埠斷開連接或讀取錯誤。將狀態設置為 FAILED。")
                # 【FIX 3】出錯時，不能呼叫不存在的 stop_controller_threads()，
                # 而是應該切換狀態機到 FAILED 狀態，讓主控制迴圈來處理後續。
                self._set_internal_state(HWState.FAILED)
                break # 退出讀取迴圈
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}", exc_info=True)
                self._set_internal_state(HWState.FAILED)
                break


    def _set_internal_state(self, new_state: HWState):
        """【v4.0.2 新增】一個安全的狀態切換函式，包含日誌和時間戳。"""
        if self.internal_state != new_state:
            log.info(f"硬體控制器狀態: {self.internal_state.name} -> {new_state.name}")
            self.internal_state = new_state
            self.last_state_change_time = time.time()
            # 【v4.0.2 新增】將硬體控制器的運行狀態同步到全局 State
            with self.state.lock:
                self.state.hardware_is_running = (new_state == HWState.RUNNING)


    def _control_loop(self):
        """【v4.0.2 重構版】控制執行緒，管理狀態機並執行命令。"""
        log.info("--- 硬體控制執行緒已就緒，等待命令 ---")

        while self._is_running_event.is_set():
            # 1. 檢查命令隊列
            try:
                command: HWCommand = self.command_queue.get_nowait()
                # 處理收到的命令
                if command == HWCommand.START and self.internal_state in [HWState.STOPPED, HWState.FAILED]:
                    self._execute_start()
                elif command == HWCommand.STOP and self.internal_state == HWState.RUNNING:
                    self._execute_stop()
                elif command == HWCommand.TOGGLE_AI and self.internal_state == HWState.RUNNING:
                    self._execute_toggle_ai()
            except Empty:
                # 隊列為空，什麼都不做
                pass

            # 2. 根據當前狀態執行循環任務
            if self.internal_state == HWState.RUNNING and self.ai_control_active:
                self._perform_ai_step()
            
            # 3. 迴圈延遲
            # 即使 AI 未激活，也保持一定的輪詢頻率以響應命令
            time.sleep(1.0 / self.config.control_freq)


    def _execute_start(self):
        """【v4.0.2 新增】執行啟動流程，包含所有可能阻塞的操作。"""
        self._set_internal_state(HWState.STARTING)

        if not self.serial_comm.is_connected:
            log.error("❌ 硬體啟動失敗：序列埠未連接。")
            self._set_internal_state(HWState.FAILED)
            return

        self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            log.error("❌ 硬體啟動失敗：無法從通訊器獲取有效連接。")
            self._set_internal_state(HWState.FAILED)
            return

        log.info(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True

        try:
            log.info("  -> 命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            log.info("  -> Teensy 模式切換指令已發送。")
            self._set_internal_state(HWState.RUNNING)
            # 預設不開啟 AI，等待用戶指令
            self.ai_control_active = False
            with self.state.lock: self.state.hardware_ai_is_active = False

        except serial.SerialException as e:
            log.error(f"❌ 發送模式切換指令失敗: {e}")
            self.serial_comm.is_managed_by_hardware_controller = False
            self._set_internal_state(HWState.FAILED)

    def _execute_stop(self):
        """【v4.0.2 新增】執行停止流程。"""
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False # 停止時必須禁用AI
        with self.state.lock: self.state.hardware_ai_is_active = False

        if self.ser and self.ser.is_open:
            try:
                log.info("  -> 命令 Teensy 停止運動並恢復 HUMAN 遙測模式...")
                self.ser.write(b"stop\n"); time.sleep(0.05)
                self.ser.write(b"monitor h\n"); time.sleep(0.05)
            except serial.SerialException as e:
                log.warning(f"  -> 警告: 發送停止指令失敗: {e}")
        
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            log.info("  -> 序列埠控制權已交還。")
        
        self.ser = None
        self._set_internal_state(HWState.STOPPED)

    def _execute_toggle_ai(self):
        """【v4.0.2 新增】執行切換AI的邏輯。"""
        self.ai_control_active = not self.ai_control_active
        with self.state.lock:
            self.state.hardware_ai_is_active = self.ai_control_active
        
        log.info(f"🤖 硬體 AI 控制已 {'啟用' if self.ai_control_active else '暫停'}.")
        
        if self.ai_control_active:
            self.policy.reset() # 啟用時重置策略狀態
        else:
            if self.ser and self.ser.is_open:
                try: self.ser.write(b"stop\n") # 暫停時發送停止指令
                except serial.SerialException as e: log.error(f"發送停止指令失敗: {e}")

    def _perform_ai_step(self):
        """【v4.0.2 新增】執行單步 AI 計算與控制。"""
        # (此處的邏輯是從舊的 _control_loop 中提取出來的，保持不變)
        observation = self.construct_observation()
        if observation.size > 0:
            _, action_raw = self.policy.get_action_for_hardware(observation)
            
            with self.lock:
                self.hw_state_data.last_action[:] = action_raw
            
            action_scale = self.config.initial_tuning_params.action_scale
            default_pose_hardware = np.zeros(12) # 應從config讀取
            final_command = default_pose_hardware + action_raw * action_scale
            
            action_str = ' '.join(f"{a:.4f}" for a in final_command)
            command_to_send = f"move all {action_str}\n"

            if self.ser and self.ser.is_open:
                try: self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException:
                    log.error("AI 步驟中發送指令失敗，連接可能已斷開。")
                    self._set_internal_state(HWState.FAILED)


