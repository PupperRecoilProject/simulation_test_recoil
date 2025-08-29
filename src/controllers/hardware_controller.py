# src/controllers/hardware_controller.py

# 【v4.3.2 修改】 調整 import，移除不再需要的 Enum
import serial
import threading
import time
from src.core.logger import log
import numpy as np
from typing import TYPE_CHECKING
from queue import Queue, Empty
from enum import Enum, auto

from src.core.event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED

if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.hardware.policy import PolicyManager
    from src.hardware.serial_communicator import SerialCommunicator
    from src.core.state import SimulationState

# 【v4.3.2 刪除】 RobotStateHardware 類別
# 這個類別的所有狀態都已遷移到 SimulationState 的 raw_... 屬性中，
# 以實現統一的數據流。

class HWCommand(Enum):
    START = auto()
    STOP = auto()
    TOGGLE_AI = auto()

class HWState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAILED = auto()

class HardwareController:
    """【v4.3.2 修改】管理硬體AI控制迴圈，作為原始數據提供者。"""
    
    # 【v4.3.2 修改】 __init__ 方法
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        self.config = config
        self.policy = policy
        self.state = state
        self.serial_comm = serial_comm 
        
        self.ser: serial.Serial | None = None 
        self.read_thread: threading.Thread | None = None 
        self.control_thread: threading.Thread | None = None 
        
        # 【v4.3.2 刪除】 刪除獨立的 hw_state_data 和 lock
        # self.hw_state_data = RobotStateHardware()
        # self.lock = threading.Lock()
        
        self._is_running_event = threading.Event()
        self.command_queue = Queue()
        self.internal_state = HWState.STOPPED
        self.last_state_change_time = time.time()
        self.ai_control_active = False

        self._subscribe_to_events()
        log.info("✅ 硬體控制器 (v4.3.2 數據流統一版) 已初始化。")

    def _subscribe_to_events(self):
        event_bus.subscribe(EVENT_HARDWARE_AI_TOGGLE_REQUESTED, 
                            lambda: self.command_queue.put(HWCommand.TOGGLE_AI))
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件 (將發送至內部隊列)。")

    def request_start(self) -> None:
        """(外部API, 非阻塞) 請求啟動硬體控制器。"""
        if self.internal_state in [HWState.STOPPED, HWState.FAILED]:
            log.info("收到啟動請求，向控制執行緒發送 START 命令。")
            self._start_threads_if_not_alive()
            self.command_queue.put(HWCommand.START)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略啟動請求。")

    def request_stop(self) -> None:
        """(外部API, 非阻塞) 請求停止硬體控制器。"""
        if self.internal_state == HWState.RUNNING:
            log.info("收到停止請求，向控制執行緒發送 STOP 命令。")
            self.command_queue.put(HWCommand.STOP)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略停止請求。")
    
    def shutdown(self):
        """(外部API, 阻塞) 應用程式關閉時的強制清理。"""
        self._is_running_event.clear()
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        log.info("硬體控制器所有執行緒已關閉。")

    def _start_threads_if_not_alive(self):
        """(內部) 確保背景執行緒被創建並啟動。"""
        self._is_running_event.set()
        if not self.control_thread or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            log.info("硬體控制執行緒已啟動。")
        
        if not self.read_thread or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            log.info("硬體讀取執行緒已啟動。")

    def _read_from_port(self):
        # 【v4.7.4 修改】將診斷日誌降級為 DEBUG
        log.debug("[硬體讀取執行緒已啟動] 等待數據...")
        while self._is_running_event.is_set():
            if self.internal_state != HWState.RUNNING or not self.ser or not self.ser.is_open:
                # log.debug(...) # 使用 debug 級別避免刷屏
                time.sleep(0.1)
                continue

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore') # <--- 移除 .strip()
                    # 【v4.7.4 修改】將診斷日誌降級為 DEBUG
                    log.debug(f"原始串口接收: {repr(line)}")
                    if line:
                        self.parse_policy_stream(line) 

            except (serial.SerialException, OSError):
                log.error("❌ 讀取時序列埠斷開或出錯。將狀態設置為 FAILED。")
                self._set_internal_state(HWState.FAILED)
                break
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}", exc_info=True)
                self._set_internal_state(HWState.FAILED)
                break
            time.sleep(0.01)

    def _set_internal_state(self, new_state: HWState):
        """(內部) 安全地切換狀態機並同步到全局 State。"""
        if self.internal_state != new_state:
            # 【v4.7.4 修改】將診斷日誌降級為 DEBUG
            log.debug(f"硬體控制器狀態: {self.internal_state.name} -> {new_state.name}") # <--- 增加診斷日誌
            self.internal_state = new_state
            self.last_state_change_time = time.time()
            with self.state.lock:
                self.state.hardware_is_running = (new_state == HWState.RUNNING)

    def _control_loop(self):
        """(執行緒) 狀態機驅動者和命令派發中心。"""
        log.info("--- 硬體控制執行緒已就緒，等待命令 ---")
        while self._is_running_event.is_set():
            try:
                command: HWCommand = self.command_queue.get_nowait()
                if command == HWCommand.START and self.internal_state in [HWState.STOPPED, HWState.FAILED]:
                    self._execute_start()
                elif command == HWCommand.STOP and self.internal_state == HWState.RUNNING:
                    self._execute_stop()
                elif command == HWCommand.TOGGLE_AI and self.internal_state == HWState.RUNNING:
                    self._execute_toggle_ai()
            except Empty:
                pass

            if self.internal_state == HWState.RUNNING and self.ai_control_active:
                self._perform_ai_step()
            
            time.sleep(1.0 / self.config.control_freq)

    def _execute_start(self):
        """(內部, 可能阻塞) 執行啟動流程。"""
        self._set_internal_state(HWState.STARTING)
        if not self.serial_comm.is_connected:
            log.error("❌ 硬體啟動失敗：序列埠未連接。")
            self._set_internal_state(HWState.FAILED)
            return

        self.ser = self.serial_comm.get_serial_connection()  # 取得序列連線實體
        if not self.ser:
            log.error("❌ 硬體啟動失敗：無法獲取有效連接。")
            self._set_internal_state(HWState.FAILED)
            return

        # --- 接管與初始化 ---
        log.info(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True  # 告知 serial_comm 不再管理 serial

        try:
            log.info("  -> 命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            log.info("  -> 已發送 Teensy 模式指令。")
            self._set_internal_state(HWState.RUNNING)

            # 【v4.7.4 修正】硬體模式啟動後，預設自動啟用 AI 控制。
            self.ai_control_active = True
            with self.state.lock: 
                self.state.hardware_ai_is_active = True
            log.info("🤖 硬體模式啟動成功，AI 控制已自動啟用。")

        except serial.SerialException as e:
            log.error(f"❌ 發送模式指令失敗: {e}")
            self.serial_comm.is_managed_by_hardware_controller = False
            self._set_internal_state(HWState.FAILED)

    def _execute_stop(self):
        """(內部, 可能阻塞) 執行停止流程。"""
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False
        with self.state.lock: self.state.hardware_ai_is_active = False

        if self.ser and self.ser.is_open:
            try:
                log.info("  -> 命令 Teensy 停止並恢復 HUMAN 模式...")
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
        """(內部) 執行切換AI的邏輯。"""
        self.ai_control_active = not self.ai_control_active
        with self.state.lock:
            self.state.hardware_ai_is_active = self.ai_control_active
        
        log.info(f"🤖 硬體 AI 控制已 {'啟用' if self.ai_control_active else '暫停'}.")
        
        if self.ai_control_active:
            self.policy.reset()
        elif self.ser and self.ser.is_open:
            try: self.ser.write(b"stop\n")
            except serial.SerialException as e: log.error(f"發送停止指令失敗: {e}")


    def _perform_ai_step(self):
        """
        (內部) 執行單步 AI 計算與控制。
        【v4.3.2 修改】 _perform_ai_step 方法
        【v4.7.1b 修改】 _perform_ai_step 方法，修復 last_action 時序
        """
        # 【v4.3.2 刪除】 不再自行構建觀測向量
        # observation = self.construct_observation()
        # if observation.size == 0: return

        # 【v4.3.2 修改】 直接呼叫 get_action_for_hardware，無需傳遞參數。
        # 【v4.4.7 修改】 調用統一的 get_action API，並傳入當前的 command 狀態。
        # PolicyManager 會自動從 state.std_obs 獲取數據。
        _, action_raw = self.policy.get_action(self.state.command)
        
        # 【v4.7.1b 新增】在硬體模式下，也需要遵循正確的時序來更新 last_action
        # 在發送指令之前，將本幀的動作寫入 state，供下一幀使用。
        # 【v4.7.4 修正】同時更新用於 UI 顯示的 state 和用於下一幀 AI 輸入的 state
        # 這是解決硬體模式下「原始動作 (Raw)」UI 不更新 Bug 的關鍵。
        with self.state.lock:
            self.state.latest_action_raw = action_raw.copy()
            self.state.raw_last_action = action_raw.copy()
        
        action_scale = self.config.initial_tuning_params.action_scale
        default_pose_hardware = np.zeros(12)
        final_command = default_pose_hardware + action_raw * action_scale
        
        action_str = ' '.join(f"{a:.4f}" for a in final_command)
        command_to_send = f"move all {action_str}\n"

        if self.ser and self.ser.is_open:
            try: self.ser.write(command_to_send.encode('utf-8'))
            except serial.SerialException:
                log.error("AI 步驟中發送指令失敗，連接可能已斷開。")
                self._set_internal_state(HWState.FAILED)


    def parse_policy_stream(self, line: str):
        """
        【v4.3.2 修改】 parse_policy_stream 方法
        【v4.4.2 重構】嚴格按照數據契約解析 Teensy 數據流。
        【v4.6.0 修改】增加 .strip() 來移除換行符，並強化錯誤日誌記錄。
        【v4.7.1b 修改】修復了使用未經清理的行進行分割的 Bug。

        職責說明：本函式是 Teensy 原始數據進入統一數據流系統的唯一入口。
        """
        # 【v4.7.4 修改】將診斷日誌降級為 DEBUG
        log.debug(f"parse_policy_stream 正在處理: {repr(line)}")
        try:
            # 【v4.7.1b 修正】必須使用 strip() 後的乾淨行來進行後續所有操作
            clean_line = line.strip()
            if not clean_line:
                return
            
            # 【v4.7.1b 修正】從 clean_line 分割，而不是原始的 line
            parts = clean_line.split(',')

            if len(parts) != 34:
                log.warning(f"數據幀欄位數量錯誤。預期 34，收到 {len(parts)}。原始數據: '{clean_line}'")
                return
            
            data_vec = np.array(parts, dtype=np.float32)
            
            with self.state.lock:
                self.state.raw_torso_angular_velocity[:] = data_vec[0:3]
                self.state.raw_gravity_vector[:] = data_vec[3:6]
                self.state.raw_accelerometer[:] = data_vec[6:9]
                self.state.raw_pitch_rad = data_vec[9]
                self.state.raw_joint_positions[:] = data_vec[10:22]
                self.state.raw_joint_velocities[:] = data_vec[22:34]
        
        except (ValueError, IndexError) as e:
            # 【v4.7.1b 增強】在日誌中同時打印 clean_line 和原始 line，方便調試
            log.error(f"解析數據幀失敗: {e}。Cleaned Line: '{line.strip()}', Original Line: '{repr(line)}'")


    # 【v4.3.2 刪除】 construct_observation 方法
    # 此方法的職責已完全轉移給 ObservationManager。