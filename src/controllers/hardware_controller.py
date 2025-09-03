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
# 【v4.10.1 新增】導入 State 和 Enum 以進行類型提示
from src.core.state import SimulationState, HardwareLinkStatus
from src.core.event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED
# 【v4.9.0 新增】導入新的 TeensyAPI
from src.hardware.teensy_api import TeensyAPI

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
        # 【v4.9.0 新增】宣告 teensy_api 屬性
        self.teensy_api: TeensyAPI | None = None
        
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
        """
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        【v4.10.1 重構】實現包含回傳驗證的、基於狀態機的穩健硬體啟動流程。
        【v4.10.2 重構】在啟動前，先透過握手協議安全地獲取序列埠控制權。
        【v4.10.4 修改】使用 TeensyAPI.execute_command 更新啟動序列。

        (內部, 可能阻塞) 執行啟動流程。
        """
        self._set_internal_state(HWState.STARTING)
        with self.state.lock:
            self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED

        if not self.serial_comm.is_connected:
            log.error("❌ 硬體啟動失敗：序列埠未連線。")
            self._set_internal_state(HWState.FAILED)
            return

        # --- 【v4.10.2 新增】步驟 1：等待綠燈 (安全獲取控制權) ---
        if not self.serial_comm.relinquish_control():
            log.error("❌ 硬體啟動失敗：無法從 SerialCommunicator 安全地獲取序列埠的控制權 (等待逾時)。")
            self._set_internal_state(HWState.FAILED)
            return

        # --- 步驟 2：安全通行 (執行原有啟動流程) ---
        self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            log.error("❌ 硬體啟動失敗：無法獲取有效連線。")
            self._set_internal_state(HWState.FAILED)
            self.serial_comm.resume_control() # 發生錯誤，立即歸還控制權
            return

        self.teensy_api = TeensyAPI(self.serial_comm, self.state)

        # 【v4.10.3 新增】增加檢查，強制執行生命週期契約。
        # 如果 TeensyAPI 因任何原因未能成功初始化其 ser 物件，
        # 流程將在此處立即失敗，並提供清晰的錯誤信息。
        if not self.teensy_api.ser:
            log.error("❌ 硬體啟動失敗: TeensyAPI 未能獲取序列埠參考。這通常意味著在交接控制權時發生了問題。")
            self._set_internal_state(HWState.FAILED)
            self.serial_comm.resume_control()
            return

        try:
            log.info("--- 開始執行硬體啟動序列 (v4.10.4 協定感知版) ---")
            
            # 【v4.10.4 修改】啟動指令序列現在調用新的 execute_command API
            # 1. 發送 'stop' (協定: NONE, 只需確認發送成功)
            log.info("  -> 步驟 1/3：發送 'stop' 指令...")
            if not self.teensy_api.execute_command("stop"):
                raise serial.SerialException("發送 'stop' 指令失敗，啟動中止。")
            
            # 2. 發送 'monitor freq' (協定: OK, 會等待 [OK] 確認)
            log.info(f"  -> 步驟 2/3：設定遙測頻率為 {self.config.control_freq} Hz 並等待確認...")
            if not self.teensy_api.execute_command(f"monitor freq {self.config.control_freq}", timeout=1.0):
                raise serial.SerialException("Teensy 未能確認 'monitor freq' 指令，啟動中止。")

            # 3. 發送 'monitor p' (協定: NONE, 只需確認發送成功)
            log.info("  -> 步驟 3/3：命令 Teensy 切換至 POLICY_STREAM 模式...")
            if not self.teensy_api.execute_command("monitor p"):
                raise serial.SerialException("發送 'monitor p' 指令失敗，啟動中止。")

            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            
            log.info("✅ 硬體啟動序列成功完成，通訊連線已驗證。")
            # 【v4.10.4 修改】將狀態設定為 VERIFIED
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.VERIFIED
            self._set_internal_state(HWState.RUNNING)
            
            self.ai_control_active = True
            with self.state.lock: 
                self.state.hardware_ai_is_active = True
            log.info("🤖 AI 控制已自動啟用。")

        except serial.SerialException as e:
            log.error(f"❌ 硬體啟動序列失敗: {e}")
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED
            self._set_internal_state(HWState.FAILED)
            self.teensy_api = None
            self.serial_comm.resume_control()


    def _execute_stop(self):
        """
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        【v4.10.2 重構】在停止後，將序列埠的控制權安全地歸還。
        
        (內部, 可能阻塞) 執行停止流程。
        """
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False
        with self.state.lock: 
            self.state.hardware_ai_is_active = False
            # 【v4.10.2 新增】停止時，將連線狀態重設為未驗證
            self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED

        if self.teensy_api:
            log.info("  -> 命令 Teensy 停止並恢復 HUMAN 模式...")
            # 【v4.10.2 修改】這裡需要繞過 Mute 檢查，直接發送指令
            self.teensy_api.send_command_and_wait_for_ok("stop", timeout=0.5)
            self.teensy_api.send_command_and_wait_for_ok("monitor h", timeout=0.5)
        
        # 清理自身資源
        self.ser = None
        self.teensy_api = None
        
        # 【v4.10.2 修改】將歸還控制權的邏輯放在所有操作的最後
        if self.serial_comm:
            self.serial_comm.resume_control()
            log.info("  -> 序列埠的控制權已歸還。")
        
        self._set_internal_state(HWState.STOPPED)


    def _execute_toggle_ai(self):
        """
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        
        (內部) 執行切換AI的邏輯。
        """
        self.ai_control_active = not self.ai_control_active
        with self.state.lock:
            self.state.hardware_ai_is_active = self.ai_control_active
        
        log.info(f"🤖 硬體 AI 控制已 {'啟用' if self.ai_control_active else '暫停'}.")
        
        if self.ai_control_active:
            self.policy.reset()
        # 【v4.9.0 修改】使用 API 取代硬編碼字串
        elif self.teensy_api:
            self.teensy_api.stop_ai_and_motors()


    def _perform_ai_step(self):
        """
        【v4.3.2 修改】 _perform_ai_step 方法
        【v4.7.1b 修改】 _perform_ai_step 方法，修復 last_action 時序
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        【v4.10.4 修改】簡化指令發送邏輯。

        (內部) 執行單步 AI 計算與控制。
        """
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
        
        # 【v4.10.4 修改】簡化指令發送調用
        # 這裡直接調用 send_motor_commands，它負責處理底層的發送、
        # 協定選擇（NONE）以及安全守衛。
        if self.teensy_api:
            if not self.teensy_api.send_motor_commands(final_command):
                # 如果 send_motor_commands 返回 False，則意味著發送失敗（可能是連接問題）
                log.error("AI 步驟中發送馬達指令失敗，連接可能已斷開或處於靜默狀態。")
                # 【v4.10.4 新增】在發送失敗時，將狀態設為 FAILED
                self._set_internal_state(HWState.FAILED)
        else:
            # 如果 teensy_api 本身就未初始化，也無法發送
            log.warning("TeensyAPI 未初始化，無法發送馬達指令。")


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