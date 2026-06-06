# src/controllers/hardware_controller.py

# 【v4.3.2 修改】 調整 import
import serial
import threading
import time
from collections import deque  # 確保 deque 已導入
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
    """【v4.11.0-w 修改】內建頻率監控並修復資料流。"""
    
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
        
        # 【v4.11.0-w 新增】實現頻率監控方案
        # 根據您的設計，初始化兩個雙端佇列用於儲存最近的時間戳
        self.data_received_times = deque(maxlen=100) # 儲存成功解析數據幀的時間
        self.ai_step_times = deque(maxlen=100)       # 儲存執行AI決策的時間
        # 【v4.13.9-w】頻率隊列執行緒安全鎖
        self._freq_lock = threading.Lock()

        self._subscribe_to_events()
        log.info("✅ 硬體控制器 (v4.11.0-w 頻率監控版) 已初始化。")

        # 【修正】移除重複的訂閱與初始化日誌（避免重複訂閱與重複輸出）
        # self._subscribe_to_events()
        # log.info("✅ 硬體控制器 (v4.3.2 數據流統一版) 已初始化。")

    # 【v4.14.3 新增】Sim2Real 關節方向校準函數
    def _apply_motor_direction_calibration(self, joint_vector: np.ndarray) -> np.ndarray:
        """
        應用 Sim2Real 的馬達方向校準。

        此函數作為模擬空間 (Sim-Space) 與硬體空間 (HW-Space) 之間的雙向轉換器。
        由於校準向量只包含 1 和 -1，其本身就是自己的逆，因此同一個函數
        可用於兩個方向的轉換：
        - Sim-to-HW (指令): 將 AI 的 Sim-Space 指令轉換為 HW-Space。
        - HW-to-Sim (回饋): 將 Teensy 的 HW-Space 回饋轉換為 Sim-Space。
        """
        # 從配置中安全地獲取校準向量
        vector_list = self.config.sim2real_motor_calibration.get('correction_vector', [1]*self.config.num_motors)
        calibration_vector = np.array(vector_list, dtype=np.float32)
        
        # 執行逐元素乘法進行校準
        return joint_vector * calibration_vector

    def _subscribe_to_events(self):
        # 【v4.13.1-w】允許事件帶參數，避免 TypeError
        event_bus.subscribe(
            EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
            lambda *args, **kwargs: self.command_queue.put(HWCommand.TOGGLE_AI)
        )
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件 (將發送至內部隊列)。")

    def request_start(self) -> None:
        """(外部API, 非阻塞) 請求啟動硬體控制器。"""
        # 【v4.10.5 修改】在允許啟動的狀態中，排除 CONNECTION_LOST。
        # 一旦連接中斷，必須通過重新連接序列埠（重啟應用或重新掃描）來重置。
        if self.internal_state in [HWState.STOPPED, HWState.FAILED] and self.state.hardware_link_status != HardwareLinkStatus.CONNECTION_LOST:
            log.info("收到啟動請求，向控制執行緒發送 START 命令。")
            self._start_threads_if_not_alive()
            self.command_queue.put(HWCommand.START)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name} (Link: {self.state.hardware_link_status.value})，忽略啟動請求。")

    def request_stop(self) -> None:
        """(外部API, 非阻塞) 請求停止硬體控制器。"""
        if self.internal_state == HWState.RUNNING:
            log.info("收到停止請求，向控制執行緒發送 STOP 命令。")
            self.command_queue.put(HWCommand.STOP)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略停止請求。")
    
    def shutdown(self):
        """
        【v4.14.1-w 最小修正】優先讓控制迴圈處理 STOP；如超時未停，執行後援停機。
        交還控制權後，喚醒 SerialCommunicator 並在 HUMAN 模式下設回 monitor freq，
        確保 UI 仍能收到 Teensy 遙測。
        """
        # 1) 如果還在 RUNNING，先丟 STOP 給控制迴圈（正常路徑）
        if self.internal_state == HWState.RUNNING:
            try:
                self.command_queue.put(HWCommand.STOP)
            except Exception as e:
                log.warning(f"shutdown(): 無法送出 STOP 命令，將直接進入後援停機。原因: {e}")

            # 等待控制迴圈去執行 _execute_stop()
            deadline = time.time() + 0.8
            while time.time() < deadline:
                if self.internal_state in (HWState.STOPPING, HWState.STOPPED, HWState.FAILED):
                    break
                time.sleep(0.02)

            # 2) 超時仍 RUNNING → 啟動後援停機（不關串口，只切人類模式並交還控制權）
            if self.internal_state == HWState.RUNNING:
                log.warning("shutdown(): 正常停機超時，改用安全停機後援。")
                self._set_internal_state(HWState.STOPPING)

                # 關掉 AI 旗標
                self.ai_control_active = False
                try:
                    with self.state.lock:
                        self.state.hardware_ai_is_active = False
                except Exception:
                    pass

                # 優先用 TeensyAPI（協定感知），否則退回直寫
                ser_open = bool(self.ser and getattr(self.ser, "is_open", False))
                try:
                    if self.teensy_api and ser_open:
                        self.teensy_api.execute_command("stop")        # NONE
                        self.teensy_api.execute_command("monitor h")   # NONE
                    elif ser_open:
                        self.ser.write(b"stop\n"); time.sleep(0.05)
                        self.ser.write(b"monitor h\n"); time.sleep(0.05)
                    else:
                        log.info("shutdown(): 序列埠未開啟，略過停止指令。")
                except serial.SerialException as e:
                    log.warning(f"shutdown(): 後援停機發送指令失敗（略過）：{e}")

                # >>> 這裡是你要插入「交還控制權後的兩步」的位置 <<<
                # 交還 serial_comm 管理權（不在這裡 close；main.py 會收尾）
                try:
                    if self.serial_comm:
                        self.serial_comm.is_managed_by_hardware_controller = False
                        log.info("shutdown(): 序列埠控制權已交還給 SerialCommunicator。")
                except Exception as e:
                    log.warning(f"shutdown(): 交還控制權時發生非致命錯誤：{e}")

                # 新增 Step A：喚醒 SerialCommunicator 的讀取迴圈
                if hasattr(self.serial_comm, "resume_control"):
                    try:
                        self.serial_comm.resume_control()
                        log.info("shutdown(): SerialCommunicator 已恢復接手串口讀取。")
                    except Exception as e:
                        log.warning(f"shutdown(): resume_control() 失敗（略過）：{e}")

                # 新增 Step B：在 HUMAN 模式下維持遙測頻率（避免 UI 看起來沒資料）
                try:
                    if self.teensy_api and self.serial_comm and self.serial_comm.is_connected:
                        self.teensy_api.execute_command(f"monitor freq {self.config.control_freq}")
                        log.info(f"shutdown(): HUMAN 模式下已設置 monitor freq = {self.config.control_freq} Hz")
                except Exception as e:
                    log.warning(f"shutdown(): monitor freq 設置失敗（略過）：{e}")

                # （可選，但推薦）把 Link 狀態設回可讀狀態，避免 UI 用狀態濾掉顯示
                try:
                    with self.state.lock:
                        self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED
                except Exception:
                    pass

                # 解除本地 ser 引用並標記已停止（實際 close 由 main.py 最後執行）
                self.ser = None
                self._set_internal_state(HWState.STOPPED)

        # 3) 關閉控制事件並等待執行緒結束
        try:
            self._is_running_event.clear()
        except Exception:
            pass

        try:
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=1.2)
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=1.2)
        except Exception as e:
            log.warning(f"shutdown(): 等待執行緒結束時出現非致命錯誤：{e}")

        self.control_thread = None
        self.read_thread = None

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
        """
        【v4.7.4 修改】將診斷日誌降級為 DEBUG
        【v4.10.5 修改】增加對 SerialException 的處理，觸發 CONNECTION_LOST 熔斷狀態。
        """

        log.debug("[硬體讀取執行緒已啟動] 等待數據...")
        while self._is_running_event.is_set():
            if self.internal_state != HWState.RUNNING or not self.ser or not self.ser.is_open:
                # 若未 RUNNING，改用短等待事件或極短讓渡，避免把刷新率鎖死在 10Hz
                # 【v4.13.2-w】使用位置參數以避免某些版本對關鍵字參數的兼容性問題
                self._is_running_event.wait(0.02)
                continue

            try:
                # 直接呼叫 readline() 讓 timeout=0.02 自然節流；無資料會在 ~20ms 返回空字串
                line = self.ser.readline().decode('utf-8', errors='ignore')
                if line:
                    # 【v4.11.0-w】先記錄原始資料，再解析
                    log.debug(f"原始串口接收: {repr(line)}")
                    self.parse_policy_stream(line)

            # 【v4.10.5 修改】實現 FEAT-SAFETY-FUSE 熔斷器
            except (serial.SerialException, OSError) as e:
                # 【v4.10.5 核心修改】實現安全熔斷器
                log.error(f"❌ 讀取時序列埠斷開或出錯: {e}。觸發安全熔斷！")
                # 將全局狀態設置為 CONNECTION_LOST
                with self.state.lock:
                    self.state.hardware_link_status = HardwareLinkStatus.CONNECTION_LOST
                # 將控制器內部狀態也設置為 FAILED，以阻止後續操作
                self._set_internal_state(HWState.FAILED)
                self._is_running_event.clear()  # 【v4.13.3-w】讓控制執行緒也能退出
                break
            
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}", exc_info=True)
                self._set_internal_state(HWState.FAILED)
                self._is_running_event.clear()
                break
            # 不再固定 sleep；節流交給 serial timeout 與事件等待

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
        """
        【v4.11.1 重構】採用固定時間步長邏輯取代 time.sleep() 以實現精準頻率控制。
        【v4.13.0 重構】採用與 SimulationController v4.12.4 完全對稱的時間控制架構。
        
        (執行緒) 狀態機驅動者和命令派發中心。
        """
        log.info("--- 硬體控制執行緒已就緒，等待指令 ---")

        # --- 步驟 1: 初始化時間控制器 ---
        # 獲取契約中定義的邏輯週期 (e.g., 5Hz -> 0.2s)
        logic_interval = 1.0 / self.config.control_freq
        
        # 獲取執行緒中最小的時間單位，用於累加器
        # 在硬體模式下，我們沒有 physics_timestep，但可以定義一個理論上的最小步長
        # 或者直接使用 logic_interval 作為我們的時間塊
        time_block = logic_interval # 為了簡化，我們先讓時間塊等於邏輯週期

        # 初始化計時器和累加器
        last_frame_time = time.perf_counter()
        time_accumulator = 0.0
        
        # 保護機制，防止從休眠恢復時的「螺旋式死亡」
        MAX_FRAME_TIME = 0.25 

        # --- 步驟 2: 進入主控制迴圈 ---
        while self._is_running_event.is_set():
            # --- 2a: 累積真實時間債務 ---
            current_time = time.perf_counter()
            frame_time = current_time - last_frame_time
            last_frame_time = current_time
            if frame_time > MAX_FRAME_TIME:
                frame_time = MAX_FRAME_TIME
            time_accumulator += frame_time
            
            # --- 2b: 償還時間債務 (執行邏輯幀) ---
            # 只要累積的時間債務足夠執行一個完整的邏輯幀，就執行
            if time_accumulator >= time_block:
                # --- 執行一個完整的邏輯幀 ---
                # 處理掛起的命令
                # 【v4.13.6-w】批次處理多筆指令避免堆積
                for _ in range(8):
                    try:
                        command: HWCommand = self.command_queue.get_nowait()
                    except Empty:
                        break
                    if command == HWCommand.START and self.internal_state in [HWState.STOPPED, HWState.FAILED]:
                        self._execute_start()
                    elif command == HWCommand.STOP and self.internal_state == HWState.RUNNING:
                        self._execute_stop()
                    elif command == HWCommand.TOGGLE_AI and self.internal_state == HWState.RUNNING:
                        self._execute_toggle_ai()

                # 執行 AI 決策與指令發送
                if self.internal_state == HWState.RUNNING and self.ai_control_active:
                    self.ai_step_times.append(time.perf_counter())
                    self._perform_ai_step()

                # 更新觀測管理器
                if self.state.observation_manager_ref:
                    try:
                        self.state.observation_manager_ref.update_all_observations()
                    except Exception as e:
                        log.warning(f"[HW] 觀測更新失敗，已跳過本幀: {e}")
                
                # 更新頻率監控數據
                self._update_frequencies()
                
                # 「償還」一筆時間債務
                time_accumulator -= time_block
            
            # --- 2c: 短暫休眠讓出 CPU ---
            # 【修正】僅在尚未達到下一個 time_block 時，短暫讓出 CPU，避免多餘抖動
            remaining = time_block - time_accumulator
            if remaining > 0:
                time.sleep(min(0.001, remaining))
            
    def _update_frequencies(self):
        """【v4.11.0-w 新增】計算 I/O 和 AI 頻率並寫入中央狀態。"""
        current_time = time.perf_counter()
        
        # 計算數據接收 (I/O) 頻率
        if len(self.data_received_times) > 1:
            # 移除超過1秒的舊數據，使頻率計算更即時
            with self._freq_lock:  # 【v4.13.9-w】加鎖確保與解析執行緒安全
                while len(self.data_received_times) > 1 and current_time - self.data_received_times[0] > 1.0:
                    self.data_received_times.popleft()
                if len(self.data_received_times) > 1:
                    elapsed = self.data_received_times[-1] - self.data_received_times[0]
                    self.state.hw_data_freq = (len(self.data_received_times) - 1) / elapsed if elapsed > 0 else 0.0

        # 計算 AI 決策頻率
        if len(self.ai_step_times) > 1:
            # 【v4.14.1 修復】使用 deque.popleft() 而不是 deque.pop(0) 來移除最左側（最舊）的元素。
            # deque.pop(0) 是不合法的語法，會導致 TypeError 並使控制執行緒崩潰。
            while len(self.ai_step_times) > 1 and current_time - self.ai_step_times[0] > 1.0:
                self.ai_step_times.popleft()

            if len(self.ai_step_times) > 1:
                elapsed = self.ai_step_times[-1] - self.ai_step_times[0]
                self.state.hw_ai_freq = (len(self.ai_step_times) - 1) / elapsed if elapsed > 0 else 0.0
            
    def _execute_start(self):
        """
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        【v4.10.1 重構】實現包含回傳驗證的、基於狀態機的穩健硬體啟動流程。
        【v4.10.2 重構】在啟動前，先透過握手協議安全地獲取序列埠控制權。
        【v4.10.4 修改】使用 TeensyAPI.execute_command 更新啟動序列。
        【v4.10.5 修改】啟動成功後，進入 MUTED 狀態以實現「預設安全」。

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
        # 取得控制權後，任何失敗路徑都必須成對歸還控制權。用 started_ok 旗標 + try/finally 保證：
        # 只要啟動未成功（含非預期例外），resume_control() 必被呼叫一次，避免 SerialCommunicator
        # 讀取執行緒永久暫停與序列埠洩漏（症狀：退出硬體後下次連不上）。成功時則保留控制權給本控制器。
        started_ok = False
        try:
            self.ser = self.serial_comm.get_serial_connection()
            if not self.ser:
                log.error("❌ 硬體啟動失敗：無法獲取有效連線。")
                self._set_internal_state(HWState.FAILED)
                return

            self.teensy_api = TeensyAPI(self.serial_comm, self.state)

            # 強制執行生命週期契約：若 TeensyAPI 未能初始化其 ser 物件，立即失敗。
            if not self.teensy_api.ser:
                log.error("❌ 硬體啟動失敗: TeensyAPI 未能獲取序列埠參考。這通常意味著在交接控制權時發生了問題。")
                self._set_internal_state(HWState.FAILED)
                self.teensy_api = None
                return

            # 統一 ser 來源，避免讀寫端不同步
            self.ser = self.teensy_api.ser

            log.info("--- 開始執行硬體啟動序列 (協定感知版) ---")
            # 啟動序列不再發送 'stop'：為了從現有站姿無縫進入硬體模式，避免機器人掉電癱軟。
            # 前提：假設 Teensy 在此之前已處於一個已知的安全狀態（例如，站立）。

            # 2. 發送 'monitor freq' (協定: OK, 會等待 [OK] 確認)
            log.info(f"  -> 步驟 2/3：設定遙測頻率為 {self.config.control_freq} Hz 並等待確認...")
            if not self.teensy_api.execute_command(f"monitor freq {self.config.control_freq}", timeout=1.0):
                raise serial.SerialException("Teensy 未能確認 'monitor freq' 指令，啟動中止。")

            # 3. 發送 'monitor p' (協定: NONE, 只需確認發送成功)
            log.info("  -> 步驟 3/3：命令 Teensy 切換至 POLICY_STREAM 模式...")
            if not self.teensy_api.execute_command("monitor p"):
                raise serial.SerialException("發送 'monitor p' 指令失敗，啟動中止。")

            time.sleep(0.1)
            if self.ser:
                self.ser.reset_input_buffer()

            log.info("✅ 硬體啟動序列成功完成，通訊連線已驗證。")
            # 實現預設安全：啟動成功後預設進入 MUTED 狀態，等待使用者明確啟用。
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.MUTED

            self._set_internal_state(HWState.RUNNING)

            # 預設靜默，AI 不自動跑
            self.ai_control_active = False
            with self.state.lock:
                self.state.hardware_ai_is_active = False

            log.info("🤖 啟動完成，預設為 MUTED 並關閉 AI 決策，待 UI 啟用。")
            started_ok = True

        except serial.SerialException as e:
            log.error(f"❌ 硬體啟動序列失敗: {e}")
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED
            self._set_internal_state(HWState.FAILED)
            self.teensy_api = None
        except Exception as e:
            # 非預期例外也要落入 FAILED；歸還控制權由下方 finally 統一處理，避免序列埠洩漏。
            log.error(f"❌ 硬體啟動發生非預期錯誤: {e}", exc_info=True)
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED
            self._set_internal_state(HWState.FAILED)
            self.teensy_api = None
        finally:
            if not started_ok:
                # 啟動未成功 → 必定歸還控制權（與上面的 relinquish_control 成對），喚醒讀取迴圈。
                try:
                    self.serial_comm.resume_control()
                    log.info("  -> 啟動未成功，已歸還序列埠控制權給 SerialCommunicator。")
                except Exception as e:
                    log.warning(f"resume_control() 失敗（略過）：{e}")
                self.ser = None


    def _execute_stop(self):
        """
        【【v4.14.0 修改】移除了會導致死鎖的執行緒 self-join 邏輯。此函式現在只負責業務層面的停止操作。
        【v4.10.5 修改】更新 API 呼叫以匹配 v4.10.4 的 TeensyAPI 重構。
        【v4.10.2 重構】在停止後，將序列埠的控制權安全地歸還。
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        
        執行硬體停止流程。
        
        此方法負責將 AI 控制暫停，命令 Teensy 停止馬達並切換回人類可讀模式，
        最後安全地釋放對序列埠的獨佔控制權。
        """
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False
        with self.state.lock: 
            self.state.hardware_ai_is_active = False

        # 1) 防呆：檢查序列埠是否仍然開啟
        ser_open = bool(self.ser and getattr(self.ser, "is_open", False))

        try:
            # 2) 優先走協定感知 API（可處理 [OK]/NONE 等）
            if self.teensy_api and ser_open:
                self.teensy_api.execute_command("stop")        # 對應 v4.10.4 協定
                self.teensy_api.execute_command("monitor h")   # 對應 v4.10.4 協定
            # 3) 後備：若尚未建立 TeensyAPI，但序列埠可用，才直接寫入
            elif ser_open:
                self.ser.write(b"stop\n"); time.sleep(0.05)
                self.ser.write(b"monitor h\n"); time.sleep(0.05)
            else:
                log.warning("序列埠未開啟，略過停止指令。")
        except serial.SerialException as e:
            log.warning(f"停止流程：發送停止/切換指令失敗，將繼續交還控制權。原因: {e}")

        # --- 交還 serial_comm 管理權（你原本就有） ---
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            log.info("  -> 序列埠控制權已交還。")

        # --- 新增：喚醒 SerialCommunicator 的讀取迴圈 ---
        if hasattr(self.serial_comm, "resume_control"):
            try:
                self.serial_comm.resume_control()
                log.info("  -> SerialCommunicator 已恢復接手串口讀取。")
            except Exception as e:
                log.warning(f"resume_control() 失敗（略過）：{e}")

        # --- 新增：在 HUMAN 模式下維持遙測頻率 ---
        try:
            if self.teensy_api and self.serial_comm and self.serial_comm.is_connected:
                self.teensy_api.execute_command(f"monitor freq {self.config.control_freq}")
                log.info(f"  -> HUMAN 模式下已設置 monitor freq = {self.config.control_freq} Hz")
        except Exception as e:
            log.warning(f"monitor freq 設置失敗（略過）：{e}")

        # --- 可選：調整鏈路狀態，避免 UI 鎖住 ---
        try:
            with self.state.lock:
                self.state.hardware_link_status = HardwareLinkStatus.UNVERIFIED
        except Exception:
            pass

        # 5) 解除本地引用並結束
        self.ser = None
        self._set_internal_state(HWState.STOPPED)

        # 【v4.14.0 刪除】移除導致死鎖的執行緒停止邏輯。
        # 執行緒的生命週期現在由 shutdown() 方法統一管理，以支持多次啟動/停止。

    def _execute_toggle_ai(self):
        """
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        
        (內部) 執行切換AI的邏輯。
        """
        self.ai_control_active = not self.ai_control_active
        with self.state.lock:
            self.state.hardware_ai_is_active = self.ai_control_active
        
        log.info(f"🤖 硬體 AI 控制已 {'啟用' if self.ai_control_active else '暫停'}。")
        
        if self.ai_control_active:
            self.policy.reset()

        # 【v4.9.0 修改】使用 API 取代硬編碼字串
        # 【v4.10.5 修正】當暫停 AI 時，應發送 'stop' 指令。
        # 舊的 stop_ai_and_motors() 已被移除，導致 AttributeError。
        # 此處使用正確的 execute_command API。
        elif self.teensy_api:
            self.teensy_api.execute_command("stop")


    def _perform_ai_step(self):
        """
        【v4.3.2 修改】 _perform_ai_step 方法
        【v4.7.1b 修改】 _perform_ai_step 方法，修復 last_action 時序
        【v4.9.0 修改】使用 TeensyAPI 進行指令通訊。
        【v4.10.4 修改】簡化指令發送邏輯。
        【v4.11.0-w 修改】修復 final_control 資料流斷裂點。
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

        # 在 Sim-Space 計算出 AI 期望的目標關節指令。
        final_command_sim_space = default_pose_hardware + action_raw * action_scale
        
        # 【v4.14.3 新增】將 Sim-Space 的指令校準為 HW-Space，以便 Teensy 正確執行。
        final_command_hw_space = self._apply_motor_direction_calibration(final_command_sim_space)

        # UI 應顯示發送給硬體的實際值（HW-Space）。
        with self.state.lock:
            self.state.latest_final_ctrl = final_command_hw_space.copy()
        
        # 將校準後的 HW-Space 指令發送給 Teensy。
        if self.teensy_api:
            if not self.teensy_api.send_motor_commands(final_command_hw_space):
                # 如果 send_motor_commands 返回 False，則意味著發送失敗（可能是連接問題）
                log.error("AI 步驟中發送馬達指令失敗，連接可能已斷開或處於靜默狀態。")
                # 【v4.13.7-w】在失敗時結束控制執行緒並標示掉線
                with self.state.lock:
                    self.state.hardware_link_status = HardwareLinkStatus.CONNECTION_LOST
                self._set_internal_state(HWState.FAILED)
                self._is_running_event.clear()
                return
        else:
            # 如果 teensy_api 本身就未初始化，也無法發送
            log.warning("TeensyAPI 未初始化，無法發送馬達指令。")


    def parse_policy_stream(self, line: str):
        """
        【v4.3.2 修改】 parse_policy_stream 方法
        【v4.4.2 重構】嚴格按照數據契約解析 Teensy 數據流。
        【v4.6.0 修改】增加 .strip() 來移除換行符，並強化錯誤日誌記錄。
        【v4.7.1b 修改】修復了使用未經清理的行進行分割的 Bug。
        【v4.10.5 重構】實現智慧日誌解析器，分類處理系統訊息和數據幀。
        【v4.13.0 修改】增強序列埠數據流解析器，以記錄更多詳細的指令回應和數據幀資訊。

        此函式是 Teensy 原始數據進入統一數據流系統的唯一入口。
        它能區分系統訊息（如 [CMD], [ERROR]）和純數據幀，並分別處理。
        """
        # 移除換行符並檢查是否為空行的邏輯
        clean_line = line.strip()
        if not clean_line:
            return
            
        # 【v4.10.5 新增】智慧訊息分類器
        # 如果該行以 '[' 開頭，我們就假定它是一個系統訊息（如 [CMD], [OK], [錯誤]），
        # 而不是一個資料幀。
        # 【v4.13.0 增強】智慧訊息分類器
        if clean_line.startswith('['):
            # 【v4.13.8-w】降級到 debug，避免洗版
            log.debug(f"[Teensy Msg]: {clean_line}")
            
            # 【v4.13.0 增強】如果收到的是 Teensy 執行指令的回應，嘗試解析其內容
            if "[CMD] Executing: move all " in clean_line:
                try:
                    # 提取指令後面的角度字串
                    angle_str = clean_line.split("move all ", 1)[1]
                    # 移除可能的後綴或括號，並轉換為浮點數陣列
                    angles = np.array(angle_str.split(), dtype=np.float32)
                    log.info(f"  -> Teensy 已執行 Move 指令，目標角度: {np.array2string(angles, precision=4, suppress_small=True)}")
                except Exception as e:
                    log.warning(f"  -> 無法解析 Teensy 的 Move 指令回應: {e}")
            
            return  # 系統訊息處理完畢，直接返回

        try:
            # 只有在確認不是系統訊息後，才嘗試按資料幀格式進行解析
            parts = clean_line.split(',')

            if len(parts) != 34:
                # 在非系統訊息且欄位數不對時警告
                log.warning(f"數據幀欄位數量錯誤。預期 34，收到 {len(parts)}。原始數據: '{clean_line}'")
                return
            
            data_vec = np.array(parts, dtype=np.float32)
            
            # 【v4.11.0-w】在成功解析後記錄時間戳
            with self._freq_lock:  # 【v4.13.9-w】多執行緒安全 append
                self.data_received_times.append(time.perf_counter())
            
            # 提取 HW-Space 的原始關節回饋數據。
            raw_joint_positions_hw_space = data_vec[10:22]
            raw_joint_velocities_hw_space = data_vec[22:34]

            # 【v4.14.3 新增】將 HW-Space 的關節回饋數據校準回 Sim-Space。
            # 這樣，ObservationManager 和 AI 模型就能在與其訓練時一致的座標系中接收這些數據。
            raw_joint_positions_sim_space = self._apply_motor_direction_calibration(raw_joint_positions_hw_space)
            raw_joint_velocities_sim_space = self._apply_motor_direction_calibration(raw_joint_velocities_hw_space)

            with self.state.lock:
                self.state.raw_torso_angular_velocity[:] = data_vec[0:3]
                self.state.raw_gravity_vector[:] = data_vec[3:6]
                self.state.raw_accelerometer[:] = data_vec[6:9]
                self.state.raw_pitch_rad = data_vec[9]
                # 【v4.14.3 修改】儲存校準後的 Sim-Space 關節回饋數據。
                self.state.raw_joint_positions[:] = raw_joint_positions_sim_space
                self.state.raw_joint_velocities[:] = raw_joint_velocities_sim_space
                
            # 【v4.13.0 新增】成功解析數據幀後，記錄關節位置 (高頻，使用 debug 級別)
            # 因為這是高頻數據，預設只有在日誌級別設為 DEBUG 時才顯示，避免刷屏
            if self.state.raw_joint_positions.size == 12:
                log.debug(f"[Teensy Data] Joint Pos: {np.array2string(self.state.raw_joint_positions, precision=4, suppress_small=True)}")
        
        except (ValueError, IndexError) as e:
            log.error(f"解析數據幀失敗: {e}。Cleaned Line: '{clean_line}', Original Line: '{repr(line)}'")


    # 【v4.3.2 刪除】 construct_observation 方法
    # 此方法的職責已完全轉移給 ObservationManager。
