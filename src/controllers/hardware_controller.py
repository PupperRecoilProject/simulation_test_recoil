# src/controllers/hardware_controller.py

# [合併] 引入所有必要的模組
import serial
import threading
import time
from src.core.logger import log
import numpy as np
from typing import TYPE_CHECKING
from queue import Queue, Empty
from enum import Enum, auto

# [合併] 導入專案內部模組
from src.core.event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED

# [合併] 條件式導入 VirtualTeensy，這是整合的核心
try:
    from src.hardware.virtual_teensy import VirtualTeensy
except ImportError:
    VirtualTeensy = None # 如果檔案不存在，設為 None

# [合併] 類型檢查區塊
if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.hardware.policy import PolicyManager
    from src.hardware.serial_communicator import SerialCommunicator
    from src.core.state import SimulationState
    # 如果 VirtualTeensy 存在，也加入類型檢查
    if VirtualTeensy:
        from src.hardware.virtual_teensy import VirtualTeensy

# [保留 dev4.3 架構] 移除 RobotStateHardware 和 construct_observation_51
# 這些功能已被 ObservationManager 和 SimulationState.raw_... 取代，不再需要。

class HWCommand(Enum):
    """定義可以發送給硬體控制器的命令類型。"""
    START = auto()
    STOP = auto()
    TOGGLE_AI = auto()


class HWState(Enum):
    """定義硬體控制器內部的狀態機狀態。"""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAILED = auto()


class HardwareController:
    """
    [合併後的版本] 管理硬體AI控制迴圈，作為原始數據提供者。
    此版本整合了 VirtualTeensy 功能，並採用了最新的數據流架構。
    """
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', state: 'SimulationState', serial_comm: 'SerialCommunicator'): # 初始化函式
        self.config = config # 儲存設定
        self.policy = policy # 儲存AI策略管理器
        self.state = state # 儲存中央狀態
        self.serial_comm = serial_comm # 儲存序列埠通訊器
        
        # [合併] self.ser 現在可以是真實序列埠或虛擬Teensy
        self.ser: serial.Serial | VirtualTeensy | None = None 
        self.read_thread: threading.Thread | None = None # 讀取執行緒
        self.control_thread: threading.Thread | None = None # 控制執行緒
        
        # [保留 dev4.3 架構] 不再需要獨立的 lock 和 hw_state_data
        
        self._is_running_event = threading.Event() # 控制執行緒運行的事件
        self.command_queue = Queue() # 執行緒安全的命令佇列
        self.internal_state = HWState.STOPPED # 內部狀態機
        self.ai_control_active = False # AI是否啟用

        self._subscribe_to_events() # 訂閱事件
        log.info("✅ 硬體控制器 (含虛擬Teensy支持) 已初始化。")

    def _subscribe_to_events(self): # 訂閱事件處理函式
        event_bus.subscribe(EVENT_HARDWARE_AI_TOGGLE_REQUESTED, 
                            lambda: self.command_queue.put(HWCommand.TOGGLE_AI))
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件。")

    def request_start(self) -> None: # 請求啟動硬體控制器
        if self.internal_state in [HWState.STOPPED, HWState.FAILED]:
            log.info("收到啟動請求，向控制執行緒發送 START 命令。")
            self._start_threads_if_not_alive()
            self.command_queue.put(HWCommand.START)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略啟動請求。")

    def request_stop(self) -> None: # 請求停止硬體控制器
        if self.internal_state == HWState.RUNNING:
            log.info("收到停止請求，向控制執行緒發送 STOP 命令。")
            self.command_queue.put(HWCommand.STOP)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略停止請求。")
    
    def shutdown(self): # 關閉硬體控制器
        self._is_running_event.clear()
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        log.info("硬體控制器所有執行緒已關閉。")

    def _start_threads_if_not_alive(self): # 啟動背景執行緒
        self._is_running_event.set()
        if not self.control_thread or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            log.info("硬體控制執行緒已啟動。")
        
        if not self.read_thread or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            log.info("硬體讀取執行緒已啟動。")

    def _set_internal_state(self, new_state: HWState): # 設定內部狀態
        if self.internal_state != new_state:
            log.info(f"硬體控制器狀態: {self.internal_state.name} -> {new_state.name}")
            self.internal_state = new_state
            self.last_state_change_time = time.time()
            with self.state.lock:
                self.state.hardware_is_running = (new_state == HWState.RUNNING)

    def _control_loop(self): # 控制迴圈
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

    def _execute_start(self): # 執行啟動流程
        """ [合併後的版本] 執行啟動流程，包含真實/虛擬模式的選擇。"""
        self._set_internal_state(HWState.STARTING)

        if self.config.use_virtual_teensy:
            log.info("🚀 正在啟用【虛擬 Teensy】模式...")
            if VirtualTeensy is None:
                log.error("❌ 虛擬Teensy啟動失敗：virtual_teensy.py 檔案不存在或無法導入。")
                self._set_internal_state(HWState.FAILED)
                return
            self.ser = VirtualTeensy(self.state, rate_hz=50.0)
            self.serial_comm.is_managed_by_hardware_controller = True
        else:
            log.info("🔌 正在啟用【真實硬體】模式...")
            if not self.serial_comm.is_connected:
                log.error("❌ 硬體啟動失敗：序列埠未連接。")
                self._set_internal_state(HWState.FAILED)
                return

            self.ser = self.serial_comm.get_serial_connection()
            if not self.ser:
                log.error("❌ 硬體啟動失敗：無法獲取有效連接。")
                self._set_internal_state(HWState.FAILED)
                return
            log.info(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
            self.serial_comm.is_managed_by_hardware_controller = True

        # 後續初始化流程對真實/虛擬Teensy一視同仁
        try:
            log.info("  -> 命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            log.info("  -> Teensy 模式指令已發送。")
            self._set_internal_state(HWState.RUNNING)
            self.ai_control_active = False
            with self.state.lock: self.state.hardware_ai_is_active = False
        except (serial.SerialException, AttributeError) as e:
            log.error(f"❌ 發送模式指令失敗: {e}")
            if not self.config.use_virtual_teensy:
                self.serial_comm.is_managed_by_hardware_controller = False
            self._set_internal_state(HWState.FAILED)

    def _execute_stop(self): # 執行停止流程
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False
        with self.state.lock: self.state.hardware_ai_is_active = False

        if self.ser and hasattr(self.ser, 'is_open') and self.ser.is_open:
            try:
                log.info("  -> 命令 Teensy 停止並恢復 HUMAN 模式...")
                self.ser.write(b"stop\n"); time.sleep(0.05)
                self.ser.write(b"monitor h\n"); time.sleep(0.05)
            except (serial.SerialException, AttributeError) as e:
                log.warning(f"  -> 警告: 發送停止指令失敗: {e}")
        
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            log.info("  -> 序列埠控制權已交還。")
        
        self.ser = None
        self._set_internal_state(HWState.STOPPED)

    def _execute_toggle_ai(self): # 切換AI開關
        self.ai_control_active = not self.ai_control_active
        with self.state.lock:
            self.state.hardware_ai_is_active = self.ai_control_active
        
        log.info(f"🤖 硬體 AI 控制已 {'啟用' if self.ai_control_active else '暫停'}.")
        
        if self.ai_control_active:
            self.policy.reset()
        elif self.ser and hasattr(self.ser, 'is_open') and self.ser.is_open:
            try: self.ser.write(b"stop\n")
            except (serial.SerialException, AttributeError) as e: log.error(f"發送停止指令失敗: {e}")

    def _perform_ai_step(self): # 執行單步AI運算
        """[保留 dev4.3 架構] 執行單步 AI 計算與控制。"""
        # 直接呼叫 policy manager，它會從 ObservationManager 獲取數據
        _, action_raw = self.policy.get_action_for_hardware()
        
        # 舊的 last_action 更新邏輯已移至 SimulationState.on_tick_update，這裡無需處理
        
        action_scale = self.config.initial_tuning_params.action_scale
        # 注意：硬體模式下的 default_pose 應該是零，因為 action_raw 應該是絕對角度或已經包含了姿態偏移
        default_pose_hardware = np.zeros(12)
        final_command = default_pose_hardware + action_raw * action_scale
        
        action_str = ' '.join(f"{a:.4f}" for a in final_command)
        command_to_send = f"move all {action_str}\n"

        if self.ser and hasattr(self.ser, 'is_open') and self.ser.is_open:
            try:
                self.ser.write(command_to_send.encode('utf-8'))
            except (serial.SerialException, AttributeError) as e:
                log.error(f"AI 步驟中發送指令失敗: {e}")
                self._set_internal_state(HWState.FAILED)

    def _read_from_port(self): # 從序列埠讀取資料
        log.info("[硬體讀取執行緒已啟動] 等待數據...")
        while self._is_running_event.is_set():
            if self.internal_state != HWState.RUNNING or not self.ser or not (hasattr(self.ser, 'is_open') and self.ser.is_open):
                time.sleep(0.1)
                continue

            try:
                line = self.ser.readline() # 讀取原始位元組
                if line:
                    self.parse_policy_stream(line.decode('utf-8', errors='ignore').strip()) 
            except (serial.SerialException, OSError, AttributeError) as e:
                log.error(f"❌ 讀取時序列埠斷開或出錯: {e}。將狀態設置為 FAILED。")
                self._set_internal_state(HWState.FAILED)
                break # 發生嚴重錯誤時退出讀取迴圈
            except Exception as e:
                log.error(f"❌ _read_from_port 發生未知錯誤: {e}", exc_info=True)
                self._set_internal_state(HWState.FAILED)
                break
            # 在 VirtualTeensy 模式下，readline 內部已經有頻率控制，所以這裡的 sleep 可以很短
            time.sleep(0.001)

    def parse_policy_stream(self, line: str): # 解析資料流
        """[保留 dev4.3 架構] 解析數據流並直接寫入中央 SimulationState。"""
        try:
            parts = line.split(',')
            if len(parts) != 34: return
            data_vec = np.array(parts, dtype=np.float32)
            
            # 使用全局 state.lock 保護對 SimulationState 的寫入
            with self.state.lock:
                # 數據直接寫入 SimulationState 的 raw 屬性，供 ObservationManager 使用
                self.state.raw_torso_angular_velocity_world[:] = data_vec[0:3]
                self.state.raw_gravity_vector[:] = data_vec[3:6]
                self.state.raw_accelerometer[:] = data_vec[6:9]
                self.state.raw_joint_positions[:] = data_vec[10:22]
                self.state.raw_joint_velocities[:] = data_vec[22:34]
        except (ValueError, IndexError):
            pass # 在高頻率流中，靜默忽略單次的解析錯誤