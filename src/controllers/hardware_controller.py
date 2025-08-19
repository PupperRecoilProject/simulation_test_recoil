# src/controllers/hardware_controller.py

# 【v4.3.2 修改】 調整 import，移除不再需要的 Enum
import serial
import threading
import time
from src.core.logger import log
import numpy as np
from typing import TYPE_CHECKING
from queue import Queue, Empty  # 引入佇列，用於實現執行緒安全的命令傳遞
from enum import Enum, auto  # 引入枚舉，用於定義清晰的狀態和命令

# 導入專案內部模組
from src.core.logger import log
from src.core.event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED

# 每 LOG_EVERY_N 筆資料列印一次，避免終端被大量輸出淹沒
LOG_EVERY_N = 50


def construct_observation_51(state, hw):
    """將 34 維硬體資料 + 內部狀態組裝成 51 維觀測。"""
    # linear_velocity(3)：實體端無此量，虛擬模式可從模擬器取得
    if getattr(state.config, "use_virtual_teensy", False) and hasattr(state.sim, "linear_velocity_local"):
        lin_vel = np.asarray(state.sim.linear_velocity_local(), dtype=np.float32)
    else:
        lin_vel = np.zeros(3, dtype=np.float32)

    ang_vel = np.asarray(hw.angular_velocity_radps, dtype=np.float32)
    g_vec = np.asarray(hw.gravity_vector_norm, dtype=np.float32)
    accel = np.asarray(hw.accelerometer_ms2, dtype=np.float32)
    qpos = np.asarray(hw.joint_positions_rad, dtype=np.float32)
    qvel = np.asarray(hw.joint_velocities_radps, dtype=np.float32)

    last_action = np.asarray(getattr(state, "last_action", np.zeros(12, np.float32)), dtype=np.float32)
    cmd = np.asarray(getattr(state, "command", np.zeros(3, np.float32)), dtype=np.float32)

    scale = getattr(getattr(state, "tuning_params", None), "command_scale", [1.0, 1.0, 1.0])
    if isinstance(scale, (list, tuple, np.ndarray)) and len(scale) == 3:
        cmd = cmd * np.asarray(scale, dtype=np.float32)

    obs = np.concatenate([lin_vel, ang_vel, g_vec, accel, qpos, qvel, last_action, cmd]).astype(np.float32)
    assert obs.shape[0] == 51, f"觀測維度應為 51，實得 {obs.shape[0]}"
    return obs

# 類型檢查區塊，僅在靜態分析時執行，避免循環導入
if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.hardware.policy import PolicyManager
    from src.hardware.serial_communicator import SerialCommunicator
    from src.core.state import SimulationState

# 【v4.3.2 刪除】 RobotStateHardware 類別
# 這個類別的所有狀態都已遷移到 SimulationState 的 raw_... 屬性中，
# 以實現統一的數據流。


class HWCommand(Enum):
    """定義可以發送給硬體控制器的命令類型。"""

    START = auto()  # 啟動命令
    STOP = auto()  # 停止命令
    TOGGLE_AI = auto()  # 切換AI開/關命令


class HWState(Enum):
    """定義硬體控制器內部的狀態機狀態。"""

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
        """訂閱來自事件系統的外部事件，並將其轉換為內部命令放入隊列。"""
        # 當收到外部的AI切換請求時，將一個TOGGLE_AI命令放入隊列，由控制迴圈處理
        event_bus.subscribe(
            EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
            lambda: self.command_queue.put(HWCommand.TOGGLE_AI),
        )
        log.info("  -> HardwareController 已訂閱 AI 切換請求事件。")

    def request_start(self) -> None:
        """(外部API, 非阻塞) 請求啟動硬體控制器。"""
        # 只有在控制器處於可以啟動的狀態時才接受請求
        if self.internal_state in [HWState.STOPPED, HWState.FAILED]:
            log.info("收到啟動請求，向控制執行緒發送 START 命令。")
            self._start_threads_if_not_alive()  # 確保執行緒已啟動
            self.command_queue.put(HWCommand.START)  # 發送命令
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略啟動請求。")

    def request_stop(self) -> None:
        """(外部API, 非阻塞) 請求停止硬體控制器。"""
        # 只有在控制器正在運行時才接受停止請求
        if self.internal_state == HWState.RUNNING:
            log.info("收到停止請求，向控制執行緒發送 STOP 命令。")
            self.command_queue.put(HWCommand.STOP)
        else:
            log.warning(f"當前狀態為 {self.internal_state.name}，忽略停止請求。")

    def shutdown(self):
        """(外部API, 阻塞) 應用程式關閉時的強制清理。"""
        self._is_running_event.clear()  # 通知所有執行緒退出迴圈
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)  # 等待執行緒結束
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        log.info("硬體控制器所有執行緒已關閉。")

    def _start_threads_if_not_alive(self):
        """(內部) 確保背景執行緒只被創建和啟動一次。"""
        self._is_running_event.set()  # 設定運行信號
        if not self.control_thread or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(
                target=self._control_loop, daemon=True
            )
            self.control_thread.start()
            log.info("硬體控制執行緒已啟動。")

        if not self.read_thread or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(
                target=self._read_from_port, daemon=True
            )
            self.read_thread.start()
            log.info("硬體讀取執行緒已啟動。")

    def _set_internal_state(self, new_state: HWState):
        """(內部) 安全地切換狀態機並同步到全局 State。"""
        if self.internal_state != new_state:
            log.info(f"硬體控制器狀態: {self.internal_state.name} -> {new_state.name}")
            self.internal_state = new_state
            # 將運行狀態同步到全局 state，供UI等模組讀取
            with self.state.lock:
                self.state.hardware_is_running = new_state == HWState.RUNNING

    def _control_loop(self):
        """(執行緒) 狀態機驅動者和命令派發中心。"""
        log.info("--- 硬體控制執行緒已就緒，等待命令 ---")
        while self._is_running_event.is_set():
            # 檢查命令隊列
            try:
                command: HWCommand = self.command_queue.get_nowait()
                if command == HWCommand.START and self.internal_state in [
                    HWState.STOPPED,
                    HWState.FAILED,
                ]:
                    self._execute_start()
                elif (
                    command == HWCommand.STOP and self.internal_state == HWState.RUNNING
                ):
                    self._execute_stop()
                elif (
                    command == HWCommand.TOGGLE_AI
                    and self.internal_state == HWState.RUNNING
                ):
                    self._execute_toggle_ai()
            except Empty:
                pass  # 隊列為空，無事發生

            # 如果處於運行狀態且AI已啟用，則執行AI決策
            if self.internal_state == HWState.RUNNING and self.ai_control_active:
                self._perform_ai_step()

            # 維持固定的迴圈頻率
            time.sleep(1.0 / self.config.control_freq)

    def _execute_start(self):
        """(內部) 執行啟動流程，包含真實/虛擬模式的選擇。"""
        self._set_internal_state(HWState.STARTING)

        if self.config.use_virtual_teensy:
            log.info("🚀 正在啟用【虛擬 Teensy】模式...")
            # 避免在模擬器缺失時發生導入錯誤，於此處動態導入
            from src.hardware.virtual_teensy import VirtualTeensy
            self.ser = VirtualTeensy(self.state, rate_hz=50.0)
            # 虛擬模式也視為已接管序列埠，避免 SerialCommunicator 介入
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
            self.ai_control_active = False  # 啟動後AI預設關閉
            with self.state.lock:
                self.state.hardware_ai_is_active = False
        except (serial.SerialException, AttributeError) as e:
            log.error(f"❌ 發送模式指令失敗: {e}")
            if not self.config.use_virtual_teensy:
                self.serial_comm.is_managed_by_hardware_controller = False
            self._set_internal_state(HWState.FAILED)

    def _execute_stop(self):
        """(內部) 執行停止流程。"""
        self._set_internal_state(HWState.STOPPING)
        self.ai_control_active = False
        with self.state.lock:
            self.state.hardware_ai_is_active = False

        if self.ser and self.ser.is_open:
            try:
                log.info("  -> 命令 Teensy 停止並恢復 HUMAN 模式...")
                self.ser.write(b"stop\n")
                time.sleep(0.05)
                self.ser.write(b"monitor h\n")
                time.sleep(0.05)
            except (serial.SerialException, AttributeError) as e:
                log.warning(f"  -> 警告: 發送停止指令失敗: {e}")

        # 真實與虛擬模式皆需要釋放控制權
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
            self.policy.reset()  # 啟用時重置策略歷史
        elif self.ser and self.ser.is_open:
            try:
                self.ser.write(b"stop\n")  # 暫停時發送停止指令
            except (serial.SerialException, AttributeError) as e:
                log.error(f"發送停止指令失敗: {e}")

    # 【v4.3.2 修改】 _perform_ai_step 方法
    def _perform_ai_step(self):
        """(內部) 執行單步 AI 計算與控制。"""
        # 【v4.3.2 刪除】 不再自行構建觀測向量
        # observation = self.construct_observation()
        # if observation.size == 0: return

        # 【v4.3.2 修改】 直接呼叫 get_action_for_hardware，無需傳遞參數。
        # PolicyManager 會自動從 ObservationManager 獲取數據。
        _, action_raw = self.policy.get_action_for_hardware()
        
        # 【v4.3.2 刪除】 last_action 的更新已轉移到 SimulationState.on_tick_update
        # with self.lock:
        #     self.hw_state_data.last_action[:] = action_raw
        
        action_scale = self.config.initial_tuning_params.action_scale
        default_pose_hardware = np.zeros(12)
        final_command = default_pose_hardware + action_raw * action_scale
        
        action_str = ' '.join(f"{a:.4f}" for a in final_command)
        command_to_send = f"move all {action_str}\n"

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(command_to_send.encode("utf-8"))
            except (serial.SerialException, AttributeError):
                log.error("AI 步驟中發送指令失敗，連接可能已斷開。")
                self._set_internal_state(HWState.FAILED)

    # 【v4.3.2 修改】 parse_policy_stream 方法
    def parse_policy_stream(self, line: str):
        """
        【v4.3.2 修改】
        解析來自Teensy的數據流，並將原始數據直接寫入中央的SimulationState。
        """
        try:
            parts = line.split(",")
            if len(parts) != 34:
                return
            data_vec = np.array(parts, dtype=np.float32)
            
            # 使用全局 state.lock 保護對 SimulationState 的寫入
            with self.state.lock:
                # 注意：Teensy 發送的是世界座標系下的數據，這裡直接寫入
                self.state.raw_torso_angular_velocity_world[:] = data_vec[0:3]
                # Teensy 通常直接發送已經計算好的重力向量（相對於機身）
                self.state.raw_gravity_vector[:] = data_vec[3:6]
                self.state.raw_accelerometer[:] = data_vec[6:9]
                # pitch 數據暫時不存入 raw 狀態，因為通常可以從四元數推導
                # self.state.raw_pitch_rad = data_vec[9] 
                self.state.raw_joint_positions[:] = data_vec[10:22]
                self.state.raw_joint_velocities[:] = data_vec[22:34]
                # 注意：四元數和線速度並不在這個數據流中，ObservationManager 需要能處理這種情況
                # （例如，在硬體模式下，linear_velocity 可能為零）

        except (ValueError, IndexError):
            pass  # 在高頻率流中，靜默忽略單次的解析錯誤

    # 【v4.3.2 刪除】 construct_observation 方法
    # 此方法的職責已完全轉移給 ObservationManager。