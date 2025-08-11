# hardware_controller.py (v4.0.2 - Cleaned Version)
import serial
import threading
import time
from logger import log
import numpy as np
from typing import TYPE_CHECKING
from queue import Queue, Empty
from enum import Enum, auto

from event_system import event_bus, EVENT_HARDWARE_AI_TOGGLE_REQUESTED

if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager
    from serial_communicator import SerialCommunicator
    from state import SimulationState

class RobotStateHardware:
    def __init__(self):
        self.angular_velocity_radps = np.zeros(3, dtype=np.float32)
        self.gravity_vector_norm = np.zeros(3, dtype=np.float32)
        self.accelerometer_ms2 = np.zeros(3, dtype=np.float32)
        self.pitch_rad = 0.0
        self.joint_positions_rad = np.zeros(12, dtype=np.float32)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.last_update_time = 0.0

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
    """【v4.0.2】管理硬體AI控制迴圈，採用異步命令和狀態機，確保非阻塞。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        self.config = config
        self.policy = policy
        self.state = state
        self.serial_comm = serial_comm 
        
        self.ser: serial.Serial | None = None 
        self.read_thread: threading.Thread | None = None 
        self.control_thread: threading.Thread | None = None 
        
        self.hw_state_data = RobotStateHardware()
        self.lock = threading.Lock()
        
        self._is_running_event = threading.Event()
        self.command_queue = Queue()
        self.internal_state = HWState.STOPPED
        self.last_state_change_time = time.time()
        self.ai_control_active = False

        self._subscribe_to_events()
        log.info("✅ 硬體控制器 (v4.0.2 異步版) 已初始化。")

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
        """(執行緒) 在背景持續讀取序列埠數據。"""
        log.info("[硬體讀取執行緒已啟動] 等待數據...")
        while self._is_running_event.is_set():
            if self.internal_state != HWState.RUNNING or not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
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
            log.info(f"硬體控制器狀態: {self.internal_state.name} -> {new_state.name}")
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

        self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            log.error("❌ 硬體啟動失敗：無法獲取有效連接。")
            self._set_internal_state(HWState.FAILED)
            return

        self.serial_comm.is_managed_by_hardware_controller = True
        try:
            log.info("  -> 命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1) 
            self.ser.reset_input_buffer()
            log.info("  -> Teensy 模式指令已發送。")
            self._set_internal_state(HWState.RUNNING)
            self.ai_control_active = False
            with self.state.lock: self.state.hardware_ai_is_active = False
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
        """(內部) 執行單步 AI 計算與控制。"""
        observation = self.construct_observation()
        if observation.size == 0: return
            
        _, action_raw = self.policy.get_action_for_hardware(observation)
        
        with self.lock:
            self.hw_state_data.last_action[:] = action_raw
        
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
        except (ValueError, IndexError):
            pass # 在高頻率流中，忽略解析錯誤比打印日誌更好

    def construct_observation(self) -> np.ndarray:
        with self.lock:
            obs_components = {
                'angular_velocity': self.hw_state_data.angular_velocity_radps,
                'gravity_vector': self.hw_state_data.gravity_vector_norm,
                'accelerometer': self.hw_state_data.accelerometer_ms2,
                'pitch': np.array([self.hw_state_data.pitch_rad]),
                'joint_positions': self.hw_state_data.joint_positions_rad,
                'joint_velocities': self.hw_state_data.joint_velocities_radps,
                'last_action': self.hw_state_data.last_action,
                'commands': self.hw_state_data.command,
                'linear_velocity': np.zeros(3), 
            }
            
        recipe = self.policy.get_active_recipe()
        if not recipe: return np.array([])
        
        try:
            final_obs_list = [obs_components[key] for key in recipe]
            return np.concatenate(final_obs_list).astype(np.float32)
        except KeyError as e:
            log.error(f"❌ 觀察向量構建失敗: 缺少 '{e}'。")
            return np.array([])