# src/core/state.py

# ... imports 和 TuningParams dataclass 保持不變 ...
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from src.core.config import AppConfig
from src.core.logger import log
from typing import TYPE_CHECKING
import threading

from src.core.event_system import (
    event_bus,
    EventSystem,
    EVENT_SIMULATION_TICK,
    EVENT_HARDWARE_TICK,
    EVENT_COMMAND_UPDATED,
    EVENT_MODE_CHANGED,
)

if TYPE_CHECKING:
    from src.simulation.floating_controller import FloatingController
    from src.hardware.policy import PolicyManager
    from src.controllers.hardware_controller import HardwareController
    from src.simulation.terrain_manager import TerrainManager
    from src.hardware.serial_communicator import SerialCommunicator
    from src.simulation.simulation import Simulation
    from src.input_handlers.xbox_input_handler import XboxInputHandler

@dataclass
class TuningParams:
    """用於即時調整機器人控制參數的類別。"""
    kp: float
    kd: float
    action_scale: float
    bias: float


@dataclass
class SimulationState:
    """
    [合併後的版本] 中央狀態管理者 (Central State Manager)
    
    這個類別是整個應用程式的"單一真相來源"。它整合了新架構的數據流
    和虛擬Teensy所需的所有狀態欄位。
    """

    # --- 核心屬性 ---
    config: AppConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: EventSystem = field(default_factory=lambda: event_bus, repr=False)
    
    # --- 用戶輸入與指令狀態 ---
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    
    # --- 系統控制與模式狀態 ---
    control_mode: str = "WALKING"
    mode_change_request: str | None = None
    previous_control_mode: str = "WALKING"
    input_mode: str = "KEYBOARD"
    
    # --- 運行時高頻數據 ---
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # --- 物理狀態 ---
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    latest_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # --- [合併] 原始感測器數據 (Raw sensor data) ---
    # 這是 ObservationManager 的唯一數據源，由 SimulationController 或 HardwareController 填充。
    raw_torso_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    raw_torso_linear_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_torso_angular_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_gravity_vector: np.ndarray = field(default_factory=lambda: np.zeros(3)) # 從 fake-Teensy 分支合併
    raw_last_action: np.ndarray = field(default_factory=lambda: np.zeros(12)) # 從 dev4.3 分支合併
    
    # --- 請求旗標 ---
    hard_reset_requested: bool = False
    soft_reset_requested: bool = False
    shutdown_requested: bool = False
    manual_float_toggle_request: bool | None = None
    
    # --- 模擬器特定狀態 ---
    control_timer: float = 0.0
    single_step_mode: bool = False
    execute_one_step: bool = False
    
    # --- 地形相關狀態 ---
    terrain_mode: str = "INFINITE"
    single_terrain_index: int = 0
    
    # --- 手動與測試模式狀態 ---
    joint_test_index: int = 0
    joint_test_offsets: np.ndarray = field(default_factory=lambda: np.zeros(12))
    manual_ctrl_index: int = 0
    manual_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    manual_mode_is_floating: bool = False
    
    # --- 設備連接與狀態 ---
    serial_is_connected: bool = False
    gamepad_is_connected: bool = False
    ui_gamepad_connected: bool = False
    hardware_is_running: bool = False
    hardware_ai_is_active: bool = False
    hardware_status_text: str = "Not Connected"
    
    # --- UI 相關狀態 ---
    display_page: int = 0
    num_display_pages: int = 2
    tuning_param_index: int = 0
    
    # --- 模組參考 ---
    sim: 'Simulation' = None
    floating_controller_ref: 'FloatingController' = None
    terrain_manager_ref: 'TerrainManager' = None
    policy_manager_ref: 'PolicyManager' = None
    hardware_controller_ref: 'HardwareController' = None
    serial_communicator_ref: 'SerialCommunicator' = None
    xbox_handler_ref: 'XboxInputHandler' = None
    available_policies: list = field(default_factory=list)


    def __post_init__(self): # 後置初始化函式
        """
        初始化函式，設定初始值並訂閱事件。
        """
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__)
        self.latest_action_raw = np.zeros(self.config.num_motors)
        self.latest_final_ctrl = np.zeros(self.config.num_motors)
        self.manual_final_ctrl = np.zeros(self.config.num_motors)
        self.latest_joint_positions = np.zeros(self.config.num_motors)

        # [合併] 初始化所有 raw 屬性
        self.raw_joint_positions = np.zeros(self.config.num_motors)
        self.raw_joint_velocities = np.zeros(self.config.num_motors)
        self.raw_last_action = np.zeros(self.config.num_motors)

        # 訂閱核心數據更新事件
        self.events.subscribe(EVENT_SIMULATION_TICK, self.on_tick_update)
        self.events.subscribe(EVENT_HARDWARE_TICK, self.on_tick_update)
        self.events.subscribe(EVENT_COMMAND_UPDATED, self.on_command_update)
        self.events.subscribe(EVENT_MODE_CHANGED, self.on_mode_changed)

        log.info("✅ SimulationState 初始化完成，並已訂閱核心事件。")

    def on_tick_update(self, onnx_input: np.ndarray, action_raw: np.ndarray, final_ctrl: np.ndarray): # TICK事件處理函式
        """
        [保留 dev4.3 架構] 收到TICK事件時，安全地更新與AI決策相關的狀態。
        """
        with self.lock:
            self.latest_onnx_input = onnx_input
            self.latest_action_raw = action_raw
            self.latest_final_ctrl = final_ctrl
            # 統一更新 raw_last_action，這是 ObservationManager 的數據源
            self.raw_last_action = action_raw

    def on_command_update(self, command: np.ndarray): # 指令更新事件處理函式
        """當收到指令更新事件時，更新指令狀態。"""
        with self.lock:
            self.command = command.copy()

    def on_mode_changed(self, old_mode: str, new_mode: str): # 模式切換事件處理函式
        """
        當控制模式成功切換後，執行相關的狀態清理和初始化。
        """
        with self.lock:
            if new_mode == "JOINT_TEST":
                self.joint_test_offsets.fill(0.0)
            elif new_mode == "MANUAL_CTRL":
                initial_pose = self.sim.default_pose.copy() if hasattr(self.sim, 'default_pose') else np.zeros(self.config.num_motors)
                self.manual_final_ctrl[:] = initial_pose

            is_entering_ai_mode = new_mode in ["WALKING", "FLOATING"]
            is_leaving_manual_mode = old_mode in ["JOINT_TEST", "MANUAL_CTRL", "SERIAL_MODE"]
            if is_entering_ai_mode and is_leaving_manual_mode:
                log.info("從手動/序列埠模式返回，正在重置 AI 狀態...")
                if self.policy_manager_ref:
                    self.policy_manager_ref.reset()
                self.clear_command()

    def set_control_mode(self, new_mode: str): # 設定控制模式
        """
        僅更新模式變數，不包含副作用。
        """
        if self.control_mode == new_mode:
            return
        self.previous_control_mode = self.control_mode
        self.control_mode = new_mode
        log.info(f"內部狀態: 控制模式已設定為: {self.control_mode}")

    def reset_control_state(self, sim_time: float): # 重置控制狀態
        """重置控制迴圈的計時器。"""
        self.control_timer = sim_time
        log.info("✅ 控制狀態已重置。")

    def clear_command(self): # 清除指令
        """清除使用者輸入的運動指令。"""
        self.command.fill(0.0)
        log.info("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str, clear_cmd: bool = True): # 切換輸入模式
        """切換輸入模式，可選擇是否清除現有指令。"""
        with self.lock:
            if self.input_mode != new_mode:
                self.input_mode = new_mode
                if clear_cmd:
                    self.clear_command()
                log.info(f"輸入模式已切換至: {self.input_mode}")