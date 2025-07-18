# state.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from floating_controller import FloatingController
    from policy import ONNXPolicy

@dataclass
class TuningParams:
    """用於即時調整機器人控制參數的類別。"""
    kp: float
    kd: float
    action_scale: float
    bias: float

@dataclass
class SimulationState:
    """管理所有模擬中動態變化的狀態，取代 global 變數。"""
    config: AppConfig
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    
    hard_reset_requested: bool = False
    soft_reset_requested: bool = False

    control_timer: float = 0.0
    
    sim_mode_text: str = "Initializing"
    input_mode: str = "KEYBOARD"
    control_mode: str = "WALKING"

    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    display_page: int = 0
    num_display_pages: int = 2

    serial_command_buffer: str = ""
    serial_command_to_send: str = ""
    serial_latest_messages: list = field(default_factory=list)

    joint_test_index: int = 0
    joint_test_offsets: np.ndarray = field(default_factory=lambda: np.zeros(12))

    manual_ctrl_index: int = 0
    manual_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    manual_mode_is_floating: bool = False

    serial_is_connected: bool = False
    gamepad_is_connected: bool = False

    tuning_param_index: int = 0

    floating_controller_ref: 'FloatingController' = None
    policy_ref: 'ONNXPolicy' = None

    single_step_mode: bool = False
    execute_one_step: bool = False

    def __post_init__(self):
        """在初始化後，根據設定檔設定初始值。"""
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__)
        self.latest_action_raw = np.zeros(self.config.num_motors)
        self.latest_final_ctrl = np.zeros(self.config.num_motors)
        self.manual_final_ctrl = np.zeros(self.config.num_motors)
        print("✅ SimulationState 初始化完成。")

    def reset_control_state(self, sim_time: float):
        self.control_timer = sim_time
        print("✅ 控制狀態已重置。")

    def clear_command(self):
        self.command.fill(0.0)
        print("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str):
        if self.input_mode != new_mode:
            self.input_mode = new_mode
            self.clear_command()
            print(f"輸入模式已切換至: {self.input_mode}")
            
    def set_control_mode(self, new_mode: str):
        """切換主控制模式，並呼叫對應的啟用/禁用函式。"""
        if self.control_mode == new_mode: return

        old_mode = self.control_mode
        
        # --- 離開舊模式時的清理工作 ---
        if old_mode == "FLOATING":
            if self.floating_controller_ref: self.floating_controller_ref.disable()
        elif old_mode == "MANUAL_CTRL" and self.manual_mode_is_floating:
             if self.floating_controller_ref: self.floating_controller_ref.disable()
             self.manual_mode_is_floating = False
        
        self.control_mode = new_mode
        print(f"控制模式已切換至: {self.control_mode}")

        # --- 進入新模式時的設定工作 ---
        if new_mode == "FLOATING":
            if self.floating_controller_ref: self.floating_controller_ref.enable(self.latest_pos)
        elif new_mode == "JOINT_TEST":
            self.joint_test_offsets.fill(0.0)
        elif new_mode == "MANUAL_CTRL":
            self.manual_final_ctrl[:] = self.latest_final_ctrl

        # --- 【核心】模式切換穩定性修復 ---
        # 如果是從手動模式切換回 AI 模式
        is_entering_ai_mode = new_mode in ["WALKING", "FLOATING"]
        is_leaving_manual_mode = old_mode in ["JOINT_TEST", "MANUAL_CTRL"]
        
        if is_entering_ai_mode and is_leaving_manual_mode:
            print("從手動模式返回，正在重置 AI 狀態以確保平滑過渡...")
            if self.policy_ref:
                self.policy_ref.reset()
            self.clear_command()