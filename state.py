# state.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from floating_controller import FloatingController

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
    
    # --- 重置旗標 ---
    hard_reset_requested: bool = False # R 鍵觸發的完全重置
    soft_reset_requested: bool = False # X 鍵觸發的空中姿態重置

    control_timer: float = 0.0
    
    # --- 模式狀態 ---
    sim_mode_text: str = "Initializing"
    input_mode: str = "KEYBOARD"
    control_mode: str = "WALKING"  # 可選值: "WALKING", "FLOATING", "SERIAL_MODE", "JOINT_TEST"

    # --- UI & 跨模組資料 ---
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    display_page: int = 0
    num_display_pages: int = 2

    # --- 序列埠模式相關狀態 ---
    serial_command_buffer: str = ""
    serial_command_to_send: str = ""
    serial_latest_messages: list = field(default_factory=list)

    # --- 關節手動測試模式相關狀態 ---
    joint_test_index: int = 0
    joint_test_offsets: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # --- 物件引用 ---
    floating_controller_ref: 'FloatingController' = None

    # --- 單步執行模式 ---
    single_step_mode: bool = False
    execute_one_step: bool = False

    def __post_init__(self):
        """在初始化後，根據設定檔設定初始值。"""
        self.tuning_params = TuningParams(
            kp=self.config.initial_tuning_params.kp,
            kd=self.config.initial_tuning_params.kd,
            action_scale=self.config.initial_tuning_params.action_scale,
            bias=self.config.initial_tuning_params.bias
        )
        self.latest_action_raw = np.zeros(self.config.num_motors)
        self.latest_final_ctrl = np.zeros(self.config.num_motors)
        print("✅ SimulationState 初始化完成。")

    def reset_control_state(self, sim_time: float):
        self.control_timer = sim_time
        # 注意：這裡不再重置 reset_requested 旗標，由主迴圈處理
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

        if self.control_mode == "FLOATING":
            if self.floating_controller_ref and self.floating_controller_ref.is_functional:
                self.floating_controller_ref.disable()
        
        self.control_mode = new_mode
        print(f"控制模式已切換至: {self.control_mode}")

        if new_mode == "FLOATING":
            if self.floating_controller_ref and self.floating_controller_ref.is_functional:
                self.floating_controller_ref.enable(self.latest_pos)
        elif new_mode == "JOINT_TEST":
            self.joint_test_offsets.fill(0.0)