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
    reset_requested: bool = False
    control_timer: float = 0.0
    
    # --- 模式狀態 ---
    sim_mode_text: str = "Initializing" # 用於UI顯示的模式文字
    input_mode: str = "KEYBOARD"        # 輸入設備模式: KEYBOARD / GAMEPAD
    control_mode: str = "WALKING"       # 控制模式: WALKING / FLOATING

    # --- UI & 跨模組資料 ---
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    display_page: int = 0
    num_display_pages: int = 2
    
    # --- 物件引用 ---
    floating_controller_ref: 'FloatingController' = None

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
        self.reset_requested = False
        print("✅ 控制狀態已重置。")

    def clear_command(self):
        self.command.fill(0.0)
        print(f"運動指令已清除。")

    def toggle_input_mode(self, new_mode: str):
        if self.input_mode != new_mode:
            self.input_mode = new_mode
            self.clear_command()
            print(f"輸入模式已切換至: {self.input_mode}")
            
    def toggle_control_mode(self, current_pos: np.ndarray, current_quat: np.ndarray):
        """切換固定懸浮模式 (WALKING <-> FLOATING)。"""
        if self.floating_controller_ref and self.floating_controller_ref.is_functional:
            is_now_floating = self.floating_controller_ref.toggle_floating_mode(current_pos, current_quat)
            self.control_mode = "FLOATING" if is_now_floating else "WALKING"
        else:
            print("❌ 懸浮控制器不可用，無法切換模式。")