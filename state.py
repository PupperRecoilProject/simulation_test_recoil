# state.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig
from typing import TYPE_CHECKING, Any # 導入 Any 以處理循環依賴

# 使用 TYPE_CHECKING 來處理循環依賴，避免運行時錯誤
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
    """
    管理所有模擬中動態變化的狀態，取代 global 變數。
    為所有屬性提供明確的型別提示以消除 Pylance 警告。
    """
    # --- 核心設定與狀態 ---
    config: AppConfig
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # 明確提示 tuning_params 的型別，並告知 dataclass 它將在 post_init 中被初始化
    tuning_params: TuningParams = field(init=False)
    
    reset_requested: bool = False
    control_timer: float = 0.0
    
    # --- 模式相關狀態 ---
    sim_mode_text: str = "Initializing"
    input_mode: str = "KEYBOARD"
    control_mode: str = "WALKING"
    
    # --- 用於 UI 顯示的最新數據 ---
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # --- 用於輸入處理器和模式切換的最新資訊 ---
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))

    # --- UI 相關狀態 ---
    display_page: int = 0
    num_display_pages: int = 2
    
    # --- 物件引用 ---
    # 使用 TYPE_CHECKING 導入的型別來提示，消除 Pylance 警告
    floating_controller_ref: 'FloatingController' = None

    def __post_init__(self):
        """在初始化後，根據設定檔設定初始值。"""
        # 初始化 tuning_params
        self.tuning_params = TuningParams(
            kp=self.config.initial_tuning_params.kp,
            kd=self.config.initial_tuning_params.kd,
            action_scale=self.config.initial_tuning_params.action_scale,
            bias=self.config.initial_tuning_params.bias
        )
        
        # 根據設定檔中的馬達數量初始化陣列，確保維度正確
        self.latest_action_raw = np.zeros(self.config.num_motors)
        self.latest_final_ctrl = np.zeros(self.config.num_motors)
        
        print("✅ SimulationState 初始化完成。")

    def reset_control_state(self, sim_time: float):
        """重置與控制相關的狀態。"""
        self.control_timer = sim_time
        self.reset_requested = False
        print("✅ 控制狀態已重置。")

    def clear_command(self):
        """清除使用者運動指令。"""
        self.command.fill(0.0)
        print(f"運動指令已清除。目前指令: {self.command}")

    def toggle_input_mode(self, new_mode: str):
        """切換輸入模式並清除當前指令以避免衝突。"""
        if self.input_mode != new_mode:
            self.input_mode = new_mode
            self.clear_command()
            print(f"模式已切換至: {self.input_mode}")
            
    def toggle_control_mode(self, current_pos: np.ndarray, current_quat: np.ndarray):
        """
        切換固定懸浮模式。
        """
        if self.floating_controller_ref:
            is_now_floating = self.floating_controller_ref.toggle_floating_mode(current_pos, current_quat)
            self.control_mode = "FLOATING" if is_now_floating else "WALKING"