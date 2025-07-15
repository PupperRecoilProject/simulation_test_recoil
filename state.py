# state.py
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig

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
    """
    config: AppConfig
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    reset_requested: bool = False
    control_timer: float = 0.0
    
    # 模式文字，用於顯示在畫面上
    sim_mode_text: str = "Initializing"
    
    # 新增: 當前的輸入模式 (KEYBOARD 或 GAMEPAD)
    input_mode: str = "KEYBOARD"
    
    # 儲存與 ONNX 和控制相關的最新數據，供 DebugOverlay 使用
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # 顯示頁面控制
    display_page: int = 0
    num_display_pages: int = 2

    def __post_init__(self):
        """在初始化後，根據設定檔設定初始值。"""
        # 從設定檔中複製初始調校參數
        self.tuning_params = TuningParams(
            kp=self.config.initial_tuning_params.kp,
            kd=self.config.initial_tuning_params.kd,
            action_scale=self.config.initial_tuning_params.action_scale,
            bias=self.config.initial_tuning_params.bias
        )
        # 根據設定檔中的馬達數量初始化陣列
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
            self.clear_command() # 切換模式時清空指令，防止行為殘留
            print(f"模式已切換至: {self.input_mode}")