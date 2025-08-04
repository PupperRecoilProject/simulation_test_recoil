from __future__ import annotations
"""Central state module with clear operating modes."""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from typing import TYPE_CHECKING, Any

from utils.config import AppConfig
from utils.logger import log

if TYPE_CHECKING:
    from core.simulation import Simulation

# ========================
# Mode Enumerations  模式列舉
# ========================
class OperatingMode(Enum):
    """頂層操作模式: 模擬 or 真實硬體"""
    SIMULATION = auto()
    HARDWARE = auto()

class ControlSubMode(Enum):
    """控制子模式, both for sim & hardware"""
    WALKING = auto()       # AI 走路
    FLOATING = auto()      # AI 懸浮
    JOINT_TEST = auto()    # 手動關節測試
    MANUAL_CTRL = auto()   # 手動姿態控制
    IDLE = auto()          # 待機

# ========================
# Hardware State  硬體狀態
# ========================
@dataclass
class HardwareState:
    """即時硬體相關數據"""
    is_connected: bool = False
    ai_is_active: bool = False
    status_text: str = "Not Connected"

    # Teensy 端的感測值
    angular_velocity_radps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_positions_rad: np.ndarray = field(default_factory=lambda: np.zeros(12))
    joint_velocities_radps: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # PC 端為硬體準備的資訊 (for UI)
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))

# ========================
# Tuning Params   調校參數
# ========================
@dataclass
class TuningParams:
    kp: float
    kd: float
    action_scale: float
    bias: float
    bias_enabled: bool = False  # 是否啟用偏壓力矩, 預設關閉以更安全

# ========================
# Simulation State  主狀態容器
# ========================
@dataclass
class SimulationState:
    config: AppConfig
    lock: threading.Lock = field(default_factory=threading.Lock)

    operating_mode: OperatingMode = OperatingMode.SIMULATION
    control_sub_mode: ControlSubMode = ControlSubMode.WALKING

    # 專用硬體狀態
    hardware: HardwareState = field(default_factory=HardwareState)

    # 模擬專用最新資料
    sim_latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    sim_latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    sim_latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sim_latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.], dtype=np.float32))
    sim_latest_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # 使用者命令與調校參數
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    input_mode: str = "KEYBOARD"

    # 地形相關設定 Terrain
    terrain_mode: str = "INFINITE"            # 地形模式: 無限或單一
    single_terrain_index: int = 0              # 單一地形索引

    # 手動控制相關
    joint_test_offsets: np.ndarray = field(default_factory=lambda: np.zeros(12))
    manual_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    manual_mode_is_floating: bool = False
    joint_test_index: int = 0               # 目前選中的關節索引
    manual_ctrl_index: int = 0              # 手動控制模式下的關節索引

    # UI 旗標
    hard_reset_requested: bool = False
    soft_reset_requested: bool = False
    single_step_mode: bool = False
    execute_one_step: bool = False
    shutdown_requested: bool = False
    serial_is_connected: bool = False      # 序列埠是否已連接
    gamepad_is_connected: bool = False     # 搖桿是否已連接
    serial_command_buffer: str = ""        # 序列埠輸入緩衝
    tuning_param_index: int = 0            # 目前調整參數索引
    display_page: int = 0                  # 目前顯示頁面
    num_display_pages: int = 1             # 顯示頁面數量
    previous_control_mode: str = "WALKING"  # 追蹤前一個模式 (相容舊介面)
    control_mode_pending: str | None = None # 待切換模式 (相容舊介面)

    # 內部變數: 控制頻率，可動態調整
    _control_freq: float = field(init=False)

    # 外部模組參考 (References)
    sim: Simulation | None = None
    policy_manager_ref: Any = None
    hardware_controller_ref: Any = None
    serial_communicator_ref: Any = None
    xbox_handler_ref: Any = None
    terrain_manager_ref: Any = None
    floating_controller_ref: Any = None

    def __post_init__(self) -> None:
        """初始化後設置調校參數與控制頻率"""
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__)
        self._control_freq = self.config.control_freq
        log.info("✅ 重構版 SimulationState 初始化完成 (含便利方法與動態控制頻率)。")

    # =============
    # Convenience Methods 便利方法
    # =============
    def clear_command(self) -> None:
        """清除使用者輸入的運動指令"""
        with self.lock:
            self.command.fill(0.0)
        log.info("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str, clear_cmd: bool = True) -> None:
        """切換輸入模式，可選擇是否清除指令"""
        with self.lock:
            if self.input_mode != new_mode:
                self.input_mode = new_mode
                if clear_cmd:
                    self.command.fill(0.0)
                log.info(f"輸入模式已切換至: {self.input_mode}")

    def request_mode_change(self, new_op: OperatingMode, new_sub: ControlSubMode) -> None:
        """統一的模式切換接口"""
        with self.lock:
            self.operating_mode = new_op
            self.control_sub_mode = new_sub
        log.info(f"模式已切換至: {new_op.name}/{new_sub.name}")

    def request_sub_mode_change(self, new_sub: ControlSubMode) -> None:
        """僅改變控制子模式的便捷函式"""
        with self.lock:
            self.control_sub_mode = new_sub
        log.info(f"控制子模式已請求切換至: {new_sub.name}")

    # -------- 動態控制頻率與週期 --------
    @property
    def control_freq(self) -> float:
        """取得目前控制頻率 Hz"""
        with self.lock:
            return self._control_freq

    @control_freq.setter
    def control_freq(self, value: float) -> None:
        """更新控制頻率, 會同步改變 control_dt"""
        if value <= 0:
            return
        with self.lock:
            self._control_freq = value
        log.info(f"控制頻率已更新為 {value} Hz (dt={self.control_dt:.4f}s)")

    @property
    def control_dt(self) -> float:
        """根據控制頻率計算控制週期秒數"""
        with self.lock:
            return 1.0 / self._control_freq if self._control_freq > 0 else float('inf')

    # ---- Legacy compatibility methods ----
    def get_control_mode_string(self) -> str:
        """以舊字串格式回傳目前模式"""
        if self.operating_mode == OperatingMode.HARDWARE:
            return "HARDWARE_MODE"
        return self.control_sub_mode.name

    def set_control_mode(self, mode: str) -> None:
        """相容舊介面的模式設定函式"""
        mapping = {
            "WALKING": (OperatingMode.SIMULATION, ControlSubMode.WALKING),
            "FLOATING": (OperatingMode.SIMULATION, ControlSubMode.FLOATING),
            "JOINT_TEST": (OperatingMode.SIMULATION, ControlSubMode.JOINT_TEST),
            "MANUAL_CTRL": (OperatingMode.SIMULATION, ControlSubMode.MANUAL_CTRL),
            "HARDWARE_MODE": (OperatingMode.HARDWARE, ControlSubMode.IDLE),
        }
        with self.lock:
            self.previous_control_mode = self.get_control_mode_string()
        op, sub = mapping.get(mode, (self.operating_mode, self.control_sub_mode))
        self.request_mode_change(op, sub)

    # 提供舊屬性接口，避免舊模組存取失敗
    @property
    def control_mode(self) -> str:  # 讀取時回傳舊格式字串
        return self.get_control_mode_string()

    @control_mode.setter
    def control_mode(self, mode: str) -> None:  # 寫入時自動映射到新枚舉
        self.set_control_mode(mode)

    # Legacy alias properties for backward compatibility -----------------
    @property
    def latest_pos(self) -> np.ndarray:
        return self.sim_latest_pos

    @latest_pos.setter
    def latest_pos(self, value: np.ndarray) -> None:
        self.sim_latest_pos = value

    @property
    def latest_joint_positions(self) -> np.ndarray:
        return self.sim_latest_joint_positions

    @latest_joint_positions.setter
    def latest_joint_positions(self, value: np.ndarray) -> None:
        self.sim_latest_joint_positions = value

    @property
    def latest_onnx_input(self) -> np.ndarray:
        return self.sim_latest_onnx_input

    @latest_onnx_input.setter
    def latest_onnx_input(self, value: np.ndarray) -> None:
        self.sim_latest_onnx_input = value

    @property
    def latest_action_raw(self) -> np.ndarray:
        return self.sim_latest_action_raw

    @latest_action_raw.setter
    def latest_action_raw(self, value: np.ndarray) -> None:
        self.sim_latest_action_raw = value

    @property
    def latest_final_ctrl(self) -> np.ndarray:
        return self.sim_latest_final_ctrl

    @latest_final_ctrl.setter
    def latest_final_ctrl(self, value: np.ndarray) -> None:
        self.sim_latest_final_ctrl = value
