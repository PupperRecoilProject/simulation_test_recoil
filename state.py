from __future__ import annotations
"""Central state module with clear operating modes."""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from typing import TYPE_CHECKING

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
    sim_latest_quat: np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))
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

    # UI 旗標
    hard_reset_requested: bool = False
    soft_reset_requested: bool = False
    single_step_mode: bool = False
    execute_one_step: bool = False
    shutdown_requested: bool = False

    # 其他參考
    sim: Simulation | None = None

    def __post_init__(self) -> None:
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__)
        log.info("✅ 重構版 SimulationState 初始化完成。")

    # =============
    # Mode change
    # =============
    def request_mode_change(self, new_op: OperatingMode, new_sub: ControlSubMode) -> None:
        """統一的模式切換接口"""
        with self.lock:
            self.operating_mode = new_op
            self.control_sub_mode = new_sub
        log.info(f"模式已切換至: {new_op.name}/{new_sub.name}")
