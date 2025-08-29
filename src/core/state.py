# state.py
"""
【v2.0 修改後】中央狀態管理者 (Central State Manager)
【v4.4.2 修改】新增 Teensy 數據流相關的原始屬性。

這個類別是整個應用程式的"單一真相來源 (Single Source of Truth)"。
它不再是一個被動的數據容器，而是一個主動的管理者，通過訂閱事件來
安全地更新自己的狀態，並為其他模組提供查詢。
"""

# 所有現有的 imports
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from src.core.config import AppConfig
from src.core.logger import log
from typing import TYPE_CHECKING, Dict
import threading

# 導入我們新創建的事件系統模組和核心事件
# 這是讓 SimulationState 能夠監聽系統事件的關鍵
from src.core.event_system import (
    event_bus,
    EventSystem,  # 導入 EventSystem 類別以進行類型提示
    EVENT_SIMULATION_TICK,
    EVENT_HARDWARE_TICK,
    EVENT_COMMAND_UPDATED,
    EVENT_MODE_CHANGED,
)



# TYPE_CHECKING 區塊
if TYPE_CHECKING:
    # 這些導入僅用於類型提示，不會造成循環依賴
    from src.simulation.floating_controller import FloatingController
    from src.hardware.policy import PolicyManager
    from src.controllers.hardware_controller import HardwareController
    from src.simulation.terrain_manager import TerrainManager
    from src.hardware.serial_communicator import SerialCommunicator
    from src.simulation.simulation import Simulation
    from src.input_handlers.xbox_input_handler import XboxInputHandler
    from src.simulation.observation_manager import ObservationManager # 【v4.4.7 新增】為新屬性增加類型提示


# TuningParams 資料類別
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
    【修改後 v2.0】中央狀態管理者 (Central State Manager)
    
    這個類別是整個應用程式的"單一真相來源 (Single Source of Truth)"。
    它不再是一個被動的數據容器，而是一個主動的管理者，通過訂閱事件來
    安全地更新自己的狀態，並為其他模組提供查詢。
    """

    # --- 核心屬性 ---
    # config 和 lock 屬性
    config: AppConfig
    lock: threading.Lock = field(default_factory=threading.Lock)

    # 持有對事件匯流排的引用
    # 讓每個 SimulationState 實例都能方便地訪問全域事件匯流排。
    # repr=False 避免在打印 state 物件時產生過長的循環引用輸出。
    events: EventSystem = field(default_factory=lambda: event_bus, repr=False)
    
    # --- 用戶輸入與指令狀態 ---
    # 這些狀態現在主要由 on_command_update 事件回呼來更新
    # 【修改】將 command 向量擴展為4維，以容納目標俯仰角
    command: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    
    # --- 系統控制與模式狀態 ---
    # 這些屬性由 SimulationController 根據請求事件來管理
    control_mode: str = "WALKING"
    # 【v4.0 移除】移除舊的 pending 旗標
    # control_mode_pending: str | None = None
    # 【v4.0 新增】更明確的模式切換請求
    mode_change_request: str | None = None
    previous_control_mode: str = "WALKING"
    input_mode: str = "KEYBOARD"
    
    # --- 運行時高頻數據 (由 TICK 事件更新) ---
    # 這些屬性是UI顯示的數據源，現在由 on_tick_update 回呼統一更新
    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12))
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # --- 物理狀態 (由主驅動者直接更新) ---
    # 這些是物理世界的最直接反映，由擁有 sim.data 的模組 (SimulationController) 直接更新
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    latest_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # 【v4.4.7 新增】標準化觀測數據字典 (Standardized Observations)
    # 這是 v4.4.7 方案的核心 "單一權威數據源"。
    # ObservationManager 會在每一幀計算所有可能的觀測數據，並填充此字典。
    # 所有消費者 (PolicyManager, UIController) 都應從此處讀取數據。
    std_obs: Dict[str, np.ndarray] = field(default_factory=dict)

    # [新增] v4.3.1 原始感測器數據 (由數據源更新，供 ObservationManager 使用)
    # 這是原始數據的暫存區，由 SimulationController 或 HardwareController 填充。
    raw_torso_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    # 【v4.4.2 修改】重命名以消除坐標系歧義。註解明確指出坐標系由數據源決定。
    raw_torso_linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_torso_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    raw_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_last_action: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # 【v4.4.2 新增】新增屬性以接收來自 Teensy 的俯仰角和重力向量數據
    raw_pitch_rad: float = 0.0
    raw_gravity_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 【新增】後座力計時器相關狀態
    recoil_timer: float = 0.0 # 追蹤距離下次後座力事件的時間
    recoil_interval: float = 5.0 # 下次後座力事件的隨機間隔
    recoil_warning_active: bool = False # 後座力預警旗標，直接供 ObservationManager 使用

    # --- 請求旗標 (由 UI/輸入 發起，由主驅動者處理) ---
    hard_reset_requested: bool = False
    soft_reset_requested: bool = False
    shutdown_requested: bool = False
    # 【v4.0 新增】手動懸浮切換請求
    manual_float_toggle_request: bool | None = None # 用 bool|None 來表示三種狀態：請求開啟, 請求關閉, 無請求
    
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
    # 【v4.0 修改】這個旗標現在只反映當前的真實狀態，不再被直接寫入
    manual_mode_is_floating: bool = False
    
    # --- 設備連接與狀態 ---
    serial_is_connected: bool = False
    gamepad_is_connected: bool = False
    hardware_is_running: bool = False
    hardware_ai_is_active: bool = False
    hardware_status_text: str = "Not Connected"
    
    # --- UI 相關狀態 ---
    display_page: int = 0
    num_display_pages: int = 2
    tuning_param_index: int = 0
    
    # --- 模組參考 (用於初始化和便捷訪問) ---
    # 這些參考在應用程式啟動時被賦值
    sim: 'Simulation' = None
    floating_controller_ref: 'FloatingController' = None
    terrain_manager_ref: 'TerrainManager' = None
    policy_manager_ref: 'PolicyManager' = None
    # 【v4.4.7 新增】明確加入 observation_manager_ref 的類型提示
    observation_manager_ref: 'ObservationManager' = None
    hardware_controller_ref: 'HardwareController' = None
    serial_communicator_ref: 'SerialCommunicator' = None
    xbox_handler_ref: 'XboxInputHandler' = None
    available_policies: list = field(default_factory=list)


    def __post_init__(self):
        """
        【修改後】初始化函式。
        除了設置初始值，它的核心職責是將自己註冊為事件的訂閱者，
        從而成為一個主動的狀態管理者。
        """
        # 所有現有的初始化邏輯
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__)
        self.latest_action_raw = np.zeros(self.config.num_motors)
        self.latest_final_ctrl = np.zeros(self.config.num_motors)
        self.manual_final_ctrl = np.zeros(self.config.num_motors)
        self.latest_joint_positions = np.zeros(self.config.num_motors)
        # [新增] v4.3.1 初始化新增的 raw 屬性
        self.raw_joint_positions = np.zeros(self.config.num_motors)
        self.raw_joint_velocities = np.zeros(self.config.num_motors)
        self.raw_last_action = np.zeros(self.config.num_motors)

        # 讓 SimulationState 自己訂閱核心數據更新事件
        # 這一步是架構轉變的關鍵：State不再被動地等待被寫入，
        # 而是主動去監聽系統中發生的數據更新事件。
        self.events.subscribe(EVENT_SIMULATION_TICK, self.on_tick_update)
        self.events.subscribe(EVENT_HARDWARE_TICK, self.on_tick_update)
        self.events.subscribe(EVENT_COMMAND_UPDATED, self.on_command_update)
        self.events.subscribe(EVENT_MODE_CHANGED, self.on_mode_changed)

        log.info("✅ SimulationState 初始化完成，並已訂閱核心事件。")

    # ------------------- 事件回呼函式區 (Event Callback Section) -------------------
    # 這些函式是 SimulationState 作為"狀態管理者"的核心職責。
    # 它們定義了當特定事件發生時，應該如何安全地更新內部狀態。

    def on_tick_update(self, onnx_input: np.ndarray, action_raw: np.ndarray, final_ctrl: np.ndarray):
        """
        核心數據鏈的終點：當收到來自任何數據源(模擬或硬體)的TICK事件時，
        安全地更新所有與AI決策相關的運行時共享狀態。
        """
        with self.lock:
            self.latest_onnx_input = onnx_input
            self.latest_action_raw = action_raw
            self.latest_final_ctrl = final_ctrl
            # [新增] v4.3.1 為了確保 last_action 在模擬和硬體模式下都能被正確更新，我們在此處統一更新 raw_last_action。
            # action_raw 是AI模型未經任何縮放或修改的原始輸出，這正是 last_action 所需要的。
            self.raw_last_action = action_raw

    def on_command_update(self, command: np.ndarray):
        """當收到來自輸入處理器(搖桿/鍵盤)的指令更新事件時，更新指令狀態。"""
        with self.lock:
            self.command = command.copy()  # 使用 .copy() 確保數據獨立，避免外部修改影響內部狀態

    def on_mode_changed(self, old_mode: str, new_mode: str):
        """
        當控制模式成功切換後 (由 SimulationController 發布通知事件)，
        執行相關的狀態清理和初始化工作。
        這個函式取代了舊版本 set_control_mode 內部的大部分副作用邏輯。
        """
        with self.lock:
            # 進入新模式時，初始化該模式特定的狀態
            if new_mode == "JOINT_TEST":
                self.joint_test_offsets.fill(0.0)
            elif new_mode == "MANUAL_CTRL":
                # 當進入手動模式時，將目標角度初始化為當前的預設站姿
                initial_pose = self.sim.default_pose.copy() if hasattr(self.sim, 'default_pose') else np.zeros(self.config.num_motors)
                self.manual_final_ctrl[:] = initial_pose

            # 若從任何手動/調試模式回到AI自動模式，重置AI狀態以確保平滑過渡
            is_entering_ai_mode = new_mode in ["WALKING", "FLOATING"]
            is_leaving_manual_mode = old_mode in ["JOINT_TEST", "MANUAL_CTRL", "SERIAL_MODE"]
            if is_entering_ai_mode and is_leaving_manual_mode:
                log.info("從手動/序列埠模式返回，正在重置 AI 狀態...")
                if self.policy_manager_ref:
                    self.policy_manager_ref.reset()
                self.clear_command()

    # --------------------------------------------------------------------------------------

    def set_control_mode(self, new_mode: str):
        """
        【v4.0 Phase 1 修改】職責極度簡化，只更新模式變數。
        假定總在持有鎖的上下文中被呼叫，且不再發布事件。
        事件應由真正完成模式切換的 Controller 發布。
        """
        # with self.lock:  <-- 舊的鎖已移除
        if self.control_mode == new_mode:
            return

        # 記錄上一個模式，供返回時使用
        self.previous_control_mode = self.control_mode
        self.control_mode = new_mode
        log.info(f"內部狀態: 控制模式已設定為: {self.control_mode}")
        # 【v4.0 移除】不再從 state 發布事件。通知事件應由完成操作的 SimulationController 發布。
        # event_bus.publish(EVENT_MODE_CHANGED, old_mode=self.previous_control_mode, new_mode=new_mode)

    # 以下所有輔助函式維持不變，它們的職責依然清晰有效。
    def reset_control_state(self, sim_time: float):
        """重置控制迴圈的計時器。"""
        self.control_timer = sim_time
        log.info("✅ 控制狀態已重置。")

    def clear_command(self):
        """清除使用者輸入的運動指令。"""
        self.command.fill(0.0)
        log.info("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str, clear_cmd: bool = True):
        """切換輸入模式，可選擇是否清除現有指令。"""
        with self.lock:
            if self.input_mode != new_mode:
                self.input_mode = new_mode
                if clear_cmd:
                    self.clear_command()
                log.info(f"輸入模式已切換至: {self.input_mode}")