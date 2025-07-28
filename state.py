# state.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig
from logger import log
from typing import TYPE_CHECKING
import threading

# 為了型別提示，避免循環匯入
if TYPE_CHECKING:
    from floating_controller import FloatingController
    from policy import PolicyManager
    from hardware_controller import HardwareController
    from terrain_manager import TerrainManager
    from serial_communicator import SerialCommunicator
    from simulation import Simulation
    from xbox_input_handler import XboxInputHandler

@dataclass
class TuningParams:
    """用於即時調整機器人控制參數的類別。"""
    kp: float # P gain (Proportional gain)
    kd: float # D gain (Derivative gain)
    action_scale: float # 動作縮放比例
    bias: float # 力矩偏置

@dataclass
class SimulationState:
    """Central state shared across threads."""
    config: AppConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tuning_params: TuningParams = field(init=False)
    
    hard_reset_requested: bool = False # 硬重置請求旗標
    soft_reset_requested: bool = False # 軟重置請求旗標

    control_timer: float = 0.0 # 控制迴圈的計時器
    
    sim_mode_text: str = "Initializing" # 舊的模式文字，可能可以移除
    input_mode: str = "KEYBOARD" # 當前的輸入模式 ("KEYBOARD" 或 "GAMEPAD")
    control_mode: str = "WALKING" # 當前的總控制模式 (例如 "WALKING", "HARDWARE_MODE")
    # 【新增】UI 執行緒若想切換模式，先將欲切換的模式寫入此處，模擬執行緒會在下一迴圈處理
    control_mode_pending: str | None = None
    previous_control_mode: str = "WALKING" # 【新功能】儲存進入 SERIAL_MODE 前的模式，以便能正確返回

    terrain_mode: str = "INFINITE" # 當前的地形模式 ("INFINITE" 或 "SINGLE")
    single_terrain_index: int = 0 # 在 SINGLE 地形模式下，當前選擇的地形索引

    latest_onnx_input: np.ndarray = field(default_factory=lambda: np.array([])) # 最新一幀的 ONNX 模型輸入向量
    latest_action_raw: np.ndarray = field(default_factory=lambda: np.zeros(12)) # 最新一幀的 ONNX 模型原始輸出
    latest_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12)) # 最終計算後要傳給致動器的控制指令
    latest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3)) # 機器人軀幹的最新位置
    latest_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.])) # 機器人軀幹的最新姿態（四元數）
    display_page: int = 0 # 除錯資訊顯示的當前頁碼
    num_display_pages: int = 2 # 除錯資訊的總頁數

    # --- 【序列埠控制台模式相關狀態】 ---

    joint_test_index: int = 0 # 在關節測試模式下，當前選中的關節索引
    joint_test_offsets: np.ndarray = field(default_factory=lambda: np.zeros(12)) # 儲存各關節在測試模式下的偏移量

    manual_ctrl_index: int = 0 # 在手動控制模式下，當前選中的關節索引
    manual_final_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12)) # 儲存手動控制模式下的最終控制角度
    manual_mode_is_floating: bool = False # 標記手動控制模式下是否啟用懸浮

    serial_is_connected: bool = False # 標記序列埠是否已連接
    gamepad_is_connected: bool = False # 標記遊戲搖桿是否已連接

    tuning_param_index: int = 0 # 當前選中要調整的參數索引 (Kp, Kd, etc.)

    # --- 【核心修改】將所有主要物件的參考儲存在此，使其成為全域上下文 ---
    sim: 'Simulation' = None # 模擬環境物件的參考
    floating_controller_ref: 'FloatingController' = None # 懸浮控制器物件的參考
    terrain_manager_ref: 'TerrainManager' = None # 地形管理器物件的參考
    policy_manager_ref: 'PolicyManager' = None # 策略管理器物件的參考
    hardware_controller_ref: 'HardwareController' = None # 硬體控制器物件的參考
    serial_communicator_ref: 'SerialCommunicator' = None # 序列埠通訊器物件的參考
    xbox_handler_ref: 'XboxInputHandler' = None # Xbox 搖桿處理器的參考
    
    available_policies: list = field(default_factory=list) # 所有可用的 ONNX 策略名稱列表
    
    hardware_is_connected: bool = False # 標記硬體控制器是否已成功啟動
    hardware_ai_is_active: bool = False # 標記硬體模式下的 AI 是否已啟用
    hardware_status_text: str = "Not Connected" # 用於在 UI 上顯示的硬體狀態文字

    single_step_mode: bool = False # 標記是否處於單步模擬模式
    execute_one_step: bool = False # 在單步模式下，請求執行下一步的旗標

    def __post_init__(self):
        """在初始化後，根據設定檔設定初始值。"""
        self.tuning_params = TuningParams(**self.config.initial_tuning_params.__dict__) # 從設定檔初始化調校參數
        self.latest_action_raw = np.zeros(self.config.num_motors) # 初始化原始動作向量
        self.latest_final_ctrl = np.zeros(self.config.num_motors) # 初始化最終控制向量
        self.manual_final_ctrl = np.zeros(self.config.num_motors) # 初始化手動控制向量
        log.info("✅ SimulationState 初始化完成 (含執行緒鎖)。")

    def reset_control_state(self, sim_time: float):
        """重置控制迴圈的計時器。"""
        self.control_timer = sim_time # 將計時器設定為當前的模擬時間
        log.info("✅ 控制狀態已重置。")

    def clear_command(self):
        """清除使用者輸入的運動指令。"""
        self.command.fill(0.0)  # 將指令向量全部設為 0
        log.info("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str, clear_cmd: bool = True):
        """切換輸入模式，可選擇是否清除現有指令。"""
        with self.lock:
            if self.input_mode != new_mode:
                self.input_mode = new_mode
                if clear_cmd:
                    self.clear_command()
                log.info(f"輸入模式已切換至: {self.input_mode}")
            
    def set_control_mode(self, new_mode: str):
        """【智慧模式切換】切換主控制模式，並能記住進入 SERIAL_MODE 前的狀態。"""
        with self.lock:  # 【執行緒安全】確保模式切換時沒有競爭條件
            if self.control_mode == new_mode:
                return  # 如果模式未改變，則不執行任何操作

            old_mode = self.control_mode  # 儲存舊模式以進行清理

            # 【新邏輯】如果準備進入 SERIAL_MODE，先記下當前的模式，以便之後可以返回
            if new_mode == "SERIAL_MODE":
                self.previous_control_mode = old_mode  # 記錄切換前的模式

            # --- 處理離開舊模式時的清理工作 ---
            if old_mode == "FLOATING":
                if self.floating_controller_ref:
                    self.floating_controller_ref.disable()  # 禁用懸浮約束
            elif old_mode == "MANUAL_CTRL" and self.manual_mode_is_floating:
                if self.floating_controller_ref:
                    self.floating_controller_ref.disable()  # 禁用懸浮約束
                self.manual_mode_is_floating = False  # 重置手動懸浮旗標
            elif old_mode == "HARDWARE_MODE":
                if new_mode not in ["SERIAL_MODE", "JOINT_TEST"]:
                    if self.hardware_controller_ref:
                        threading.Thread(target=self.hardware_controller_ref.stop_controller_threads).start()
                    self.hardware_is_connected = False
                    self.hardware_ai_is_active = False

            self.control_mode = new_mode  # 正式更新到新模式
            log.info(f"控制模式已切換至: {self.control_mode}")

            # --- 處理進入新模式時的初始化工作 ---
            if new_mode == "FLOATING":
                if self.floating_controller_ref:
                    self.floating_controller_ref.enable(self.latest_pos)  # 啟用懸浮
            elif new_mode == "JOINT_TEST":
                self.joint_test_offsets.fill(0.0)
            elif new_mode == "MANUAL_CTRL":
                self.manual_final_ctrl[:] = self.latest_final_ctrl
            elif new_mode == "HARDWARE_MODE":
                if self.hardware_controller_ref and not self.hardware_controller_ref.is_running:
                    threading.Thread(target=self.hardware_controller_ref.start_controller_threads).start()

            # --- 處理從手動模式切換回 AI 模式時的重置邏輯 ---
            is_entering_ai_mode = new_mode in ["WALKING", "FLOATING"]
            is_leaving_manual_mode = old_mode in ["JOINT_TEST", "MANUAL_CTRL", "SERIAL_MODE"]

            if is_entering_ai_mode and is_leaving_manual_mode:
                log.info("從手動/序列埠模式返回，正在重置 AI 狀態以確保平滑過渡...")
                if self.policy_manager_ref:
                    self.policy_manager_ref.reset()
                self.clear_command()
