# state.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import AppConfig
from typing import TYPE_CHECKING

# 為了型別提示，避免循環匯入
if TYPE_CHECKING:
    from floating_controller import FloatingController
    from policy import PolicyManager
    from hardware_controller import HardwareController
    from terrain_manager import TerrainManager
    from serial_communicator import SerialCommunicator # 新增
    from simulation import Simulation # 新增

@dataclass
class TuningParams:
    """用於即時調整機器人控制參數的類別。"""
    kp: float # P gain (Proportional gain)
    kd: float # D gain (Derivative gain)
    action_scale: float # 動作縮放比例
    bias: float # 力矩偏置

@dataclass
class SimulationState:
    """管理所有模擬中動態變化的狀態，取代 global 變數。"""
    config: AppConfig # 儲存從 config.yaml 載入的所有設定
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32)) # 使用者輸入的命令 [vy, vx, wz]
    tuning_params: TuningParams = field(init=False) # 可即時調整的控制參數
    
    hard_reset_requested: bool = False # 硬重置請求旗標
    soft_reset_requested: bool = False # 軟重置請求旗標

    control_timer: float = 0.0 # 控制迴圈的計時器
    
    sim_mode_text: str = "Initializing" # 舊的模式文字，可能可以移除
    input_mode: str = "KEYBOARD" # 當前的輸入模式 ("KEYBOARD" 或 "GAMEPAD")
    control_mode: str = "WALKING" # 當前的總控制模式 (例如 "WALKING", "HARDWARE_MODE")
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
    serial_command_buffer: str = "" # 用於儲存使用者在序列埠模式下正在輸入的文字
    serial_command_to_send: str = "" # 當使用者按下Enter後，指令會被移到這裡，等待main.py發送
    serial_latest_messages: list = field(default_factory=list) # 儲存從硬體收到的訊息日誌，用於顯示

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
        print("✅ SimulationState 初始化完成。")

    def reset_control_state(self, sim_time: float):
        """重置控制迴圈的計時器。"""
        self.control_timer = sim_time # 將計時器設定為當前的模擬時間
        print("✅ 控制狀態已重置。")

    def clear_command(self):
        """清除使用者輸入的運動指令。"""
        self.command.fill(0.0) # 將指令向量全部設為 0
        print("運動指令已清除。")

    def toggle_input_mode(self, new_mode: str):
        """切換輸入模式 (鍵盤/搖桿)。"""
        if self.input_mode != new_mode: # 如果新模式與當前模式不同
            self.input_mode = new_mode # 更新模式
            self.clear_command() # 清除舊的指令，避免殘留
            print(f"輸入模式已切換至: {self.input_mode}")
            
    def set_control_mode(self, new_mode: str):
        """【智慧模式切換】切換主控制模式，並能記住進入 SERIAL_MODE 前的狀態。"""
        if self.control_mode == new_mode: return # 如果模式未改變，則不執行任何操作

        old_mode = self.control_mode # 儲存舊模式以進行清理
        
        # 【新邏輯】如果準備進入 SERIAL_MODE，先記下當前的模式，以便之後可以返回
        if new_mode == "SERIAL_MODE":
            self.previous_control_mode = old_mode # 記錄切換前的模式
        
        # --- 處理離開舊模式時的清理工作 ---
        if old_mode == "FLOATING":
            if self.floating_controller_ref: self.floating_controller_ref.disable() # 禁用懸浮約束
        elif old_mode == "MANUAL_CTRL" and self.manual_mode_is_floating:
             if self.floating_controller_ref: self.floating_controller_ref.disable() # 禁用懸浮約束
             self.manual_mode_is_floating = False # 重置手動懸浮旗標
        elif old_mode == "HARDWARE_MODE":
            # 只有當我們要切換到一個非硬體也非序列埠的模式時，才停止硬體控制器
            if new_mode not in ["SERIAL_MODE", "JOINT_TEST"]:
                if self.hardware_controller_ref: self.hardware_controller_ref.stop() # 停止硬體控制器執行緒
                self.hardware_is_connected = False # 更新連接狀態
                self.hardware_ai_is_active = False # 更新 AI 狀態
            
        self.control_mode = new_mode # 正式更新到新模式
        print(f"控制模式已切換至: {self.control_mode}")

        # --- 處理進入新模式時的初始化工作 ---
        if new_mode == "FLOATING":
            if self.floating_controller_ref: self.floating_controller_ref.enable(self.latest_pos) # 啟用懸浮
        elif new_mode == "JOINT_TEST":
            self.joint_test_offsets.fill(0.0) # 清空關節偏移量
        elif new_mode == "MANUAL_CTRL":
            self.manual_final_ctrl[:] = self.latest_final_ctrl # 將當前控制角度作為手動控制的初始值
        elif new_mode == "HARDWARE_MODE":
             if self.hardware_controller_ref and not self.hardware_controller_ref.is_running: # 如果硬體控制器存在且未運行
                 if self.hardware_controller_ref.connect_and_start(): # 嘗試啟動
                     self.hardware_is_connected = True
                 else:
                     print("❌ 硬體連接失敗，自動返回 WALKING 模式。")
                     self.control_mode = "WALKING" # 如果啟動失敗，自動退回安全模式
                     print(f"控制模式已自動切換至: {self.control_mode}")

        # --- 處理從手動模式切換回 AI 模式時的重置邏輯 ---
        is_entering_ai_mode = new_mode in ["WALKING", "FLOATING"] # 判斷是否進入 AI 模式
        is_leaving_manual_mode = old_mode in ["JOINT_TEST", "MANUAL_CTRL", "SERIAL_MODE"] # 判斷是否離開手動/除錯模式
        
        if is_entering_ai_mode and is_leaving_manual_mode:
            print("從手動/序列埠模式返回，正在重置 AI 狀態以確保平滑過渡...")
            if self.policy_manager_ref:
                self.policy_manager_ref.reset() # 重置 AI 策略的歷史狀態
            self.clear_command() # 清除使用者指令