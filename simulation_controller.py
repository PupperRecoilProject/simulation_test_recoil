# simulation_controller.py

import threading
import time
from typing import TYPE_CHECKING, Optional
import numpy as np
from mock_simulation import MockSimulation

# 導入 Mujoco 庫，如果沒有安裝，則將 mujoco 設定為 None。
try:
    import mujoco
except ImportError:
    mujoco = None

# 從 typing 模組導入 TYPE_CHECKING 以用於類型提示，避免循環依賴。
if TYPE_CHECKING:
    from state import SimulationState
    from policy import PolicyManager
    from hardware_controller import HardwareController
    from terrain_manager import TerrainManager
    from floating_controller import FloatingController
    from xbox_input_handler import XboxInputHandler
    from serial_communicator import SerialCommunicator
    from simulation import Simulation

# 導入日誌模組。
from logger import log

# 導入事件系統模組和所有需要使用的事件類型。
from event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_TUNING_PARAM_ADJUST_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_MODE_CHANGE_REQUESTED,
    EVENT_TERRAIN_REGENERATE_REQUESTED,
    EVENT_TERRAIN_SNAPSHOT_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUST_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED,
    EVENT_UI_PAGE_CHANGE_REQUESTED,
)

class SimulationController:
    """
    [核心控制器] 模擬控制器。
    在獨立的背景執行緒中運行 MuJoCo 模擬，並作為所有來自 UI 和輸入層的請求事件的中央處理者。
    它負責協調各模組的動作，確保系統的穩定運行和狀態一致性。
    """

    def __init__(self, state: 'SimulationState') -> None:
        """
        [初始化] 初始化 SimulationController。
        Args:
            state (SimulationState): 全域模擬狀態的參考。
        """
        self.state = state
        self.sim = state.sim # MuJoCo 模擬器物件
        self.config = state.config # 應用程式配置
        
        # 參考其他核心模組，以便在事件處理中調用它們的功能。
        self.policy_manager: 'PolicyManager' = state.policy_manager_ref
        self.terrain_manager: 'TerrainManager' = state.terrain_manager_ref
        self.floating_controller: 'FloatingController' = state.floating_controller_ref
        self.xbox_handler: 'XboxInputHandler' = state.xbox_handler_ref
        self.hardware_controller: 'HardwareController' = state.hardware_controller_ref
        self.serial_comm: 'SerialCommunicator' = state.serial_communicator_ref

        self._running = threading.Event() # 控制模擬執行緒的運行狀態。
        self.thread: threading.Thread | None = None # 模擬主迴圈執行緒。

        # 追蹤手動模式下懸浮是否已啟用，避免重複啟用/禁用操作。
        self._manual_float_active = False 

        # 訂閱所有需要處理的請求事件，使本控制器成為事件處理的中心。
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """
        [初始化] 集中訂閱所有本控制器需要響應的事件。
        這使得 `__init__` 方法保持簡潔，並明確列出所有事件處理者。
        """
        event_bus.subscribe(EVENT_SHUTDOWN_REQUESTED, self.stop)
        event_bus.subscribe(EVENT_MODE_CHANGE_REQUESTED, self.on_mode_change_requested)
        event_bus.subscribe(EVENT_SIMULATION_RESET_REQUESTED, self.on_simulation_reset_requested)
        event_bus.subscribe(EVENT_HARDWARE_AI_TOGGLE_REQUESTED, self.on_hardware_ai_toggle_requested)
        event_bus.subscribe(EVENT_INPUT_MODE_CHANGE_REQUESTED, self.on_input_mode_change_requested)
        event_bus.subscribe(EVENT_DEVICE_CONNECT_REQUESTED, self.on_device_connect_requested)
        event_bus.subscribe(EVENT_POLICY_CHANGE_REQUESTED, self.on_policy_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_MODE_CHANGE_REQUESTED, self.on_terrain_mode_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_REGENERATE_REQUESTED, self.on_terrain_regenerate_requested)
        event_bus.subscribe(EVENT_TERRAIN_SNAPSHOT_REQUESTED, self.on_terrain_snapshot_requested)
        event_bus.subscribe(EVENT_TUNING_PARAM_ADJUST_REQUESTED, self.on_tuning_param_adjust_requested)
        event_bus.subscribe(EVENT_JOINT_SELECT_REQUESTED, self.on_joint_select_requested)
        event_bus.subscribe(EVENT_JOINT_VALUE_ADJUST_REQUESTED, self.on_joint_value_adjust_requested)
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED, self.on_manual_float_toggle_requested)
        event_bus.subscribe(EVENT_UI_PAGE_CHANGE_REQUESTED, self.on_ui_page_change_requested)
        log.info("✅ SimulationController 已訂閱所有控制請求事件。")

    def _initialize_simulation_state(self) -> None:
        """
        [初始化] 在模擬啟動時初始化 MuJoCo 模擬器的狀態。
        僅在非無頭模式下執行。
        """
        # 如果是 MockSimulation (無頭模式)，則跳過實際的物理狀態初始化。
        if isinstance(self.sim, MockSimulation):
            log.info("[MOCK] 跳過模擬狀態初始化。")
            return

        # 如果地形管理器可用，則進行初始重置。
        if self.terrain_manager.is_functional:
            self.terrain_manager.reset() # 重置地形，確保中心點與高度場為最新狀態
        self.hard_reset() # 執行一次硬重置，將機器人放置在初始位置
        print("\n--- 模擬已啟動 (空格鍵: 暫停, N: 單步) ---")

    def start(self) -> None:
        """
        [核心迴圈控制] 啟動模擬執行緒。
        如果執行緒已經在運行，則不重複啟動。
        """
        if self.thread and self.thread.is_alive():
            return
        self._running.set() # 設定運行旗標，允許執行緒開始運行
        self.thread = threading.Thread(target=self.run, daemon=True) # 創建背景執行緒
        self.thread.start() # 啟動執行緒

    def run(self) -> None:
        """
        [核心迴圈] 模擬執行緒的主要入口點。
        此迴圈負責處理所有的請求、推進模擬、更新狀態和渲染場景。
        事件系統會直接在後台觸發對應的回呼函式，並在回呼函式中安全地修改 `state`。
        """
        # 判斷是否為無頭模式 (MockSimulation)
        is_headless = isinstance(self.sim, MockSimulation)
        
        # 在非無頭模式下，初始化 GLFW 視窗和渲染上下文。
        if not is_headless:
            self.sim.initialize_window_and_context()
            self._initialize_simulation_state()
        else:
            log.info("[MOCK] 無頭模式已啟動，跳過視窗/上下文初始化。")

        # 主模擬迴圈
        while self._running.is_set(): # 只要運行旗標被設定，就持續循環
            # 1. 檢查關閉條件：由 `EVENT_SHUTDOWN_REQUESTED` 觸發，或視窗手動關閉。
            should_close = False
            # 如果是真實模擬，則檢查 GLFW 視窗是否被要求關閉。
            if not is_headless:
                should_close = self.sim.should_close()

            # 如果收到關閉請求或視窗關閉，則停止執行緒並關閉 NiceGUI。
            # `self.stop()` 方法會被 `EVENT_SHUTDOWN_REQUESTED` 事件觸發。
            if should_close:
                # 如果是視窗關閉觸發的，也發布一個全域關閉事件，確保所有清理工作被執行。
                event_bus.publish(EVENT_SHUTDOWN_REQUESTED)
                from nicegui import app
                app.shutdown() # 請求 NiceGUI 應用程式安全關閉
                continue # 繼續下一輪迴圈，讓執行緒有機會停止

            # 2. 處理暫停/單步邏輯：這些是模擬器自身的控制狀態，直接從 `state` 讀取。
            with self.state.lock: # 讀取共享狀態需要加鎖
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
                mode = self.state.control_mode # 獲取當前控制模式

            # 如果處於單步模式且未請求執行一步，則只渲染並休眠。
            if single_step and not execute_one:
                self.sim.render_from_thread(self.state) # 渲染當前畫面
                time.sleep(0.01) # 短暫休眠以釋放 CPU 資源
                continue # 繼續下一輪迴圈

            # 如果請求執行一步，則在執行完畢後重置旗標。
            if execute_one:
                with self.state.lock: # 修改共享狀態需要加鎖
                    self.state.execute_one_step = False

            # 3. 執行模擬步進：
            # 只有在非無頭模式且當前模式不是硬體或序列埠模式時才執行物理模擬。
            if not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]:
                self._simulation_step()

            # 4. 更新衍生狀態並渲染：
            # 獲取機器人當前位置和地形模式，用於更新 TerrainManager 和渲染畫面。
            with self.state.lock: # 讀取共享狀態需要加鎖
                pos = self.state.latest_pos
                terrain_mode = self.state.terrain_mode
            
            self.update_derived_states_and_render(pos, terrain_mode) # 更新並渲染場景

        log.info("模擬執行緒已停止。")

    def stop(self) -> None:
        """
        [核心迴圈控制] 安全停止模擬執行緒。
        此方法被 `EVENT_SHUTDOWN_REQUESTED` 事件觸發。
        """
        self._running.clear() # 清除運行旗標，通知 `run` 迴圈停止
        if self.thread and self.thread.is_alive(): # 等待執行緒結束
            self.thread.join(timeout=1) # 設定超時，避免無限等待

    def _simulation_step(self) -> None:
        """
        [核心模擬邏輯] 執行一個模擬控制週期內的物理步進和 AI 決策。
        """
        with self.state.lock: # 讀取共享狀態需要加鎖
            command = self.state.command.copy() # 獲取當前運動指令
            control_mode = self.state.control_mode # 獲取當前控制模式
            tuning_params = self.state.tuning_params # 獲取調校參數

        # 獲取 AI 策略的動作。
        # onnx_input 是 AI 模型的原始輸入向量，action_final 是模型輸出的原始動作。
        onnx_input, action_final = self.policy_manager.get_action(command)

        # 根據當前控制模式決定最終的機器人關節目標角度 (final_ctrl)。
        if control_mode == "MANUAL_CTRL":
            with self.state.lock:
                final_ctrl = self.state.manual_final_ctrl.copy()
        elif control_mode == "JOINT_TEST":
            with self.state.lock:
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else: # WALKING 或 FLOATING 模式，使用 AI 輸出。
            # AI 輸出乘上動作縮放因子，再加上預設姿態，得到最終目標。
            final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale

        # 將計算出的目標角度應用到 MuJoCo 模擬器中進行位置控制。
        self.sim.apply_position_control(final_ctrl, tuning_params)

        # 將最新的運行時數據更新到共享狀態 `self.state`。
        # 這些數據會被 UI 或其他監控模組讀取。
        with self.state.lock: # 寫入共享狀態需要加鎖
            self.state.latest_onnx_input = onnx_input.flatten()
            self.state.latest_action_raw = action_final
            self.state.latest_final_ctrl = final_ctrl

        # 推進 MuJoCo 物理模擬，直到達到下一個控制週期時間點。
        target_time = self.sim.data.time + self.config.control_dt
        while self.sim.data.time < target_time:
            if not self._running.is_set(): # 檢查運行旗標，允許在迴圈中途停止
                break
            mujoco.mj_step(self.sim.model, self.sim.data) # 執行一個 MuJoCo 物理步。

    def update_derived_states_and_render(self, robot_pos: np.ndarray, terrain_mode: str) -> None:
        """
        [輔助] 更新非物理模擬直接驅動的衍生狀態，並調用渲染函式。
        Args:
            robot_pos (np.ndarray): 機器人當前在世界座標系中的位置。
            terrain_mode (str): 當前地形模式。
        """
        is_headless = isinstance(self.sim, MockSimulation)

        # 如果地形管理器可用且需要更新，則同步物理和渲染。
        if not is_headless and self.terrain_manager.is_functional and self.terrain_manager.needs_physics_and_scene_update:
            mujoco.mj_forward(self.sim.model, self.sim.data) # 重新計算物理引擎的前向動力學
            mujoco.mjr_uploadHField(self.sim.model, self.sim.context, self.terrain_manager.hfield_id) # 上傳高度場數據到 GPU
            self.terrain_manager.needs_physics_and_scene_update = False # 重置更新旗標
            log.info("✅ 地形物理與渲染已同步更新。")

        # 在非無頭模式下，更新機器人最新的位置、姿態和關節角度到共享狀態。
        if not is_headless:
            with self.state.lock: # 寫入共享狀態需要加鎖
                self.state.latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.latest_quat = self.sim.data.body('torso').xquat.copy()
                self.state.latest_joint_positions = self.sim.data.qpos[7:].copy() # 將當前關節角度複製到共享狀態

        # 更新地形管理器的狀態（例如在無限模式下檢查是否需要滑動地形）。
        if self.terrain_manager.is_functional:
            self.terrain_manager.update(robot_pos, terrain_mode)

        # 調用模擬器的渲染函式，將場景繪製到視窗中。
        self.sim.render_from_thread(self.state)

    # ====================================================================================
    # 事件處理函式 (Event Handlers)
    # 這些函式響應來自 EventBus 的請求，並在 SimulationController 的控制下執行相應的邏輯。
    # 按照事件名稱的字母順序排列，便於查找。
    # ====================================================================================

    def on_device_connect_requested(self, device: str):
        """
        [事件處理] 處理設備連接請求。
        Args:
            device (str): 要連接的設備類型 ('serial' 或 'gamepad')。
        """
        if device == "serial":
            # 嘗試連接序列埠，並更新共享狀態中的連接狀態。
            is_connected = self.serial_comm.scan_and_connect()
            with self.state.lock: # 寫入共享狀態需要加鎖
                self.state.serial_is_connected = is_connected
        elif device == "gamepad":
            # 嘗試連接搖桿，並更新共享狀態中的連接狀態。
            is_connected = self.xbox_handler.scan_and_connect()
            with self.state.lock: # 寫入共享狀態需要加鎖
                self.state.gamepad_is_connected = is_connected
        else:
            log.warning(f"無法識別的設備連接請求: {device}")

    def on_hardware_ai_toggle_requested(self):
        """
        [事件處理] 處理硬體 AI 控制切換請求。
        根據當前 AI 狀態，啟用或禁用硬體控制器中的 AI。
        """
        if self.state.hardware_ai_is_active:
            self.hardware_controller.disable_ai()
        else:
            self.hardware_controller.enable_ai()

    def on_input_mode_change_requested(self, mode: str):
        """
        [事件處理] 處理輸入模式切換請求。
        Args:
            mode (str): 目標輸入模式 ('KEYBOARD', 'GAMEPAD', 'VJOY')。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            self.state.toggle_input_mode(mode)

    def on_joint_select_requested(self, direction: int = 0, index: int = -1):
        """
        [事件處理] 處理關節選擇請求（用於關節測試/手動控制模式）。
        Args:
            direction (int): 選擇方向 (-1: 前一個, 1: 後一個)。
            index (int): 直接指定目標關節索引 (優先於 direction)。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            mode = self.state.control_mode
            if index != -1: # 如果直接指定了索引，則使用該索引
                if mode == "JOINT_TEST": self.state.joint_test_index = index
                elif mode == "MANUAL_CTRL": self.state.manual_ctrl_index = index
            elif direction != 0: # 如果提供了方向，則相對調整索引
                if mode == "JOINT_TEST":
                    self.state.joint_test_index = (self.state.joint_test_index + direction) % 12
                elif mode == "MANUAL_CTRL":
                    self.state.manual_ctrl_index = (self.state.manual_ctrl_index + direction) % 12
            else:
                log.warning("關節選擇請求無效：未提供方向或索引。")

    def on_joint_value_adjust_requested(self, value: Optional[float] = None, direction: Optional[int] = None, step: float = 0.1, clear: bool = False):
        """
        [事件處理] 處理關節數值調整請求（用於關節測試/手動控制模式）。
        Args:
            value (float, optional): 目標關節的絕對值。
            direction (int, optional): 調整方向 (-1: 減小, 1: 增大)。
            step (float): 調整的步長。
            clear (bool): 是否將當前關節值歸零。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            mode = self.state.control_mode
            # 根據模式選擇當前操作的關節索引。
            idx = self.state.joint_test_index if mode == "JOINT_TEST" else self.state.manual_ctrl_index
            
            if mode == "JOINT_TEST":
                if clear: # 清零操作
                    self.state.joint_test_offsets[idx] = 0.0
                elif value is not None: # 設定絕對值
                    # 在 JOINT_TEST 模式下，接收到的 value 應被視為最終目標，需要轉換為偏移量。
                    if self.sim and hasattr(self.sim, 'default_pose'):
                        self.state.joint_test_offsets[idx] = value - self.sim.default_pose[idx]
                    else:
                        log.warning("模擬器預設姿態未初始化，無法計算關節測試偏移量。")
                elif direction is not None: # 進行相對調整
                    self.state.joint_test_offsets[idx] += direction * step
                else:
                    log.warning("關節測試數值調整請求無效：未提供值、方向或清除指令。")
            elif mode == "MANUAL_CTRL":
                if clear: # 清零操作
                    self.state.manual_final_ctrl[idx] = 0.0
                elif value is not None: # 設定絕對值
                    self.state.manual_final_ctrl[idx] = value
                elif direction is not None: # 進行相對調整
                    self.state.manual_final_ctrl[idx] += direction * step
                else:
                    log.warning("手動控制數值調整請求無效：未提供值、方向或清除指令。")
            else:
                log.warning(f"當前模式 '{mode}' 不支持關節數值調整。")


    def on_manual_float_toggle_requested(self, value: bool):
        """
        [事件處理] 處理手動模式下懸浮狀態切換請求。
        Args:
            value (bool): 目標懸浮狀態 (True: 啟用, False: 禁用)。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            # 更新狀態機，記錄用戶意圖。
            self.state.manual_mode_is_floating = value
            
            is_manual_mode = self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]
            # 根據當前模式和目標狀態，啟用或禁用懸浮控制器。
            if is_manual_mode and value and not self._manual_float_active:
                self.floating_controller.enable(self.state.latest_pos)
                self._manual_float_active = True
            elif (not is_manual_mode or not value) and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False

    def on_mode_change_requested(self, mode: str):
        """
        [事件處理] 處理控制模式切換請求。
        此方法負責執行模式切換的實際邏輯和副作用。
        Args:
            mode (str): 目標控制模式。
        """
        with self.state.lock: # 讀寫共享狀態需要加鎖
            old_mode = self.state.control_mode
            if old_mode == mode: # 如果模式沒有變化，則不做任何事
                return
            self.state.set_control_mode(mode) # 更新模式狀態 (這會觸發 state 內部的 on_mode_changed 回呼)

            is_headless = isinstance(self.sim, MockSimulation)

            # 處理離開舊模式時的副作用（如禁用懸浮）。
            if not is_headless:
                if old_mode == "FLOATING":
                    self.floating_controller.disable()
                # 如果從 MANUAL_CTRL 離開，且懸浮是啟用的，則禁用它。
                if old_mode == "MANUAL_CTRL" and self._manual_float_active:
                    self.floating_controller.disable()
                    self._manual_float_active = False # 重置懸浮狀態標記

            # 處理進入新模式時的初始化邏輯。
            if not is_headless:
                if mode == "FLOATING":
                    self.floating_controller.enable(self.state.latest_pos)
                # 其他模式（如 JOINT_TEST, MANUAL_CTRL）的初始化邏輯現在在 state.on_mode_changed 中處理。
            
            # 處理硬體控制執行緒的啟動或停止。
            if mode == "HARDWARE_MODE" and not self.hardware_controller.is_running:
                log.info("派生執行緒以啟動硬體控制器...")
                # 在單獨的執行緒中啟動硬體控制器，避免阻塞主模擬執行緒。
                threading.Thread(target=self.hardware_controller.start_controller_threads, daemon=True).start()
            elif old_mode == "HARDWARE_MODE" and mode != "HARDWARE_MODE":
                if self.hardware_controller.is_running:
                    log.info("派生執行緒以停止硬體控制器...")
                    threading.Thread(target=self.hardware_controller.stop_controller_threads, daemon=True).start()

    def on_policy_change_requested(self, policy_name: str):
        """
        [事件處理] 處理 AI 策略模型切換請求。
        Args:
            policy_name (str): 目標策略模型的名稱。
        """
        if self.policy_manager:
            self.policy_manager.select_target_policy(policy_name)
        else:
            log.warning("策略管理器未初始化，無法切換策略。")

    def on_simulation_reset_requested(self, type: str):
        """
        [事件處理] 處理模擬重置請求。
        Args:
            type (str): 重置類型 ('hard' 或 'soft')。
        """
        if type == "hard":
            self.hard_reset()
        elif type == "soft":
            self.soft_reset()
        else:
            log.warning(f"無法識別的重置請求類型: {type}")

    def on_terrain_mode_change_requested(self, mode_name: Optional[str] = None, direction: Optional[int] = None):
        """
        [事件處理] 處理地形模式切換請求。
        可指定模式名稱（來自 UI）或方向（來自鍵盤循環）。
        Args:
            mode_name (str, optional): 目標地形模式名稱 ('INFINITE' 或單一地形名稱)。
            direction (int, optional): 循環切換方向 (1: 下一個, -1: 前一個)。
        """
        with self.state.lock: # 讀寫共享狀態需要加鎖
            # 根據請求參數決定新的地形模式。
            if mode_name: # 來自 UI 的指定模式名稱
                new_mode_value = 'INFINITE' if mode_name == 'INFINITE' else 'SINGLE'
                if new_mode_value == 'SINGLE' and self.terrain_manager and mode_name in self.terrain_manager.single_terrain_names:
                    self.state.single_terrain_index = self.terrain_manager.single_terrain_names.index(mode_name)
                self.state.terrain_mode = new_mode_value
            elif direction: # 來自鍵盤的循環切換請求
                if self.state.terrain_mode == 'INFINITE':
                    self.state.terrain_mode = 'SINGLE'
                    self.state.single_terrain_index = 0 # 切換到單一模式時，預設選擇第一個單一地形
                else:
                    if self.terrain_manager and self.terrain_manager.single_terrain_names:
                        num_terrains = len(self.terrain_manager.single_terrain_names)
                        self.state.single_terrain_index = (self.state.single_terrain_index + direction) % num_terrains
                    else:
                        log.warning("無法循環地形模式：無可用的單一地形。")
                        self.state.terrain_mode = 'INFINITE' # 如果沒有單一地形，切回無限模式
            else:
                log.warning("地形模式切換請求無效：未提供模式名稱或方向。")
                return # 無效請求，直接返回

            # 實際應用地形模式到 TerrainManager。
            if self.terrain_manager.is_functional:
                if self.state.terrain_mode == 'INFINITE':
                    self.terrain_manager.reset() # 無限模式下重置會清理快取並重生成地形
                else:
                    # 單一模式下，根據索引設置特定地形。
                    if self.terrain_manager.single_terrain_names:
                        selected_terrain_name = self.terrain_manager.single_terrain_names[self.state.single_terrain_index]
                        self.terrain_manager.set_single_terrain(selected_terrain_name)
                    else:
                        log.error("無法設置單一地形：沒有定義單一地形名稱。")
                        self.state.terrain_mode = 'INFINITE' # 確保系統處於有效狀態
                        self.terrain_manager.reset()
            # 切換地形後，通常需要硬重置機器人到新地形上。
            self.hard_reset()

    def on_terrain_regenerate_requested(self):
        """
        [事件處理] 處理地形重新生成請求 (僅在無限地形模式下)。
        """
        # 只有在地形管理器可用且當前處於無限模式時才執行重新生成。
        if self.terrain_manager.is_functional and self.state.terrain_mode == 'INFINITE':
            # 獲取機器人當前位置，用於重新生成地形後調整機器人高度。
            with self.state.lock: # 讀取共享狀態需要加鎖
                current_robot_pos = self.state.latest_pos.copy()
            self.terrain_manager.regenerate_terrain_and_adjust_robot(current_robot_pos)
        else:
            log.warning("地形重新生成請求無效：地形功能不可用或不在無限模式。")

    def on_terrain_snapshot_requested(self):
        """
        [事件處理] 處理地形快照保存請求。
        """
        if self.terrain_manager:
            self.terrain_manager.save_hfield_to_png()
        else:
            log.warning("地形管理器未初始化，無法保存地形快照。")

    def on_tuning_param_adjust_requested(self, param_name: str, value: Optional[float] = None, direction: Optional[int] = None):
        """
        [事件處理] 處理調校參數調整請求。
        Args:
            param_name (str): 要調整的參數名稱 (如 'kp', 'kd')。
            value (float, optional): 目標參數的絕對值 (來自滑桿)。
            direction (int, optional): 調整方向 (-1: 減小, 1: 增大) (來自按鍵)。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            # 根據提供的值類型（絕對值或相對方向）調整參數。
            if value is not None: # 如果提供了絕對值 (來自 UI 滑桿)
                setattr(self.state.tuning_params, param_name, value)
            elif direction is not None: # 如果提供了調整方向 (來自鍵盤/搖桿 D-Pad)
                step = self.config.param_adjust_steps.get(param_name, 0.1) # 獲取參數的步進值
                current_value = getattr(self.state.tuning_params, param_name) # 獲取當前參數值
                setattr(self.state.tuning_params, param_name, current_value + step * direction) # 應用調整
            else:
                log.warning(f"調校參數 '{param_name}' 調整請求無效：未提供值或方向。")
                return # 無效請求，直接返回

            # 確保 Kp, Kd, ActionScale 不為負值。
            self.state.tuning_params.kp = max(0.0, self.state.tuning_params.kp)
            self.state.tuning_params.kd = max(0.0, self.state.tuning_params.kd)
            self.state.tuning_params.action_scale = max(0.0, self.state.tuning_params.action_scale)

    def on_ui_page_change_requested(self, direction: int):
        """
        [事件處理] 處理 UI 顯示頁面切換請求。
        Args:
            direction (int): 切換方向 (1: 下一頁, -1: 上一頁)。
        """
        with self.state.lock: # 修改共享狀態需要加鎖
            # 循環切換顯示頁面。
            self.state.display_page = (self.state.display_page + direction) % self.state.num_display_pages
            log.info(f"UI 顯示頁面已切換至: {self.state.display_page + 1}/{self.state.num_display_pages}")

    # ====================================================================================
    # 輔助重置函式 (Reset Helper Functions)
    # 這些函式處理重置邏輯，由 on_simulation_reset_requested 事件回呼調用。
    # ====================================================================================

    def hard_reset(self) -> None:
        """
        [輔助] 執行機器人硬重置。
        將機器人重置到初始位置（根據地形高度調整），並清空速度和控制狀態。
        """
        with self.state.lock: # 讀寫共享狀態需要加鎖
            # 在硬體模式下不執行模擬器重置。
            if self.state.control_mode == "HARDWARE_MODE":
                log.warning("在硬體模式下無法執行硬重置模擬器。")
                return

            # 根據當前地形名稱決定初始高度偏移。
            terrain_name = self.terrain_manager.get_current_terrain_name_simple(self.state) if self.terrain_manager else "Unknown"
            difficult_terrains = ["Pyramid", "Stepped Pyramid"]
            start_z_offset = 1.5 if terrain_name in difficult_terrains else 0.3 # 困難地形需要更高初始高度。

            log.info(f"\n--- 正在執行機器人硬重置 (地形: {terrain_name}, 高度偏移: {start_z_offset}m) ---")

            # 重置 MuJoCo 數據。
            mujoco.mj_resetData(self.sim.model, self.sim.data)
            
            # 設定機器人基礎位置和姿態。
            self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0 # XY 位置歸零
            start_ground_z = self.terrain_manager.get_height_at(0, 0) if self.terrain_manager else 0.0
            self.sim.data.qpos[2] = start_ground_z + start_z_offset # Z 位置根據地形和偏移量設定
            log.info(f"機器人重置至原點：地形高度({start_ground_z:.2f}m) + 偏移({start_z_offset:.2f}m) = 世界Z({self.sim.data.qpos[2]:.2f}m)")
            self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0]) # 姿態 (單位四元數，表示無旋轉)
            self.sim.data.qpos[7:] = self.sim.default_pose # 關節角度設定為預設站姿

            # 清空速度和致動器控制。
            self.sim.data.qvel[:] = 0 # 線速度和角速度歸零
            self.sim.data.ctrl[:] = self.sim.default_pose # 致動器控制設置為預設姿態，防止機器人下墜

            # 執行少量物理步以穩定機器人（例如讓其腳部觸地）。
            for _ in range(10):
                mujoco.mj_step(self.sim.model, self.sim.data)

            # 重置 AI 策略的內部狀態（如觀察歷史）。
            if self.policy_manager:
                self.policy_manager.reset()
            
            # 如果重置前是懸浮模式，則重置後切換回走路模式。
            if self.state.control_mode == "FLOATING":
                self.state.set_control_mode("WALKING")
            
            # 重置其他相關的控制狀態。
            self.state.reset_control_state(self.sim.data.time) # 重置計時器
            self.state.clear_command() # 清空運動指令
            self.state.joint_test_offsets.fill(0.0) # 清空關節測試偏移量
            self.state.manual_final_ctrl.fill(0.0) # 清空手動控制目標
            self.state.manual_mode_is_floating = False # 禁用手動模式下的懸浮

            # 如果懸浮控制器處於啟用狀態，則禁用它。
            if self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False
            
            # 重新計算正向動力學，更新所有位置、速度、接觸等信息。
            mujoco.mj_forward(self.sim.model, self.sim.data)
            log.info("✅ 硬重置完成。")

    def soft_reset(self) -> None:
        """
        [輔助] 執行機器人空中姿態軟重置。
        僅重置機器人主幹姿態和關節角度，不清空世界位置。
        """
        with self.state.lock: # 讀寫共享狀態需要加鎖
            # 在硬體模式下不執行模擬器重置。
            if self.state.control_mode == "HARDWARE_MODE":
                log.warning("在硬體模式下無法執行軟重置模擬器。")
                return

            log.info("\n--- 正在執行空中姿態重置 ---")

            # 將機器人主幹姿態重置為單位四元數（無旋轉）。
            self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
            # 關節角度重置為預設站姿。
            self.sim.data.qpos[7:] = self.sim.default_pose
            # 清空所有速度。
            self.sim.data.qvel[:] = 0

            # 重置 AI 策略的內部狀態。
            if self.policy_manager:
                self.policy_manager.reset()
            
            # 清空運動指令和測試/手動控制相關狀態。
            self.state.clear_command()
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False

            # 如果懸浮控制器處於啟用狀態，則禁用它。
            if self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False

            # 重新計算正向動力學。
            mujoco.mj_forward(self.sim.model, self.sim.data)
            log.info("✅ 軟重置完成。")