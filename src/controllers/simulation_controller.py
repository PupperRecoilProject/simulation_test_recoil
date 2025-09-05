# src/controllers/simulation_controller.py
"""
【v4.1.0 修改版】模擬與狀態協調控制器

在獨立執行緒中運行模擬並處理所有狀態變更。
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from src.core.logger import log

import numpy as np
import random
from src.mock.mock_simulation import MockSimulation

from src.core.event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_TUNING_PARAM_ADJUSTED,
    EVENT_TUNING_PARAM_SELECT_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_SERIAL_COMMAND_SEND,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_SHUTDOWN_REQUESTED,
    # 【v4.0 新增】導入新的通知事件
    EVENT_MODE_CHANGED,
    EVENT_STATE_UPDATED,

    # ... 根據需要導入其他事件
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    from src.core.state import SimulationState


class SimulationController:
    """在獨立執行緒中運行模擬並處理所有狀態變更。"""

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.xbox_handler = state.xbox_handler_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref # 直接獲取 serial_communicator 的參考

        self._running = threading.Event()
        self.thread: threading.Thread | None = None

        self._manual_float_active = False        # 追蹤手動模式下懸浮是否已啟用
        self._subscribe_to_events()        # 訂閱所有來自輸入層的請求事件

        # 初始化將在執行緒啟動後進行

    # ============================ 事件訂閱輔助函式 ============================
    def _subscribe_to_events(self):
        """將所有事件訂閱邏輯集中到此處。"""
        event_bus.subscribe(EVENT_MODE_CHANGE_REQUESTED, self.on_mode_change_requested)
        event_bus.subscribe(EVENT_SIMULATION_RESET_REQUESTED, self.on_simulation_reset_requested)
        event_bus.subscribe(EVENT_TUNING_PARAM_ADJUSTED, self.on_tuning_param_adjusted)
        event_bus.subscribe(EVENT_TUNING_PARAM_SELECT_REQUESTED, self.on_tuning_param_select_requested)
        event_bus.subscribe(EVENT_INPUT_MODE_CHANGE_REQUESTED, self.on_input_mode_change_requested)
        event_bus.subscribe(EVENT_DEVICE_CONNECT_REQUESTED, self.on_device_connect_requested)
        event_bus.subscribe(EVENT_SERIAL_COMMAND_SEND, self.on_serial_command_send_requested)
        event_bus.subscribe(EVENT_POLICY_CHANGE_REQUESTED, self.on_policy_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_CHANGE_REQUESTED, self.on_terrain_change_requested)
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLED, self.on_manual_float_toggled)
        event_bus.subscribe(EVENT_JOINT_SELECT_REQUESTED, self.on_joint_select_requested)
        event_bus.subscribe(EVENT_JOINT_VALUE_ADJUSTED, self.on_joint_value_adjusted)
        event_bus.subscribe(EVENT_SHUTDOWN_REQUESTED, self.on_shutdown_requested)

        # 【v4.0 修改】確保訂閱了手動懸浮事件
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLED, self.on_manual_float_toggled)


        # 為了向前兼容，保留對舊有 pending_mode 的處理，但鼓勵新程式碼使用事件
        log.info("SimulationController 已訂閱所有核心請求事件。")

    # ------------------------------------------------------------------
    def _initialize_simulation_state(self) -> None:
        if isinstance(self.sim, MockSimulation):
            log.info("[MOCK] Skip simulation state initialization.")
            return

        if self.terrain_manager.is_functional:
            # 初始啟動時重置地形管理器，以確保中心點與高度場為最新狀態
            self.terrain_manager.reset()
        self.hard_reset()
        print("\n--- Simulation Started (SPACE: Pause, N: Step) ---")

    # ------------------------------------------------------------------
    def start(self) -> None:
        """啟動模擬執行緒。"""
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _update_recoil_state(self):
        """【新增】管理後座力預警計時器的邏輯。"""
        # 這個函式模擬訓練環境中的計時器，以產生 firearm_recoil_warming 訊號
        # 參數定義
        WARNING_DURATION_S = 0.15 # 預警持續時間 (秒), 對應訓練碼中的 warning_duration=3 (steps) * 0.05 (dt)
        MIN_INTERVAL_S = 2.5      # 最小間隔 (秒), 對應 50 steps
        MAX_INTERVAL_S = 10.0     # 最大間隔 (秒), 對應 200 steps

        with self.state.lock:
            # 將計時器減去控制時間間隔
            self.state.recoil_timer -= self.config.control_dt
            
            # 檢查是否進入預警階段
            if self.state.recoil_timer <= WARNING_DURATION_S:
                self.state.recoil_warning_active = True
            
            # 檢查計時器是否已到期
            if self.state.recoil_timer <= 0:
                # 重置預警旗標
                self.state.recoil_warning_active = False
                # 產生下一個隨機的後座力事件間隔
                self.state.recoil_interval = random.uniform(MIN_INTERVAL_S, MAX_INTERVAL_S)
                # 重設計時器
                self.state.recoil_timer = self.state.recoil_interval
                # 可以在這裡加入一個模擬的物理後座力效果（可選）
                log.info(f"*** RECOIL EVENT *** Next in {self.state.recoil_interval:.2f}s")
                
    # ============================ 主要運行 ============================
    def run(self) -> None:
        """
        【v4.0.2 修改版】執行緒主迴圈。
        【v4.6.0 修改】分離物理更新與數據處理，確保數據流在所有模式下暢通。
        """
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context()
        self._initialize_simulation_state()

        # 【v4.7.2 修正】用於跨幀傳遞上一幀動作的局部變數
        action_from_previous_frame = np.zeros(self.config.num_motors)

        while self._running.is_set():
            # 【v4.7.2 新增】在每一幀的開始，將上一幀的動作寫入 state
            # 這是確保 ObservationManager 讀取到正確時序數據的關鍵
            with self.state.lock:
                self.state.raw_last_action = action_from_previous_frame.copy()

            # ======================== [v4.0] 請求處理階段 ========================
            # 在一個極短的鎖定範圍內，原子性地讀取並清除所有掛起的請求。
            with self.state.lock:
                shutdown_req = self.state.shutdown_requested
                hard_reset_req = self.state.hard_reset_requested
                # 【v4.0.1 修正】正確讀取 soft_reset_requested
                soft_reset_req = self.state.soft_reset_requested
                mode_change_req = self.state.mode_change_request
                float_toggle_req = self.state.manual_float_toggle_request

                # 清除已讀取的請求
                self.state.shutdown_requested = False
                self.state.hard_reset_requested = False
                # 【v4.0.1 修正】正確清除 soft_reset_requested
                self.state.soft_reset_requested = False
                self.state.mode_change_request = None
                self.state.manual_float_toggle_request = None

            # 在鎖之外，安全地執行請求對應的操作
            if shutdown_req:
                self._handle_shutdown()
                continue # 結束迴圈

            if hard_reset_req: self.hard_reset()
            # 【v4.0.1 修正】補上對軟重置請求的處理
            if soft_reset_req: self.soft_reset()

            if mode_change_req:
                self._handle_mode_change(mode_change_req)
            
            if float_toggle_req is not None:
                self._handle_float_toggle(float_toggle_req)
            
            # ======================== 主邏輯與模擬步驟 ========================
            with self.state.lock:
                mode = self.state.control_mode
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
            
            if single_step and not execute_one:
                self.sim.render_from_thread(self.state)
                time.sleep(0.01) # 避免空轉
                continue
            
            if execute_one:
                with self.state.lock: self.state.execute_one_step = False
                
            # 【新增】更新後座力計時器狀態
            self._update_recoil_state()

            # 【v4.6.0 修改】將 is_simulation_active 的判斷移至此處
            is_headless = isinstance(self.sim, MockSimulation)
            is_simulation_active = not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]
            
            # --- 步驟 1: 物理更新 (僅在模擬模式下執行) ---
            if is_simulation_active:
                self._simulation_step()
            elif mode == "HARDWARE_MODE": # <--- 【核心修改】新增這個 elif 區塊
                self._update_simulation_from_hardware_state()
            else:
                # 在非模擬模式下，給主迴圈一個短暫休眠以控制頻率
                time.sleep(1.0 / self.config.control_freq)

            # --- 步驟 2: 數據處理 (在所有模式下執行) ---
            # 【v4.6.0 新增】將此調用移至此處，確保無論模式如何，數據處理鏈路都會被觸發。
            # 這是解決硬體模式下 UI 數據靜止問題的關鍵。
            if self.state.observation_manager_ref:
                self.state.observation_manager_ref.update_all_observations()

            # 【v4.7.2 修正】在每一幀的末尾，暫存當前幀的動作以供下一幀使用
            with self.state.lock:
                # latest_action_raw 是由 _simulation_step 或 _perform_ai_step 更新的
                action_from_previous_frame = self.state.latest_action_raw.copy()

            # --- 步驟 3: 渲染 (在所有可視模式下執行) ---
            if not is_headless:
                self.update_derived_states_and_render()

        log.info("模擬執行緒已優雅地停止。")

    def _handle_shutdown(self):
        """【v4.0 新增】處理關閉請求的邏輯。"""
        log.info("偵測到關閉請求，正在停止主迴圈...")
        self._running.clear()
        # 確保 NiceGUI 也被通知關閉
        from nicegui import app
        app.shutdown()

    def _handle_mode_change(self, new_mode: str):
        """
        【v4.0 新增】安全地處理模式切換。
        此函式在主迴圈的安全上下文中被呼叫，可以安全地修改物理狀態。
        """
        with self.state.lock:
            old_mode = self.state.control_mode
            if old_mode == new_mode: return

            # 步驟 1: 處理非物理相關的邏輯
            if new_mode == "HARDWARE_MODE":
                # 【v4.0.2 修改】呼叫非阻塞的請求函式
                log.info(f"模式切換: 發出硬體啟動請求...")
                self.hardware_controller.request_start()
            elif old_mode == "HARDWARE_MODE":
                # 【v4.0.2 修改】呼叫非阻塞的請求函式
                log.info(f"模式切換: 發出硬體停止請求...")
                self.hardware_controller.request_stop()
            # 【v4.0.2 移除】不再由 SimCtrl 直接修改 HW 狀態，改由 HWCtrl 自己負責
            # self.state.hardware_is_running = success 
            
            # 步驟 2: 原子性地更新核心模式狀態
            self.state.set_control_mode(new_mode)
        
        # 步驟 3: 在鎖之外，安全地執行物理相關的修改
        self._handle_mode_change_physics(old_mode, new_mode)

        # 【v4.0.2 新增】在完成模式切換後，如果進入了非模擬模式，
        # 我們主動渲染一次“凍結幀”，以確保UI上顯示的是正確的遮罩和文字。
        if new_mode in ["HARDWARE_MODE", "SERIAL_MODE"]:
            if not isinstance(self.sim, MockSimulation):
                log.info(f"渲染 '{new_mode}' 的凍結畫面...")
                self.sim.render_from_thread(self.state)

        # 步驟 4: 發布模式已變更的通知事件
        event_bus.publish(EVENT_MODE_CHANGED, old_mode=old_mode, new_mode=new_mode)
        log.info(f"✅ 模式已成功從 '{old_mode}' 切換至 '{new_mode}'。")

    def _handle_mode_change_physics(self, old_mode: str, new_mode: str):
        """【v4.0 新增】專門處理模式切換中涉及物理修改的部分。"""
        # 離開舊模式時的物理清理
        if old_mode == "FLOATING":
            self.floating_controller.disable()
            self._manual_float_active = False # 確保同步
        
        # 進入新模式時的物理初始化
        if new_mode == "FLOATING":
            current_pos = self.sim.data.body('torso').xpos.copy()
            self.floating_controller.enable(current_pos)
            self._manual_float_active = True # 確保同步
        
        # 進入手動/測試模式時，重置關節姿態
        if new_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
            log.info(f"進入 {new_mode}，重置關節姿態與速度。")
            self.sim.data.qpos[7:] = self.sim.default_pose.copy()
            self.sim.data.qvel[6:] = 0
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def _handle_float_toggle(self, is_floating: bool):
        """【v4.0 新增】安全地處理手動懸浮請求。"""
        is_manual_mode = self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]
        if not is_manual_mode:
            log.warning("只有在 JOINT_TEST 或 MANUAL_CTRL 模式下才能切換懸浮。")
            return

        if is_floating and not self._manual_float_active:
            current_pos = self.sim.data.body('torso').xpos.copy()
            self.floating_controller.enable(current_pos)
            self._manual_float_active = True
        elif not is_floating and self._manual_float_active:
            self.floating_controller.disable()
            self._manual_float_active = False
        
        # 更新 state 中的真實狀態
        with self.state.lock:
            self.state.manual_mode_is_floating = self._manual_float_active
        log.info(f"手動懸浮物理狀態已切換為: {self._manual_float_active}")
        
    def _update_simulation_from_hardware_state(self):
        """
        【新增】在硬體模式下，將從實體機器人收到的狀態同步到 MuJoCo 模擬器以進行視覺化。
        """
        if isinstance(self.sim, MockSimulation) or not self.state.hardware_is_running:
            return

        with self.state.lock:
            # 從 state 讀取最新的、由硬體控制器更新的原始關節角度
            hw_joint_positions = self.state.raw_joint_positions.copy()

        # 將硬體關節角度寫入 MuJoCo 的數據結構中
        # qpos[7:] 對應的是12個馬達的關節
        self.sim.data.qpos[7:] = hw_joint_positions

        # 執行 mj_forward() 來更新模型的運動學（例如身體各部分的位置）
        # 這一步是讓模型在畫面上動起來的關鍵，但它不會推進物理模擬
        mujoco.mj_forward(self.sim.model, self.sim.data)



    # ============================ 事件回呼處理函式 ============================
    def on_mode_change_requested(self, mode: str):
        """【v4.0】處理模式切換請求。只設定請求旗標。"""
        log.debug(f"接收到模式切換請求 -> {mode}，正在設定請求旗標。")
        with self.state.lock:
            self.state.mode_change_request = mode

    def on_simulation_reset_requested(self, type: str):
        """【v4.0.1 修復】處理模擬重置請求。只設定請求旗標。"""
        log.debug(f"接收到 '{type}' 重置請求，正在設定旗標。")
        with self.state.lock:
            if type == "hard":
                self.state.hard_reset_requested = True
            elif type == "soft":
                # 【v4.0.1 修正】確保 soft reset 旗標被正確設置為 True
                self.state.soft_reset_requested = True 

    def on_tuning_param_select_requested(self, direction: int):
        """處理切換當前調校參數的請求。"""
        with self.state.lock:
            num_params = len(self.state.policy_manager_ref.param_keys)
            self.state.tuning_param_index = (self.state.tuning_param_index + direction) % num_params
            log.debug(f"調校參數索引已切換至: {self.state.tuning_param_index}")
            
    def on_tuning_param_adjusted(self, param_name: str = None, value: float = None, direction: int = None):
        """
        [v3.1.1] 處理調整參數值的請求。
        此版本修正了參數來源和默認值的問題。
        """
        with self.state.lock:
            # 如果事件沒有提供 param_name (來自鍵盤/搖桿的步進調整)，
            # 我們從 state 中獲取當前選中的參數。
            if param_name is None:
                # [修正] 參數的鍵名列表應直接從 config 或 state 本身獲取，而不是 policy_manager
                # 我們可以將它定義在 SimulationState 或 Config 中，為簡潔起見，這裡暫時硬編碼
                param_keys = ['kp', 'kd', 'action_scale', 'bias']
                if 0 <= self.state.tuning_param_index < len(param_keys):
                    param_name = param_keys[self.state.tuning_param_index]
                else:
                    log.error(f"無效的調校參數索引: {self.state.tuning_param_index}")
                    return

            # 根據事件提供的參數類型執行操作
            if value is not None:  # 來自UI滑桿的絕對值設定
                setattr(self.state.tuning_params, param_name, value)
            elif direction is not None:  # 來自鍵盤/搖桿的步進調整
                step = self.config.param_adjust_steps.get(param_name, 0.1)
                current_value = getattr(self.state.tuning_params, param_name)
                new_value = current_value + step * direction
                setattr(self.state.tuning_params, param_name, new_value)
            else:
                log.warning(f"接收到無效的參數調整請求: param_name={param_name}, value={value}, direction={direction}")
                return

            # 確保參數值在合理範圍內
            self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
            self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
            self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)
            log.info(f"參數 '{param_name}' 已調整為: {getattr(self.state.tuning_params, param_name):.2f}")

    def on_input_mode_change_requested(self, mode: str):
        """處理輸入模式切換請求。"""
        # [修改] 增加一個選項，避免在切換到 VJOY 時清除指令
        clear_cmd = mode != "VJOY"
        self.state.toggle_input_mode(mode, clear_cmd=clear_cmd)

    def on_device_connect_requested(self, device: str):
        """處理設備連接請求。"""
        log.info(f"接收到連接 '{device}' 的請求...")
        if device == "serial" and self.serial_comm:
            is_connected = self.serial_comm.scan_and_connect()
            with self.state.lock:
                self.state.serial_is_connected = is_connected
        elif device == "gamepad" and self.xbox_handler:
            is_connected = self.xbox_handler.scan_and_connect()
            with self.state.lock:
                self.state.gamepad_is_connected = is_connected

    def on_serial_command_send_requested(self, command: str):
        """
        【v3.0.1】處理序列埠命令發送請求。
        將命令安全地傳遞給 SerialCommunicator。
        """
        if self.serial_comm and self.serial_comm.is_connected:
            try:
                # 這裡調用 serial_comm 的 send_command，它內部會檢查是否被硬件控制器管理
                self.serial_comm.send_command(command)
                log.info(f"成功發送序列埠命令: '{command}'")
            except Exception as e:
                log.error(f"發送序列埠命令失敗: {e}")
        else:
            log.warning("序列埠未連接，無法發送命令。")

    def on_policy_change_requested(self, policy_name: str):
        """處理AI策略切換請求。"""
        if self.policy_manager:
            log.info(f"接收到切換策略請求: {policy_name}")
            self.policy_manager.select_target_policy(policy_name)

    def on_terrain_change_requested(self, name: str):
        """
        [v3.1.2] 處理地形切換請求。
        此版本不再直接呼叫 hard_reset，而是設定請求旗標。
        """
        if not self.terrain_manager or not self.terrain_manager.is_functional:
            return
            
        log.info(f"接收到切換地形請求: {name}")
        
        # 標記是否需要重置
        needs_reset = False
        with self.state.lock:
            if name == 'INFINITE':
                if self.state.terrain_mode != 'INFINITE':
                    self.state.terrain_mode = 'INFINITE'
                    self.terrain_manager.reset()
                    needs_reset = True
            else:
                if name in self.terrain_manager.single_terrain_names:
                    new_index = self.terrain_manager.single_terrain_names.index(name)
                    if self.state.terrain_mode != 'SINGLE' or self.state.single_terrain_index != new_index:
                        self.state.terrain_mode = 'SINGLE'
                        self.state.single_terrain_index = new_index
                        self.terrain_manager.set_single_terrain(name)
                        needs_reset = True
        
        # 在鎖外設定請求旗標
        if needs_reset:
            log.info("地形已變更，請求硬重置以應用...")
            with self.state.lock:
                self.state.hard_reset_requested = True

    def on_manual_float_toggled(self, is_floating: bool):
        """【v4.0】處理手動模式下的懸浮開關請求。只設定請求旗標。"""
        log.debug(f"接收到手動懸浮切換請求 -> {is_floating}，正在設定請求旗標。")
        with self.state.lock:
            self.state.manual_float_toggle_request = is_floating

    def on_joint_select_requested(self, index: int):
        """處理關節選擇請求。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                self.state.joint_test_index = index
            elif self.state.control_mode == "MANUAL_CTRL":
                self.state.manual_ctrl_index = index
        log.debug(f"當前選中關節索引: {index}")

    def on_joint_value_adjusted(self, value: float = None, direction: float = None, clear: bool = False):
        """處理關節值調整請求。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                idx = self.state.joint_test_index
                if clear:
                    self.state.joint_test_offsets[idx] = 0.0
                elif value is not None: # 來自滑桿的絕對值
                    self.state.joint_test_offsets[idx] = value - self.state.sim.default_pose[idx]
                elif direction is not None: # 來自按鈕的步進
                    self.state.joint_test_offsets[idx] += direction
            
            elif self.state.control_mode == "MANUAL_CTRL":
                idx = self.state.manual_ctrl_index
                if clear:
                    # 在手動模式下，歸零可能意味著回到預設站姿的對應關節角度
                    self.state.manual_final_ctrl[idx] = self.sim.default_pose[idx]
                elif value is not None:
                    self.state.manual_final_ctrl[idx] = value
                elif direction is not None:
                    self.state.manual_final_ctrl[idx] += direction

    def on_shutdown_requested(self):
        """
        【新增】處理關閉應用程式的請求。
        此回呼只負責設定旗標，由主迴圈安全地執行關閉流程。
        """
        log.info("接收到全域關閉請求，正在設定旗標...")
        with self.state.lock:
            self.state.shutdown_requested = True



    # =========================================================================

    # 【v4.0 移除】移除舊的、不安全的模式處理函式
    # def process_pending_mode_change(self) -> None: ...
    # def handle_mode_change(self, old_mode: str, new_mode: str) -> None: ...

    def update_derived_states_and_render(self) -> None:
        """
        【v4.0】更新所有依賴於核心物理狀態的衍生狀態（如地形），並渲染場景。
        此方法現在自給自足，直接從 self.state 獲取所需數據。
        """
        is_headless = isinstance(self.sim, MockSimulation)

        # 步驟 1: 在函式內部，從 state 中讀取本幀需要的所有數據
        with self.state.lock:
            # 獲取地形更新所需的數據
            current_pos = self.state.latest_pos.copy()
            terrain_mode = self.state.terrain_mode
            
            # 如果不是無頭模式，則更新 state 中的物理數據以供 UI 讀取
            if not is_headless:
                self.state.latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.latest_quat = self.sim.data.body('torso').xquat.copy()
                self.state.latest_joint_positions = self.sim.data.qpos[7:].copy()

        # 步驟 2: 執行衍生狀態的更新 (此處邏輯不變)
        # 更新地形（如果需要）
        if self.terrain_manager.is_functional:
            self.terrain_manager.update(current_pos, terrain_mode)
        
        # 步驟 3: 執行渲染 (此處邏輯不變)
        self.sim.render_from_thread(self.state)

    # ------------------------------------------------------------------
    # 【v4.3.1 修改】 _simulation_step 方法
    def _simulation_step(self) -> None:
        """
        【v4.3.1 修改】執行物理模擬並更新原始物理數據到 SimulationState。
        【v4.4.2 修改】同步更新寫入的 raw_ 變數名。
        【v4.4.7 修改】在方法結尾新增對觀測管理器的統一調用。

        此函式現在除了執行模擬，還負責將原始物理數據寫入 SimulationState。
        """

        # log.info("--- _simulation_step START ---") # 增加起始 log

        # 讀取狀態和獲取 AI 動作的邏輯不變
        with self.state.lock:
            command = self.state.command.copy()
            control_mode = self.state.control_mode
            tuning_params = self.state.tuning_params

        # 【v4.4.7 修改】 PolicyManager 的 get_action 內部邏輯將改變，但此處的調用方式不變
        onnx_input, action_final = self.policy_manager.get_action(command)

        # 根據模式計算最終控制指令的邏輯不變
        if control_mode == "MANUAL_CTRL":
            with self.state.lock:
                final_ctrl = self.state.manual_final_ctrl.copy()
        elif control_mode == "JOINT_TEST":
            with self.state.lock:
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else:
            final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale

        # 應用 PD 控制的邏輯不變
        self.sim.apply_position_control(final_ctrl, tuning_params)

        # 更新 UI 顯示用的數據的邏輯不變
        with self.state.lock:
            self.state.latest_onnx_input = onnx_input.flatten()
            self.state.latest_action_raw = action_final
            self.state.latest_final_ctrl = final_ctrl

        # 執行物理模擬的迴圈不變
        target_time = self.sim.data.time + self.config.control_dt
        while self.sim.data.time < target_time:
            if not self._running.is_set():
                break
            # 這是物理引擎的核心步驟
            mujoco.mj_step(self.sim.model, self.sim.data)

        # 【v4.3.1 新增】 - 將原始物理數據寫入 State
        # 在 mj_step 之後，sim.data 中包含了最新的物理狀態，我們將其寫入 state.raw_...
        # 作為 ObservationManager 的數據源。
        with self.state.lock:
            # 讀取軀幹的姿態四元數
            self.state.raw_torso_quat = self.sim.data.body('torso').xquat.copy()
            # 讀取軀幹在世界座標系下的線速度和角速度
            self.state.raw_torso_linear_velocity = self.sim.data.cvel[self.sim.torso_id, 3:].copy()
            self.state.raw_torso_angular_velocity = self.sim.data.cvel[self.sim.torso_id, :3].copy()
            # 讀取所有關節的角度和角速度
            self.state.raw_joint_positions = self.sim.data.qpos[7:].copy()
            self.state.raw_joint_velocities = self.sim.data.qvel[6:].copy()
            # 從 XML 中定義的感測器讀取加速度計數據
            if self.sim.accelerometer_id != -1:
                 start = self.sim.model.sensor_adr[self.sim.accelerometer_id]
                 end = start + self.sim.model.sensor_dim[self.sim.accelerometer_id]
                 self.state.raw_accelerometer = self.sim.data.sensordata[start:end].copy()
            else:
                 # 如果感測器不存在，用零填充
                 self.state.raw_accelerometer.fill(0.0)
        
        # 【v4.4.7 新增】【v4.6.0 刪除】觸發標準化觀測數據的全量更新
        # 這一行將被移動到 run() 迴圈的公共區域，以確保在所有模式下都能執行。
        # self.state.observation_manager_ref.update_all_observations()

        # log.info("--- _simulation_step END ---") # 增加結束 log




    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)


    # ------------------------------------------------------------------
    def hard_reset(self) -> None:
        """根據目前地形自動決定適當高度並執行硬重置。"""
        with self.state.lock:
            # 取得當前地形名稱以判斷重置高度
            terrain_name = self.terrain_manager.get_current_terrain_name_simple(self.state)

        difficult = ["Pyramid", "Stepped Pyramid"]
        # 困難地形需要更高的初始高度以保證落地安全
        start_z_offset = 1.5 if terrain_name in difficult else 0.3

        print(f"\n--- 正在執行機器人硬重置 (地形: {terrain_name}, 高度偏移: {start_z_offset}m) ---")
        # 【核心修正】硬重置僅重置機器人狀態，不再重置地形

        with self.state.lock:
            if self.state.control_mode == "HARDWARE_MODE":
                return

            mujoco.mj_resetData(self.sim.model, self.sim.data)
            self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0
            # 依照目前地形取得原點的高度，確保重置後不會埋在地底
            start_ground_z = self.terrain_manager.get_height_at(0, 0)
            self.sim.data.qpos[2] = start_ground_z + start_z_offset
            self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
            self.sim.data.qpos[7:] = self.sim.default_pose
            self.sim.data.qvel[:] = 0
            self.sim.data.ctrl[:] = self.sim.default_pose
            for _ in range(10):
                mujoco.mj_step(self.sim.model, self.sim.data)

            self.policy_manager.reset()
            if self.state.control_mode == "FLOATING":
                self.state.set_control_mode("WALKING")
            self.state.reset_control_state(self.sim.data.time)
            self.state.clear_command()
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False
            if self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False
            
            # 【新增】重置後座力計時器狀態
            self.state.recoil_interval = random.uniform(2.5, 10.0)
            self.state.recoil_timer = self.state.recoil_interval
            self.state.recoil_warning_active = False

            self.state.hard_reset_requested = False
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def soft_reset(self) -> None:
        """【v4.0.1 確認】此函式邏輯是正確的，無需修改。"""
        print("\n--- 正在執行空中姿態重置 ---")
        with self.state.lock:
            if self.state.control_mode == "HARDWARE_MODE":
                return

            self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
            self.sim.data.qpos[7:] = self.sim.default_pose
            self.sim.data.qvel[:] = 0

            self.policy_manager.reset()
            self.state.clear_command()
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False
            if self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False
            mujoco.mj_forward(self.sim.model, self.sim.data)
            # 【v4.0.1 移除】旗標的清除工作已經在 run() 迴圈的頂部完成
            # self.state.soft_reset_requested = False



