from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from logger import log

import numpy as np
from mock_simulation import MockSimulation

from event_system import (
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
    # ... 根據需要導入其他事件
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    from state import SimulationState


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

    # ============================ 主要運行 ============================
    def run(self) -> None:
        """
        [v3.1.3] 執行緒進入點：負責處理所有請求並運行模擬。
        此版本統一了請求旗標的處理邏輯，使主迴圈更清晰。
        """
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context()
            self._initialize_simulation_state()
        else:
            log.info("[MOCK] Headless mode, skip window/context init.")

        while self._running.is_set():
            # ======================== 步驟 1: 獲取狀態快照與處理同步請求 ========================
            # 在一個鎖內，讀取所有本幀需要的旗標和狀態，然後立即釋放鎖。
            # 這是執行緒安全的標準做法。
            with self.state.lock:
                # 讀取請求旗標
                shutdown_req = self.state.shutdown_requested
                hard_reset_req = self.state.hard_reset_requested
                soft_reset_req = self.state.soft_reset_requested
                pending_mode_req = self.state.control_mode_pending

                # 讀取後立即清除旗標，避免重複執行
                if hard_reset_req: self.state.hard_reset_requested = False
                if soft_reset_req: self.state.soft_reset_requested = False
                if pending_mode_req: self.state.control_mode_pending = None
            
            # 在鎖外，安全地處理這些請求
            if hard_reset_req:
                self.hard_reset()
            if soft_reset_req:
                self.soft_reset()
            if pending_mode_req:
                self.handle_mode_change(self.state.control_mode, pending_mode_req)

            # 檢查是否應關閉視窗 (全域關閉請求或視窗被手動關閉)
            should_close = shutdown_req or (not is_headless and self.sim.should_close())
            if should_close:
                if not is_headless and not self.sim.should_close():
                    # 如果是程式邏輯觸發的關閉，而非用戶點擊關閉按鈕
                    log.info("偵測到全域關閉請求，正在關閉模擬視窗...")
                    from glfw import set_window_should_close
                    set_window_should_close(self.sim.window, 1)
                self._running.clear()
                # 確保 NiceGUI 也被通知關閉
                from nicegui import app
                app.shutdown()
                continue
            
            # ================================== 步驟 2: 主邏輯 ==================================
            with self.state.lock:
                mode = self.state.control_mode
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
                manual_float = self.state.manual_mode_is_floating

            if single_step and not execute_one:
                self.sim.render_from_thread(self.state)
                time.sleep(0.01)
                continue
            
            if execute_one:
                with self.state.lock:
                    self.state.execute_one_step = False

            if not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]:
                self._simulation_step()

            # 根據手動懸浮開關決定是否啟用懸浮控制器
            is_manual_mode = mode in ["JOINT_TEST", "MANUAL_CTRL"]
            if is_manual_mode and manual_float and not self._manual_float_active:
                self.floating_controller.enable(self.state.latest_pos)
                self._manual_float_active = True
            elif (not is_manual_mode or not manual_float) and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False
            
            # ============================== 步驟 3: 更新與渲染 ==============================
            # 更新衍生狀態 (如地形) 並渲染場景
            with self.state.lock:
                pos = self.state.latest_pos
                terrain_mode = self.state.terrain_mode
            self.update_derived_states_and_render(pos, terrain_mode)

        log.info("模擬執行緒已停止。")


    # ============================ 事件回呼處理函式 ============================
    def on_mode_change_requested(self, mode: str):
        """處理模式切換請求。"""
        log.info(f"接收到模式切換請求: {mode}")
        with self.state.lock:
            # 使用 pending 標誌來確保模式切換在主模擬執行緒中安全地進行
            self.state.control_mode_pending = mode
            
    def on_simulation_reset_requested(self, type: str):
        """
        【v3.1.3 核心修改】處理模擬重置請求。
        此函式現在只負責設定請求旗標，而不是直接執行重置操作。
        實際的重置操作將在主模擬迴圈中安全地執行。
        """
        log.info(f"接收到 '{type}' 重置請求，正在設定旗標...")
        with self.state.lock:
            if type == "hard":
                self.state.hard_reset_requested = True
            elif type == "soft":
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
        [v3.0.1] 處理序列埠命令發送請求。
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
        """處理手動模式下的懸浮開關請求。"""
        with self.state.lock:
            self.state.manual_mode_is_floating = is_floating
        log.info(f"手動懸浮狀態切換為: {is_floating}")

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


    def process_pending_mode_change(self) -> None:
        """只處理待處理的模式切換請求。"""
        with self.state.lock:
            pending_mode = self.state.control_mode_pending
            if pending_mode:
                self.state.control_mode_pending = None

        if pending_mode:
            self.handle_mode_change(self.state.control_mode, pending_mode)


    def handle_mode_change(self, old_mode: str, new_mode: str) -> None:
        """
        【v3.1.3 核心修改】執行模式切換並處理相關狀態。
        此函式現在由 run() 迴圈在安全的時間點呼叫。
        """
        self.state.set_control_mode(new_mode)
        is_headless = isinstance(self.sim, MockSimulation)

        # 離開舊模式時的物理處理
        if not is_headless:
            if old_mode == "FLOATING":
                self.floating_controller.disable()
            elif old_mode in ["JOINT_TEST", "MANUAL_CTRL"] and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False # 同步內部旗標

        # 進入新模式時的初始化
        if not is_headless:
            if new_mode == "FLOATING":
                self.floating_controller.enable(self.state.latest_pos)
            elif new_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                log.info(f"進入 {new_mode} 模式，重置機器人關節與速度")
                self.sim.data.qpos[7:] = self.sim.default_pose.copy()
                self.sim.data.qvel[6:] = 0
                if mujoco:
                    mujoco.mj_forward(self.sim.model, self.sim.data)

        if new_mode == "HARDWARE_MODE" and not self.hardware_controller.is_running:
            log.info("派生執行緒以啟動硬體控制器...")
            # 【修改】檢查啟動是否成功，並更新 state
            success = self.hardware_controller.start_controller_threads()
            with self.state.lock:
                self.state.hardware_is_running = success
                # 如果啟動失敗，應將模式切換回去，避免UI狀態不一致
                if not success:
                    self.state.control_mode = old_mode # 或者一個預設的安全模式如 WALKING
                    log.error("硬體控制器啟動失敗，模式已還原。")
                else:
                    self.state.set_control_mode(new_mode) # 只有成功才真正設定模式

        elif old_mode == "HARDWARE_MODE" and new_mode != "HARDWARE_MODE":
            if self.hardware_controller.is_running:
                log.info("派生執行緒以停止硬體控制器...")
                # 【修改】停止後更新 state
                self.hardware_controller.stop_controller_threads()
                with self.state.lock:
                    self.state.hardware_is_running = False
                self.state.set_control_mode(new_mode) # 執行模式切換


    def update_derived_states_and_render(self, pos, terrain_mode) -> None:
        """更新衍生狀態並渲染場景。"""
        is_headless = isinstance(self.sim, MockSimulation)

        if not is_headless and self.terrain_manager.is_functional and self.terrain_manager.needs_physics_and_scene_update:
            mujoco.mj_forward(self.sim.model, self.sim.data)
            mujoco.mjr_uploadHField(self.sim.model, self.sim.context, self.terrain_manager.hfield_id)
            self.terrain_manager.needs_physics_and_scene_update = False
            log.info("✅ 地形物理與渲染已同步更新。")

        if not is_headless:
            with self.state.lock:
                self.state.latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.latest_quat = self.sim.data.body('torso').xquat.copy()
                # 將當前關節角度複製到共享狀態，避免 UI 執行緒直接讀取 sim.data
                self.state.latest_joint_positions = self.sim.data.qpos[7:].copy()

        if self.terrain_manager.is_functional:
            self.terrain_manager.update(pos, terrain_mode)

        self.sim.render_from_thread(self.state)

    # ------------------------------------------------------------------
    def _simulation_step(self) -> None:
        with self.state.lock:
            command = self.state.command.copy()
            control_mode = self.state.control_mode
            tuning_params = self.state.tuning_params

        onnx_input, action_final = self.policy_manager.get_action(command)

        if control_mode == "MANUAL_CTRL":
            with self.state.lock:
                final_ctrl = self.state.manual_final_ctrl.copy()
        elif control_mode == "JOINT_TEST":
            with self.state.lock:
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else:
            final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale

        self.sim.apply_position_control(final_ctrl, tuning_params)

        with self.state.lock:
            self.state.latest_onnx_input = onnx_input.flatten()
            self.state.latest_action_raw = action_final
            self.state.latest_final_ctrl = final_ctrl

        target_time = self.sim.data.time + self.config.control_dt
        while self.sim.data.time < target_time:
            if not self._running.is_set():
                break
            mujoco.mj_step(self.sim.model, self.sim.data)


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
            self.state.hard_reset_requested = False
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def soft_reset(self) -> None:
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
            self.state.soft_reset_requested = False



