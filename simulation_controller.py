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
        """執行緒進入點：負責處理所有請求並運行模擬。"""
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context()
            self._initialize_simulation_state()
        else:
            print("[MOCK] Headless mode, skip window/context init.")
            # 無頭模式不需要初始化真實模擬狀態

        while self._running.is_set():
            with self.state.lock:
                shutdown_req = self.state.shutdown_requested
            should_close = shutdown_req
            if not is_headless:
                should_close = should_close or self.sim.should_close()
            if should_close:
                if shutdown_req and not is_headless and not self.sim.should_close():
                    log.info("偵測到全域關閉請求，正在關閉模擬視窗...")
                    from glfw import set_window_should_close
                    set_window_should_close(self.sim.window, 1)
                self._running.clear()
                from nicegui import app
                app.shutdown()
                continue

            # 1) 先處理所有待辦請求
            self.process_pending_mode_change()

            # 2) 讀取必要狀態
            with self.state.lock:
                mode = self.state.control_mode
                terrain_mode = self.state.terrain_mode
                pos = self.state.latest_pos
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

            self.update_derived_states_and_render(pos, terrain_mode)

        print("模擬執行緒已停止。")

    # ============================ 事件回呼處理函式 ============================
    def on_mode_change_requested(self, mode: str):
        """處理模式切換請求。"""
        log.info(f"接收到模式切換請求: {mode}")
        with self.state.lock:
            # 使用 pending 標誌來確保模式切換在主模擬執行緒中安全地進行
            self.state.control_mode_pending = mode
            
    def on_simulation_reset_requested(self, type: str):
        """處理模擬重置請求。"""
        log.info(f"接收到 '{type}' 重置請求。")
        # 直接在事件回呼中執行重置，因為它們被設計為可以在模擬迴圈的任何點安全呼叫
        if type == "hard":
            self.hard_reset()
        elif type == "soft":
            self.soft_reset()

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
        """執行模式切換並處理硬體控制執行緒與模擬狀態。"""
        self.state.set_control_mode(new_mode)

        is_headless = isinstance(self.sim, MockSimulation)

        if not is_headless:
            # 離開舊模式時的物理處理
            if old_mode == "FLOATING":
                self.floating_controller.disable()
            elif old_mode == "MANUAL_CTRL" and self.state.manual_mode_is_floating:
                self.floating_controller.disable()
                self.state.manual_mode_is_floating = False

            # 進入新模式時的初始化
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
            threading.Thread(target=self.hardware_controller.start_controller_threads, daemon=True).start()
        elif old_mode == "HARDWARE_MODE" and new_mode != "HARDWARE_MODE":
            if self.hardware_controller.is_running:
                log.info("派生執行緒以停止硬體控制器...")
                threading.Thread(target=self.hardware_controller.stop_controller_threads, daemon=True).start()


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



