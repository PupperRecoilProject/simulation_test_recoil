# simulation_controller.py

import threading
import time
from typing import TYPE_CHECKING
from logger import log
import numpy as np
from mock_simulation import MockSimulation

try:
    import mujoco
except ImportError:
    mujoco = None

# [修改] 導入所有 SimulationController 需要響應的事件
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

if TYPE_CHECKING:
    from state import SimulationState

class SimulationController:
    """[重構] 在獨立執行緒中運行模擬，並作為所有請求事件的中央處理者。"""

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config
        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.xbox_handler = state.xbox_handler_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref

        self._running = threading.Event()
        self.thread: threading.Thread | None = None
        self._manual_float_active = False

        # [新增] 訂閱所有需要處理的請求事件
        self._subscribe_to_events()

    # [新增] 將所有事件訂閱集中到一個方法中，使 __init__ 更簡潔
    def _subscribe_to_events(self):
        """訂閱本控制器需要處理的所有事件。"""
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

    # [保留] start 和 run 的主體邏輯，但移除 process_requests
    def start(self) -> None:
        if self.thread and self.thread.is_alive(): return
        self._running.set()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        """
        [修改] 執行緒主迴圈。現在不再需要輪詢請求旗標。
        事件系統會直接在後台觸發對應的回呼函式，並在回呼函式中安全地修改 state。
        """
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context()
            self._initialize_simulation_state()
        
        while self._running.is_set():
            # 1. 檢查關閉條件 (現在由事件觸發，但保留手動關閉視窗的檢查)
            should_close = False
            if not is_headless:
                should_close = self.sim.should_close()

            if should_close:
                # 如果是視窗關閉觸發的，也發布一個全域關閉事件
                event_bus.publish(EVENT_SHUTDOWN_REQUESTED)
                from nicegui import app
                app.shutdown()
                continue
            
            # 2. 讀取必要狀態
            with self.state.lock:
                mode = self.state.control_mode
                terrain_mode = self.state.terrain_mode
                pos = self.state.latest_pos
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
            
            # 3. 處理暫停/單步邏輯 (這部分屬於迴圈自身狀態，不適合用事件)
            if single_step and not execute_one:
                self.sim.render_from_thread(self.state)
                time.sleep(0.01)
                continue
            if execute_one:
                with self.state.lock: self.state.execute_one_step = False
            
            # 4. 執行模擬步進
            if not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]:
                self._simulation_step()
            
            # 5. 更新衍生狀態並渲染
            self.update_derived_states_and_render(pos, terrain_mode)
            
        print("模擬執行緒已停止。")

    # [刪除] process_requests 函式。其功能已被下面的事件回呼函式取代。
    
    # [重構] handle_mode_change -> on_mode_change_requested
    def on_mode_change_requested(self, mode: str):
        """處理模式切換請求。"""
        with self.state.lock:
            old_mode = self.state.control_mode
            if old_mode == mode: return
            self.state.set_control_mode(mode) # 只更新模式狀態
            is_headless = isinstance(self.sim, MockSimulation)
            if not is_headless:
                if old_mode == "FLOATING": self.floating_controller.disable()
                if old_mode == "MANUAL_CTRL" and self._manual_float_active:
                    self.floating_controller.disable()
                    self._manual_float_active = False

                if mode == "FLOATING": self.floating_controller.enable(self.state.latest_pos)
            
            if mode == "HARDWARE_MODE" and not self.hardware_controller.is_running:
                log.info("派生執行緒以啟動硬體控制器...")
                threading.Thread(target=self.hardware_controller.start_controller_threads, daemon=True).start()
            elif old_mode == "HARDWARE_MODE" and mode != "HARDWARE_MODE":
                if self.hardware_controller.is_running:
                    log.info("派生執行緒以停止硬體控制器...")
                    threading.Thread(target=self.hardware_controller.stop_controller_threads, daemon=True).start()

    # [新增] 所有新的事件回呼函式
    def on_simulation_reset_requested(self, type: str):
        if type == "hard": self.hard_reset()
        elif type == "soft": self.soft_reset()

    def on_hardware_ai_toggle_requested(self):
        if self.state.hardware_ai_is_active: self.hardware_controller.disable_ai()
        else: self.hardware_controller.enable_ai()

    def on_input_mode_change_requested(self, mode: str):
        with self.state.lock: self.state.toggle_input_mode(mode)

    def on_device_connect_requested(self, device: str):
        if device == "serial":
            is_connected = self.serial_comm.scan_and_connect()
            with self.state.lock: self.state.serial_is_connected = is_connected
        elif device == "gamepad":
            is_connected = self.xbox_handler.scan_and_connect()
            with self.state.lock: self.state.gamepad_is_connected = is_connected

    def on_policy_change_requested(self, policy_name: str):
        if self.policy_manager: self.policy_manager.select_target_policy(policy_name)

    def on_terrain_mode_change_requested(self, mode_name: str = None, direction: int = None):
        with self.state.lock:
            if mode_name: # 來自 UI 的指定模式
                new_mode = 'INFINITE' if mode_name == 'INFINITE' else 'SINGLE'
                if new_mode == 'SINGLE':
                    self.state.single_terrain_index = self.terrain_manager.single_terrain_names.index(mode_name)
                self.state.terrain_mode = new_mode
            elif direction: # 來自鍵盤的循環切換
                if self.state.terrain_mode == 'INFINITE':
                    self.state.terrain_mode = 'SINGLE'
                    self.state.single_terrain_index = 0
                else:
                    num_terrains = len(self.terrain_manager.single_terrain_names)
                    self.state.single_terrain_index = (self.state.single_terrain_index + direction) % num_terrains
            
            if self.terrain_manager.is_functional:
                if self.state.terrain_mode == 'INFINITE': self.terrain_manager.reset()
                else: self.terrain_manager.set_single_terrain(self.terrain_manager.single_terrain_names[self.state.single_terrain_index])
                self.hard_reset()

    def on_terrain_regenerate_requested(self):
        if self.terrain_manager.is_functional and self.state.terrain_mode == 'INFINITE':
            self.terrain_manager.regenerate_terrain_and_adjust_robot(self.state.latest_pos)

    def on_terrain_snapshot_requested(self):
        if self.terrain_manager: self.terrain_manager.save_hfield_to_png()
    
    def on_tuning_param_adjust_requested(self, param_name: str, value: float = None, direction: int = None):
        with self.state.lock:
            if value is not None: # 來自滑桿的絕對值
                setattr(self.state.tuning_params, param_name, value)
            elif direction is not None: # 來自按鍵的相對調整
                step = self.config.param_adjust_steps.get(param_name, 0.1)
                current_value = getattr(self.state.tuning_params, param_name)
                setattr(self.state.tuning_params, param_name, current_value + step * direction)
            
            self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
            self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
            self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)

    def on_joint_select_requested(self, direction: int = 0, index: int = -1):
        with self.state.lock:
            if index != -1: # 來自 UI 的絕對索引
                self.state.joint_test_index = index
                self.state.manual_ctrl_index = index
            elif direction != 0: # 來自鍵盤的相對索引
                if self.state.control_mode == "JOINT_TEST":
                    self.state.joint_test_index = (self.state.joint_test_index + direction) % 12
                elif self.state.control_mode == "MANUAL_CTRL":
                    self.state.manual_ctrl_index = (self.state.manual_ctrl_index + direction) % 12

    def on_joint_value_adjust_requested(self, value: float = None, direction: int = None, step: float = 0.1, clear: bool = False):
        with self.state.lock:
            mode = self.state.control_mode
            idx = self.state.joint_test_index if mode == "JOINT_TEST" else self.state.manual_ctrl_index
            
            if mode == "JOINT_TEST":
                if clear: self.state.joint_test_offsets[idx] = 0.0
                elif value is not None: self.state.joint_test_offsets[idx] = value - self.sim.default_pose[idx]
                elif direction is not None: self.state.joint_test_offsets[idx] += direction * step
            elif mode == "MANUAL_CTRL":
                if clear: self.state.manual_final_ctrl[idx] = 0.0
                elif value is not None: self.state.manual_final_ctrl[idx] = value
                elif direction is not None: self.state.manual_final_ctrl[idx] += direction * step

    def on_manual_float_toggle_requested(self, value: bool):
        with self.state.lock:
            self.state.manual_mode_is_floating = value
            is_manual_mode = self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]
            if is_manual_mode and value and not self._manual_float_active:
                self.floating_controller.enable(self.state.latest_pos)
                self._manual_float_active = True
            elif (not is_manual_mode or not value) and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False

    def on_ui_page_change_requested(self, direction: int):
        with self.state.lock:
            self.state.display_page = (self.state.display_page + direction) % self.state.num_display_pages



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

