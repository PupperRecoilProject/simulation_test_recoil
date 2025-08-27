from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from src.core.logger import log
import queue

import numpy as np
from src.mock.mock_simulation import MockSimulation

from src.core.event_system import (
    event_bus,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_MODE_CHANGED,
    EVENT_SIMULATION_RESET_REQUESTED,
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    from src.core.state import SimulationState


class SimulationController:
    """
    【v4.5.3 修改】 在獨立執行緒中運行模擬並透過隊列處理狀態變更。
    """
    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        
        self._running = threading.Event()
        self.thread: threading.Thread | None = None
        
        # 【v4.5.3 最終權威修正】 創建專用的、執行緒安全的請求隊列
        self.mode_change_queue = queue.Queue()
        self.reset_queue = queue.Queue()

        self._manual_float_active = False
        
        self.num_physics_steps_per_control_step = int(
            self.config.control_dt / self.config.physics_timestep
        )
        if not np.isclose(self.config.control_dt, self.num_physics_steps_per_control_step * self.config.physics_timestep):
            log.warning(f"Control DT ({self.config.control_dt}) 不是 Physics Timestep ({self.config.physics_timestep}) 的整數倍，可能導致時間漂移。")
        
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """訂閱自身職責範圍內的事件。"""
        event_bus.subscribe(EVENT_POLICY_CHANGE_REQUESTED, self.on_policy_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_CHANGE_REQUESTED, self.on_terrain_change_requested)
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLED, self.on_manual_float_toggled)
        event_bus.subscribe(EVENT_JOINT_SELECT_REQUESTED, self.on_joint_select_requested)
        event_bus.subscribe(EVENT_JOINT_VALUE_ADJUSTED, self.on_joint_value_adjusted)
        event_bus.subscribe(EVENT_MODE_CHANGED, self.on_mode_changed)
        log.info("SimulationController 已訂閱其核心職責事件。")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, name="SimulationThread", daemon=True)
        self.thread.start()

    def run(self) -> None:
        """主迴圈，負責驅動物理模擬和處理請求隊列。"""
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            log.info("--- 模擬執行緒已啟動並接管物理迴圈 ---")

        last_control_time = time.perf_counter()
        while self._running.is_set():
            current_time = time.perf_counter()
            elapsed = current_time - last_control_time
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt - elapsed)
            last_control_time = time.perf_counter()
            
            # 【v4.5.3 最終權威修正】 在主迴圈的安全上下文中處理所有請求
            # 處理重置請求
            try:
                reset_type = self.reset_queue.get_nowait()
                if reset_type == "hard":
                    self._execute_hard_reset()
                elif reset_type == "soft":
                    self._execute_soft_reset()
            except queue.Empty:
                pass

            # 處理模式切換請求
            try:
                new_mode_request = self.mode_change_queue.get_nowait()
                self._execute_mode_change(new_mode_request)
            except queue.Empty:
                pass

            with self.state.lock:
                mode = self.state.control_mode
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
            
            if single_step and not execute_one:
                time.sleep(0.01)
                continue
            
            if execute_one:
                with self.state.lock: self.state.execute_one_step = False

            is_simulation_active = not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]
            if is_simulation_active:
                self._update_derived_and_render_states()
                self._simulation_step()

    def on_mode_changed(self, old_mode: str, new_mode: str):
        self.mode_change_queue.put(new_mode)
        
    def _execute_mode_change(self, new_mode: str):
        log.debug(f"SimCtrl 正在從隊列執行模式切換 -> {new_mode}")
        
        with self.state.lock:
            old_mode = self.state.previous_control_mode
        
        if old_mode == "FLOATING":
            self.floating_controller.disable()
            self._manual_float_active = False
        if new_mode == "FLOATING":
            current_pos = self.sim.data.body('torso').xpos.copy()
            self.floating_controller.enable(current_pos)
            self._manual_float_active = True
        if new_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
            self.soft_reset()

    def on_policy_change_requested(self, policy_name: str):
        if self.policy_manager:
            self.policy_manager.select_target_policy(policy_name)

    def on_terrain_change_requested(self, name: str):
        log.info(f"SimulationController 接收到切換地形請求: {name}")
        needs_reset = False
        with self.state.lock:
            if name == 'INFINITE':
                if self.state.terrain_mode != 'INFINITE':
                    self.state.terrain_mode, needs_reset = 'INFINITE', True
                    self.terrain_manager.reset()
            else:
                if name in self.terrain_manager.single_terrain_names:
                    new_index = self.terrain_manager.single_terrain_names.index(name)
                    if self.state.terrain_mode != 'SINGLE' or self.state.single_terrain_index != new_index:
                        self.state.terrain_mode, self.state.single_terrain_index, needs_reset = 'SINGLE', new_index, True
                        self.terrain_manager.set_single_terrain(name)
        if needs_reset:
            event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")

    def on_manual_float_toggled(self, is_floating: bool):
        is_manual_mode = self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]
        if not is_manual_mode: return
        if is_floating and not self._manual_float_active:
            current_pos = self.sim.data.body('torso').xpos.copy()
            self.floating_controller.enable(current_pos)
            self._manual_float_active = True
        elif not is_floating and self._manual_float_active:
            self.floating_controller.disable()
            self._manual_float_active = False
        with self.state.lock:
            self.state.manual_mode_is_floating = self._manual_float_active

    def on_joint_select_requested(self, index: int):
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST": self.state.joint_test_index = index
            elif self.state.control_mode == "MANUAL_CTRL": self.state.manual_ctrl_index = index

    def on_joint_value_adjusted(self, value: float = None, direction: float = None, clear: bool = False):
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                idx = self.state.joint_test_index
                if clear: self.state.joint_test_offsets[idx] = 0.0
                elif value is not None: self.state.joint_test_offsets[idx] = value - self.sim.default_pose[idx]
                elif direction is not None: self.state.joint_test_offsets[idx] += direction
            elif self.state.control_mode == "MANUAL_CTRL":
                idx = self.state.manual_ctrl_index
                if clear: self.state.manual_final_ctrl[idx] = self.sim.default_pose[idx]
                elif value is not None: self.state.manual_final_ctrl[idx] = value
                elif direction is not None: self.state.manual_final_ctrl[idx] += direction
                
    def _update_derived_and_render_states(self) -> None:
        if not np.isfinite(self.sim.data.qpos).all():
            log.error(f"❌ 偵測到致命的物理不穩定 (qpos 包含 NaN/inf)，正在請求系統硬重置...")
            self.reset_queue.put("hard") # 將重置請求放入自己的隊列
            return

        current_pos = self.sim.data.body('torso').xpos.copy()
        with self.state.lock: terrain_mode = self.state.terrain_mode
        if self.terrain_manager.is_functional: self.terrain_manager.update(current_pos, terrain_mode)
        with self.state.lock:
            self.state.raw_torso_quat = self.sim.data.body('torso').xquat.copy()
            self.state.raw_torso_linear_velocity_world = self.sim.data.cvel[self.sim.torso_id, 3:].copy()
            self.state.raw_torso_angular_velocity_world = self.sim.data.cvel[self.sim.torso_id, :3:].copy()
            self.state.raw_joint_positions = self.sim.data.qpos[7:].copy()
            self.state.raw_joint_velocities = self.sim.data.qvel[6:].copy()
            if self.sim.accelerometer_id != -1:
                start, end = self.sim.model.sensor_adr[self.sim.accelerometer_id], self.sim.model.sensor_adr[self.sim.accelerometer_id] + self.sim.model.sensor_dim[self.sim.accelerometer_id]
                self.state.raw_accelerometer = self.sim.data.sensordata[start:end].copy()
            self.state.latest_pos, self.state.latest_quat, self.state.latest_joint_positions = current_pos, self.state.raw_torso_quat, self.state.raw_joint_positions
        with self.state.render_data_lock:
           self.state.render_data_buffer = {'time': self.sim.data.time, 'qpos': self.sim.data.qpos.copy()}

    def _simulation_step(self) -> None:
        with self.state.lock:
            command, control_mode, tuning_params = self.state.command.copy(), self.state.control_mode, self.state.tuning_params
        onnx_input, action_final = self.policy_manager.get_action(command)
        if control_mode == "MANUAL_CTRL":
            with self.state.lock: final_ctrl = self.state.manual_final_ctrl.copy()
        elif control_mode == "JOINT_TEST":
            with self.state.lock: final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else:
            final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale
        
        self.sim.apply_position_control(final_ctrl, tuning_params)
        
        with self.state.lock:
            self.state.latest_onnx_input, self.state.latest_action_raw, self.state.latest_final_ctrl = onnx_input.flatten(), action_final, final_ctrl
        for _ in range(self.num_physics_steps_per_control_step):
            if not self._running.is_set(): break
            mujoco.mj_step(self.sim.model, self.sim.data)

    def stop(self) -> None:
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    # 【v4.5.3 最終權威修正】 公共 API 只負責接收請求並放入隊列
    def hard_reset(self) -> None:
        """[公共 API] 請求一次硬重置。"""
        self.reset_queue.put("hard")

    def soft_reset(self) -> None:
        """[公共 API] 請求一次軟重置。"""
        self.reset_queue.put("soft")

    # 【v4.5.3 最終權威修正】 實際操作在主迴圈的私有方法中執行
    def _execute_hard_reset(self) -> None:
        """[內部執行] 執行物理硬重置。"""
        log.info("--- (SimCtrl) 正在從隊列執行物理硬重置 ---")
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0
        start_ground_z = self.terrain_manager.get_height_at(0, 0)
        self.sim.data.qpos[2] = start_ground_z + 0.3
        self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        self.sim.data.qpos[7:] = self.sim.default_pose
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = self.sim.default_pose
        mujoco.mj_forward(self.sim.model, self.sim.data)
        with self.state.render_data_lock:
            self.state.render_data_buffer = {'time': self.sim.data.time, 'qpos': self.sim.data.qpos.copy()}
        log.info("--- (SimCtrl) 已將重置後的新狀態同步至渲染緩衝區。 ---")

    def _execute_soft_reset(self) -> None:
        """[內部執行] 執行物理軟重置。"""
        log.info("--- (SimCtrl) 正在從隊列執行物理軟重置 ---")
        self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        self.sim.data.qpos[7:] = self.sim.default_pose
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = self.sim.default_pose
        mujoco.mj_forward(self.sim.model, self.sim.data)
        with self.state.render_data_lock:
            self.state.render_data_buffer = {'time': self.sim.data.time, 'qpos': self.sim.data.qpos.copy()}
        log.info("--- (SimCtrl) 已將軟重置後的新狀態同步至渲染緩衝區。 ---")