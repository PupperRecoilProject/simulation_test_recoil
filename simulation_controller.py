"""Run MuJoCo simulation in a background thread."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type hints
    from state import SimulationState


class SimulationController:
    """Handle stepping and rendering of the simulation in its own thread."""

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.xbox_handler = state.xbox_handler_ref

        self._running = threading.Event()
        self._running.set()

        self._initialize_simulation()

    # ------------------------------------------------------------------
    def _initialize_simulation(self) -> None:
        if self.terrain_manager.is_functional:
            self.terrain_manager.initial_generate()
        self.hard_reset()
        print("\n--- Simulation Started (SPACE: Pause, N: Step) ---")

    # ------------------------------------------------------------------
    def run(self) -> None:
        while self._running.is_set():
            with self.state.lock:
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
                current_input_mode = self.state.input_mode
                hard_reset_req = self.state.hard_reset_requested
                soft_reset_req = self.state.soft_reset_requested
                current_control_mode = self.state.control_mode

            if single_step and not execute_one:
                self.sim.render_from_thread(self.state)
                time.sleep(0.01)
                continue
            if execute_one:
                with self.state.lock:
                    self.state.execute_one_step = False

            if hard_reset_req:
                self.hard_reset()
            if soft_reset_req:
                self.soft_reset()

            if current_input_mode == "GAMEPAD":
                self.xbox_handler.update_state()

            if current_control_mode in ["HARDWARE_MODE", "SERIAL_MODE"]:
                pass
            else:
                self._simulation_step()

            with self.state.lock:
                self.state.latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.latest_quat = self.sim.data.body('torso').xquat.copy()

            if self.terrain_manager.is_functional:
                self.terrain_manager.update(self.state.latest_pos, self.state.terrain_mode)

            self.sim.render_from_thread(self.state)

        print("simulation thread stopped")

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

    # ------------------------------------------------------------------
    def hard_reset(self) -> None:
        print("\n--- 正在執行機器人硬重置 ---")
        with self.state.lock:
            if self.state.control_mode == "HARDWARE_MODE":
                return

            mujoco.mj_resetData(self.sim.model, self.sim.data)
            self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0
            start_ground_z = self.terrain_manager.get_height_at(0, 0)
            robot_height_offset = 0.3
            self.sim.data.qpos[2] = start_ground_z + robot_height_offset
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
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.state.soft_reset_requested = False

