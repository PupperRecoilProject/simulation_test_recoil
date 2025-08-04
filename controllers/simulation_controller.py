"""Run MuJoCo simulation in a background thread.

以背景執行緒方式運行模擬，並整合新的 OperatingMode 與 ControlSubMode 架構。"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
import numpy as np

from utils.logger import log
from mock_simulation import MockSimulation
from state import OperatingMode, ControlSubMode

try:
    import mujoco
except ImportError:  # 無頭環境可能沒有安裝
    mujoco = None

if TYPE_CHECKING:
    from state import SimulationState


class SimulationController:
    """在獨立執行緒中運行 MuJoCo 模擬。"""

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        # 其他模組參考
        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.hardware_controller = state.hardware_controller_ref

        self._running = threading.Event()
        self.thread: threading.Thread | None = None

        # 手動模式下懸浮控制器啟動狀態
        self._manual_float_active = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    # ------------------------------------------------------------------
    def _initialize_simulation_state(self) -> None:
        if isinstance(self.sim, MockSimulation):
            log.info("[MOCK] Skip simulation state initialization.")
            return
        if self.terrain_manager.is_functional:
            self.terrain_manager.reset()
        self.hard_reset()
        log.info("\n--- Simulation Started ---")

    # ------------------------------------------------------------------
    def run(self) -> None:
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context(self.state)
            self._initialize_simulation_state()

        while self._running.is_set():
            # --- 檢查關閉請求 ---
            with self.state.lock:
                shutdown_req = self.state.shutdown_requested
            should_close = shutdown_req or (not is_headless and self.sim.should_close())
            if should_close:
                if shutdown_req and not is_headless and not self.sim.should_close():
                    from glfw import set_window_should_close
                    set_window_should_close(self.sim.window, 1)
                self._running.clear()
                from nicegui import app
                app.shutdown()
                continue

            # --- 處理請求並讀取當前模式 ---
            self.process_requests()
            with self.state.lock:
                op_mode = self.state.operating_mode
                sub_mode = self.state.control_sub_mode
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step

            if single_step and not execute_one:
                self.sim.render_from_thread(self.state)
                time.sleep(0.01)
                continue
            if execute_one:
                with self.state.lock:
                    self.state.execute_one_step = False

            # 只有在模擬模式下才進行物理步進
            if not is_headless and op_mode == OperatingMode.SIMULATION:
                self._simulation_step()
            else:
                time.sleep(1.0 / 60.0)

            self.update_derived_states_and_render()

        log.info("模擬執行緒已停止。")

    # ------------------------------------------------------------------
    def process_requests(self) -> None:
        """檢查並處理所有待處理的狀態變更請求。"""
        with self.state.lock:
            hard_reset = self.state.hard_reset_requested
            soft_reset = self.state.soft_reset_requested
            if hard_reset:
                self.state.hard_reset_requested = False
            if soft_reset:
                self.state.soft_reset_requested = False

        # 啟停硬體控制器
        self.manage_hardware_controller()

        if not isinstance(self.sim, MockSimulation):
            if hard_reset:
                self.hard_reset()
            if soft_reset:
                self.soft_reset()

    def manage_hardware_controller(self) -> None:
        """根據頂層操作模式決定是否啟動硬體控制器。"""
        with self.state.lock:
            op_mode = self.state.operating_mode
        is_hw_running = self.hardware_controller.is_running
        if op_mode == OperatingMode.HARDWARE and not is_hw_running:
            log.info("偵測到切換至硬體模式，啟動硬體控制器...")
            threading.Thread(target=self.hardware_controller.start, daemon=True).start()
        elif op_mode == OperatingMode.SIMULATION and is_hw_running:
            log.info("偵測到切換至模擬模式，停止硬體控制器...")
            threading.Thread(target=self.hardware_controller.stop, daemon=True).start()

    # ------------------------------------------------------------------
    def update_derived_states_and_render(self) -> None:
        with self.state.lock:
            op_mode = self.state.operating_mode
            sub_mode = self.state.control_sub_mode
            manual_float_req = self.state.manual_mode_is_floating
            sim_pos = self.state.sim_latest_pos
            terrain_mode = self.state.terrain_mode

        if op_mode == OperatingMode.SIMULATION:
            is_manual_mode = sub_mode in (ControlSubMode.JOINT_TEST, ControlSubMode.MANUAL_CTRL)
            if is_manual_mode and manual_float_req and not self._manual_float_active:
                self.floating_controller.enable(sim_pos)
                self._manual_float_active = True
            elif (not is_manual_mode or not manual_float_req) and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False

        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            if self.terrain_manager.is_functional and self.terrain_manager.needs_physics_and_scene_update:
                mujoco.mj_forward(self.sim.model, self.sim.data)
                mujoco.mjr_uploadHField(self.sim.model, self.sim.context, self.terrain_manager.hfield_id)
                self.terrain_manager.needs_physics_and_scene_update = False
                log.info("✅ 地形物理與渲染已同步更新。")

            with self.state.lock:
                self.state.sim_latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.sim_latest_quat = self.sim.data.body('torso').xquat.copy()
                self.state.sim_latest_joint_positions = self.sim.data.qpos[7:].copy()

        if self.terrain_manager.is_functional:
            self.terrain_manager.update(sim_pos, terrain_mode)

        self.sim.render_from_thread(self.state)

    # ------------------------------------------------------------------
    def _simulation_step(self) -> None:
        with self.state.lock:
            command = self.state.command.copy()
            sub_mode = self.state.control_sub_mode
            tuning = self.state.tuning_params

        onnx_input, action_final = self.policy_manager.get_action(command)

        if sub_mode == ControlSubMode.MANUAL_CTRL:
            with self.state.lock:
                final_ctrl = self.state.manual_final_ctrl.copy()
        elif sub_mode == ControlSubMode.JOINT_TEST:
            with self.state.lock:
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else:  # WALKING or FLOATING
            final_ctrl = self.sim.default_pose + action_final * tuning.action_scale

        self.sim.apply_position_control(final_ctrl, tuning)

        with self.state.lock:
            self.state.sim_latest_onnx_input = onnx_input.flatten()
            self.state.sim_latest_action_raw = action_final
            self.state.sim_latest_final_ctrl = final_ctrl

        # 使用 state.control_dt 以支援動態控制頻率
        target_time = self.sim.data.time + self.state.control_dt
        while self.sim.data.time < target_time:
            if not self._running.is_set():
                break
            mujoco.mj_step(self.sim.model, self.sim.data)

    # ------------------------------------------------------------------
    def hard_reset(self) -> None:
        log.info("\n--- 正在執行機器人硬重置 ---")
        with self.state.lock:
            if self.state.operating_mode == OperatingMode.HARDWARE:
                log.warning("硬重置請求在硬體模式下被忽略。")
                return
            terrain_name = self.terrain_manager.get_current_terrain_name_simple(self.state)
        start_z_offset = 1.5 if terrain_name in ["Pyramid", "Stepped Pyramid"] else 0.3
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0
        start_ground_z = self.terrain_manager.get_height_at(0, 0)
        self.sim.data.qpos[2] = start_ground_z + start_z_offset
        self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        self.sim.data.qpos[7:] = self.sim.default_pose
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = self.sim.default_pose
        for _ in range(10):
            mujoco.mj_step(self.sim.model, self.sim.data)
        self.policy_manager.reset()
        with self.state.lock:
            if self.state.control_sub_mode == ControlSubMode.FLOATING:
                self.state.control_sub_mode = ControlSubMode.WALKING
            self.state.command.fill(0.0)
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False
        if self._manual_float_active:
            self.floating_controller.disable()
            self._manual_float_active = False
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def soft_reset(self) -> None:
        log.info("\n--- 正在執行空中姿態重置 ---")
        with self.state.lock:
            if self.state.operating_mode == OperatingMode.HARDWARE:
                log.warning("軟重置請求在硬體模式下被忽略。")
                return
        self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        self.sim.data.qpos[7:] = self.sim.default_pose
        self.sim.data.qvel[:] = 0
        self.policy_manager.reset()
        with self.state.lock:
            self.state.command.fill(0.0)
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False
        if self._manual_float_active:
            self.floating_controller.disable()
            self._manual_float_active = False
        mujoco.mj_forward(self.sim.model, self.sim.data)
