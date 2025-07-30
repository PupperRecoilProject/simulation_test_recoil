"""Run MuJoCo simulation in a background thread."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from logger import log

import numpy as np
from mock_simulation import MockSimulation

try:
    import mujoco
except ImportError:  # 無頭環境可能沒有安裝
    mujoco = None

if TYPE_CHECKING:  # pragma: no cover - type hints
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

        self._running = threading.Event()
        self.thread: threading.Thread | None = None

        # 追蹤手動模式下懸浮是否已啟用
        self._manual_float_active = False

        # 初始化將在執行緒啟動後進行

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
            self.process_requests()

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

    def process_requests(self) -> None:
        """檢查並處理所有待處理的狀態變更請求。"""
        with self.state.lock:
            pending_mode = self.state.control_mode_pending
            hard_reset = self.state.hard_reset_requested
            soft_reset = self.state.soft_reset_requested
            if pending_mode:
                self.state.control_mode_pending = None
            if hard_reset:
                self.state.hard_reset_requested = False
            if soft_reset:
                self.state.soft_reset_requested = False

        if pending_mode:
            self.handle_mode_change(self.state.control_mode, pending_mode)

        # 無頭模式下沒有真實模擬，跳過重置流程
        if not isinstance(self.sim, MockSimulation):
            if hard_reset:
                self.hard_reset()
            if soft_reset:
                self.soft_reset()

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
        start_z_offset = 1.0 if terrain_name in difficult else 0.3

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



