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
        self.state = state # 儲存對全域狀態物件的參考
        self.sim = state.sim # 儲存對模擬物件的參考
        self.config = state.config # 儲存對設定物件的參考

        self.policy_manager = state.policy_manager_ref # 儲存對策略管理器的參考
        self.terrain_manager = state.terrain_manager_ref # 儲存對地形管理器的參考
        self.floating_controller = state.floating_controller_ref # 儲存對懸浮控制器的參考
        self.xbox_handler = state.xbox_handler_ref # 儲存對Xbox搖桿處理器的參考
        self.hardware_controller = state.hardware_controller_ref # 儲存對硬體控制器的參考

        self._running = threading.Event() # 用於控制執行緒是否應該繼續運行的事件旗標
        self.thread: threading.Thread | None = None # 將要運行的背景執行緒

        # 追蹤手動模式下懸浮是否已啟用
        self._manual_float_active = False # 控制器內部用來追蹤懸浮物理狀態的旗標

    def _initialize_simulation_state(self) -> None:
        """採用極簡初始化，解決白屏問題。"""
        if isinstance(self.sim, MockSimulation): # 檢查是否為無頭模擬模式
            log.info("[MOCK] Skip simulation state initialization.")
            return

        if self.terrain_manager.is_functional: # 如果地形管理器可用
            self.terrain_manager.reset() # 重置地形管理器狀態

        mujoco.mj_resetData(self.sim.model, self.sim.data) # 重置MuJoCo的物理數據

        home_key_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, 'home') # 查找名為'home'的關鍵幀
        if home_key_id != -1:
            self.sim.data.qpos[:] = self.sim.model.key_qpos[home_key_id] # 使用keyframe的qpos作為初始位置和姿態
        else:
            log.warning("在XML中未找到 'home' keyframe，使用預設姿態。")
            self.sim.data.qpos[2] = 0.3 # 手動設定一個安全的初始高度
            self.sim.data.qpos[3] = 1.0 # 手動設定單位四元數（無旋轉）
            self.sim.data.qpos[7:] = self.sim.default_pose # 設定關節的初始角度

        mujoco.mj_forward(self.sim.model, self.sim.data) # 根據新的qpos計算所有運動學量
        self.policy_manager.reset() # 重置AI策略的內部狀態

        with self.state.lock:
             self.state.set_control_mode("WALKING") # 確保啟動時是走路模式
             self.state.reset_control_state(self.sim.data.time) # 重置控制計時器

        log.info("✅ Minimal simulation initialization complete.")
        print("\n--- Simulation Started ---")

    def start(self) -> None:
        """啟動模擬執行緒。"""
        if self.thread and self.thread.is_alive(): # 檢查執行緒是否已在運行
            return
        self._running.set() # 設定運行旗標為True
        self.thread = threading.Thread(target=self.run, daemon=True) # 建立背景執行緒
        self.thread.start() # 啟動執行緒

    def run(self) -> None:
        """執行緒進入點：負責處理所有請求並運行模擬。"""
        is_headless = isinstance(self.sim, MockSimulation) # 檢查是否為無頭模式
        if not is_headless:
            self.sim.initialize_window_and_context() # 初始化GLFW視窗和渲染上下文
            self._initialize_simulation_state() # 執行極簡的初始化
        else:
            print("[MOCK] Headless mode, skip window/context init.")

        while self._running.is_set(): # 主迴圈
            with self.state.lock:
                shutdown_req = self.state.shutdown_requested # 檢查是否有來自UI的關閉請求
            should_close = shutdown_req
            if not is_headless:
                should_close = should_close or self.sim.should_close() # 檢查視窗是否被手動關閉
            if should_close: # 如果需要關閉
                if shutdown_req and not is_headless and not self.sim.should_close():
                    log.info("偵測到全域關閉請求，正在關閉模擬視窗...")
                    from glfw import set_window_should_close
                    set_window_should_close(self.sim.window, 1) # 程式化地關閉GLFW視窗
                self._running.clear() # 清除運行旗標
                from nicegui import app
                app.shutdown() # 關閉NiceGUI應用
                continue

            self.process_requests() # 處理模式切換、重置等請求

            with self.state.lock:
                mode = self.state.control_mode
                terrain_mode = self.state.terrain_mode
                pos = self.state.latest_pos
                single_step = self.state.single_step_mode
                execute_one = self.state.execute_one_step
                manual_float = self.state.manual_mode_is_floating

            if single_step and not execute_one: # 如果是暫停模式且沒有單步請求
                self.sim.render_from_thread(self.state) # 僅渲染
                time.sleep(0.01)
                continue
            if execute_one: # 如果有單步請求
                with self.state.lock:
                    self.state.execute_one_step = False # 消耗掉請求

            if not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]:
                self._simulation_step() # 執行一個控制和物理步驟

            is_manual_mode = mode in ["JOINT_TEST", "MANUAL_CTRL"]
            if is_manual_mode and manual_float and not self._manual_float_active:
                self.floating_controller.enable(self.state.latest_pos)
                self._manual_float_active = True
            elif (not is_manual_mode or not manual_float) and self._manual_float_active:
                self.floating_controller.disable()
                self._manual_float_active = False

            self.update_derived_states_and_render(pos, terrain_mode) # 更新並渲染

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
            if old_mode == "FLOATING":
                self.floating_controller.disable()
            if old_mode in ["JOINT_TEST", "MANUAL_CTRL"] and self._manual_float_active:
                log.info(f"從 {old_mode} 模式離開，正在停用手動懸浮...")
                self.floating_controller.disable()
                self._manual_float_active = False
                with self.state.lock:
                    self.state.manual_mode_is_floating = False
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
                self.state.latest_joint_positions = self.sim.data.qpos[7:].copy()
        if self.terrain_manager.is_functional:
            self.terrain_manager.update(pos, terrain_mode)
        self.sim.render_from_thread(self.state)

    def _simulation_step(self) -> None:
        """執行一個完整的控制-物理模擬步驟，現在包含穩定化狀態的處理邏輯。"""
        with self.state.lock:
            command = self.state.command.copy()
            control_mode = self.state.control_mode
            tuning_params = self.state.tuning_params

        # 【核心修正：狀態機邏輯】根據目前的 control_mode 決定要執行什麼動作
        if control_mode == "STABILIZING":
            # 如果處於穩定化模式，施加預設姿態的控制
            final_ctrl = self.sim.default_pose
            # 檢查穩定化時間是否已到
            if self.sim.data.time >= self.state.stabilization_end_time:
                log.info("物理引擎穩定化完成，切換回 WALKING 模式。")
                self.state.set_control_mode("WALKING") # 結束穩定化，切換到正常行走
                self.policy_manager.reset() # 重置AI狀態
        
        elif control_mode in ["MANUAL_CTRL", "JOINT_TEST", "WALKING", "FLOATING"]:
            # 這些是需要AI或手動控制的模式
            onnx_input, action_final = self.policy_manager.get_action(command)

            if control_mode == "MANUAL_CTRL":
                with self.state.lock:
                    final_ctrl = self.state.manual_final_ctrl.copy()
            elif control_mode == "JOINT_TEST":
                with self.state.lock:
                    final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
            else: # WALKING 或 FLOATING
                final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale
            
            # 更新UI顯示用的狀態
            with self.state.lock:
                self.state.latest_onnx_input = onnx_input.flatten()
                self.state.latest_action_raw = action_final
        else:
            # 對於 HARDWARE_MODE 或 SERIAL_MODE，不需要計算 final_ctrl
            final_ctrl = self.sim.data.ctrl.copy() # 保持當前的控制指令

        self.sim.apply_position_control(final_ctrl, tuning_params) # 將最終計算出的控制指令應用到模擬器
        with self.state.lock:
            self.state.latest_final_ctrl = final_ctrl # 更新UI顯示用的最終控制指令
        
        # 推進物理時間
        target_time = self.sim.data.time + self.config.control_dt
        while self.sim.data.time < target_time:
            if not self._running.is_set():
                break
            mujoco.mj_step(self.sim.model, self.sim.data)

    def stop(self) -> None:
        """停止模擬執行緒。"""
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def hard_reset(self) -> None:
        """【最終修正：解決死鎖問題】重構為非阻塞的狀態切換請求。"""
        log.info("--- 正在請求機器人硬重置 ---")

        with self.state.lock:
            if self.state.control_mode == "HARDWARE_MODE": # 硬體模式下不執行模擬重置
                return

            # 1. 執行所有瞬間完成的重置操作
            terrain_name = self.terrain_manager.get_current_terrain_name_simple(self.state)
            start_z_offset = 1.5 if terrain_name in ["Pyramid", "Stepped Pyramid"] else 0.3
            log.info(f"地形: {terrain_name}, 高度偏移: {start_z_offset}m")

            if self.state.control_mode == "FLOATING" or self._manual_float_active:
                log.info("硬重置前偵測到懸浮已啟用，正在強制停用...")
                self.floating_controller.disable()
                self._manual_float_active = False

            mujoco.mj_resetData(self.sim.model, self.sim.data)
            self.sim.data.qpos[0], self.sim.data.qpos[1] = 0, 0
            start_ground_z = self.terrain_manager.get_height_at(0, 0)
            self.sim.data.qpos[2] = start_ground_z + start_z_offset
            self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
            self.sim.data.qpos[7:] = self.sim.default_pose
            self.sim.data.qvel[:] = 0
            
            # 2. 【關鍵】請求進入穩定化狀態，而不是在這裡執行迴圈
            log.info("請求進入物理穩定化狀態...")
            self.state.set_control_mode("STABILIZING")
            self.state.stabilization_end_time = self.sim.data.time + 0.2 # 設定穩定化結束的時間點

            # 3. 清理其他邏輯狀態
            self.state.clear_command()
            self.state.joint_test_offsets.fill(0.0)
            self.state.manual_final_ctrl.fill(0.0)
            self.state.manual_mode_is_floating = False
            
            mujoco.mj_forward(self.sim.model, self.sim.data)
            log.info("✅ 硬重置請求已處理，系統進入穩定化模式。")

    def soft_reset(self) -> None:
        """強化版軟重置，確保清理所有相關狀態。"""
        log.info("\n--- 正在執行空中姿態重置 ---")
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
            
            if self._manual_float_active:
                log.info("軟重置期間，停用手動懸浮...")
                self.floating_controller.disable()
                self._manual_float_active = False
            self.state.manual_mode_is_floating = False
            
            mujoco.mj_forward(self.sim.model, self.sim.data)