from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from src.core.logger import log
import queue

import numpy as np
from src.mock.mock_simulation import MockSimulation

# 【v4.5.0 最終修正】 只導入 SimulationController 真正關心的事件
from src.core.event_system import (
    event_bus,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_MODE_CHANGED,
    EVENT_SIMULATION_RESET_REQUESTED, # 雖然不處理，但在地形變更時需要發布
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
    【v4.5.0 修改】 在獨立執行緒中運行模擬並處理所有狀態變更。
    職責已純化為：
    1.  以固定的 50Hz 頻率驅動 MuJoCo 物理模擬。
    2.  將原始物理數據寫入 SimulationState 的 raw_ 屬性。
    3.  將渲染所需的物理數據寫入 SimulationState 的 render_data_buffer。
    4.  處理與物理狀態直接相關的事件。
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
        
        self.mode_change_queue = queue.Queue()

        self._manual_float_active = False
        
        self.num_physics_steps_per_control_step = int(
            self.config.control_dt / self.config.physics_timestep
        )
        if not np.isclose(self.config.control_dt, self.num_physics_steps_per_control_step * self.config.physics_timestep):
            log.warning(f"Control DT ({self.config.control_dt}) 不是 Physics Timestep ({self.config.physics_timestep}) 的整數倍，可能導致時間漂移。")
        
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """
        【v4.5.0 最終修正】 只訂閱自身職責範圍內的事件。
        """
        event_bus.subscribe(EVENT_POLICY_CHANGE_REQUESTED, self.on_policy_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_CHANGE_REQUESTED, self.on_terrain_change_requested)
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLED, self.on_manual_float_toggled)
        event_bus.subscribe(EVENT_JOINT_SELECT_REQUESTED, self.on_joint_select_requested)
        event_bus.subscribe(EVENT_JOINT_VALUE_ADJUSTED, self.on_joint_value_adjusted)
        event_bus.subscribe(EVENT_MODE_CHANGED, self.on_mode_changed)

        log.info("SimulationController 已訂閱其核心職責事件。")

    def start(self) -> None:
        """啟動模擬執行緒。"""
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, name="SimulationThread", daemon=True)
        self.thread.start()

    def run(self) -> None:
        """
        【v4.5.0 最終修正】 主迴圈不再負責初始化或處理全局請求。
        """
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

            # 【v4.0.2 修正】UX 優化
            is_simulation_active = not is_headless and mode not in ["HARDWARE_MODE", "SERIAL_MODE"]
            
            if is_simulation_active:
                # 【v4.5.3 最終權威修正 - 時序問題】
                # 根本原因：舊的順序是先 _simulation_step (使用 T-1 的數據)，
                # 然後才 _update_derived_and_render_states (生成 T 的數據)，
                # 這導致 AI 控制存在一幀的延遲。
                #
                # 解決方案：顛倒執行順序。
                # 1. 首先，更新所有衍生狀態，確保 AI 能看到最新的物理世界。
                # 2. 然後，執行模擬步驟，讓 AI 根據最新的狀態做出決策。
                
                # 步驟 1: 更新狀態，讓 AI 看到最新的物理數據
                self._update_derived_and_render_states()
                
                # 步驟 2: 執行模擬，讓 AI 根據最新數據行動
                self._simulation_step()

    def on_mode_changed(self, old_mode: str, new_mode: str):
        """[事件回呼] 接收到模式變更通知後，僅將請求放入隊列。"""
        self.mode_change_queue.put(new_mode)
        
    def _execute_mode_change(self, new_mode: str):
        """[內部執行] 在主迴圈的安全上下文中，執行實際的模式切換邏輯。"""
        log.debug(f"SimCtrl 正在從隊列執行模式切換 -> {new_mode}")
        
        with self.state.lock:
            old_mode = self.state.previous_control_mode
        
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
            # 進入手動模式時，只需重置姿態，無需重置位置
            self.soft_reset()

    def on_policy_change_requested(self, policy_name: str):
        """處理AI策略切換請求。"""
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
        """處理關節選擇請求。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST": self.state.joint_test_index = index
            elif self.state.control_mode == "MANUAL_CTRL": self.state.manual_ctrl_index = index

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
                if clear: self.state.manual_final_ctrl[idx] = self.sim.default_pose[idx]
                elif value is not None: self.state.manual_final_ctrl[idx] = value
                elif direction is not None: self.state.manual_final_ctrl[idx] += direction

    def _update_derived_and_render_states(self) -> None:
        """
        【v4.5.0 新增】 更新所有依賴於核心物理狀態的衍生狀態，並將數據寫入渲染緩衝區。
        """
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
            # 【時序檢驗探針 1】 記錄數據的生產時間戳
            production_time = time.perf_counter()
            self.state.render_data_buffer = {
                'time': self.sim.data.time, 
                'qpos': self.sim.data.qpos.copy(),
                'production_timestamp': production_time  # 將高精度時間戳放入數據包
            }
            log.info(f"[SimCtrl] 數據生產 @ {production_time:.6f} (Sim Time: {self.sim.data.time:.4f})")

    def _simulation_step(self) -> None:
        """
        【v4.5.0 修改】
        此函式現在除了執行模擬，還負責將原始物理數據寫入 SimulationState。
        物理步進迴圈已被修改為確定性的 for 迴圈。
        """
        # [保留] 讀取狀態和獲取 AI 動作的邏輯不變
        with self.state.lock:
            command, control_mode, tuning_params = self.state.command.copy(), self.state.control_mode, self.state.tuning_params
        onnx_input, action_final = self.policy_manager.get_action(command)

        # [保留] 根據模式計算最終控制指令的邏輯不變
        if control_mode == "MANUAL_CTRL":
            with self.state.lock:
                final_ctrl = self.state.manual_final_ctrl.copy()
        elif control_mode == "JOINT_TEST":
            with self.state.lock:
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
        else:
            final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale

        # [保留] 應用 PD 控制的邏輯不變
        self.sim.apply_position_control(final_ctrl, tuning_params)

        # [保留] 更新 UI 顯示用的數據的邏輯不變
        with self.state.lock:
            self.state.latest_onnx_input, self.state.latest_action_raw, self.state.latest_final_ctrl = onnx_input.flatten(), action_final, final_ctrl
        for _ in range(self.num_physics_steps_per_control_step):
            if not self._running.is_set():
                break
            # 這是物理引擎的核心步驟
            mujoco.mj_step(self.sim.model, self.sim.data)

    def stop(self) -> None:
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)


    # ------------------------------------------------------------------
    def hard_reset(self) -> None:
        """【v4.5.0 最終修正】只負責物理重置，並將重置後的狀態立即同步到渲染緩衝區。"""
        log.info("--- (SimCtrl) 正在執行物理硬重置 ---")
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

    def soft_reset(self) -> None:
        """【v4.5.0 最終修正】只負責物理軟重置，並同步狀態。"""
        log.info("--- (SimCtrl) 正在執行物理軟重置 ---")
        self.sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        self.sim.data.qpos[7:] = self.sim.default_pose
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = self.sim.default_pose
        mujoco.mj_forward(self.sim.model, self.sim.data)
        with self.state.render_data_lock:
            self.state.render_data_buffer = {'time': self.sim.data.time, 'qpos': self.sim.data.qpos.copy()}
        log.info("--- (SimCtrl) 已將軟重置後的新狀態同步至渲染緩衝區。 ---")