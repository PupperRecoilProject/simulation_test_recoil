from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from src.core.logger import log

import numpy as np
from src.mock.mock_simulation import MockSimulation

# 【v4.5.0 修正】 精簡事件導入，只導入 SimulationController 真正關心的事件
from src.core.event_system import (
    event_bus,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_MODE_CHANGED, # 仍然需要，用於處理模式切換後的狀態清理
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    from src.core.state import SimulationState


class SimulationController:
    """
    【v4.5.0 修改】 在獨立執行緒中運行模擬並處理所有狀態變更。
    職責已純化為：
    1.  以固定的 50Hz 頻率驅動 MuJoCo 物理模擬。
    2.  將原始物理數據寫入 SimulationState 的 raw_ 屬性。
    3.  將渲染所需的物理數據寫入 SimulationState 的 render_data_buffer。
    4.  處理與模擬時間步、重置和物理狀態直接相關的請求。
    """
    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref # 直接獲取 serial_communicator 的參考

        self._running = threading.Event()
        self.thread: threading.Thread | None = None

        self._manual_float_active = False
        self._subscribe_to_events()

        # 【v4.5.0 新增】 預先計算每個控制週期的物理步數，以確保確定性
        self.num_physics_steps_per_control_step = int(
            self.config.control_dt / self.config.physics_timestep
        )
        if not np.isclose(self.config.control_dt, self.num_physics_steps_per_control_step * self.config.physics_timestep):
            log.warning(f"Control DT ({self.config.control_dt}) 不是 Physics Timestep ({self.config.physics_timestep}) 的整數倍，可能導致時間漂移。")

    # ============================ 事件訂閱輔助函式 ============================
    # 【v4.5.0 重大修正】 移除所有不再屬於 SimulationController 職責的事件訂閱
    def _subscribe_to_events(self):
        """將所有事件訂閱邏輯集中到此處。"""
        # --- 保留的訂閱 (直接影響物理模擬) ---
        event_bus.subscribe(EVENT_SIMULATION_RESET_REQUESTED, self.on_simulation_reset_requested)
        event_bus.subscribe(EVENT_POLICY_CHANGE_REQUESTED, self.on_policy_change_requested)
        event_bus.subscribe(EVENT_TERRAIN_CHANGE_REQUESTED, self.on_terrain_change_requested)
        event_bus.subscribe(EVENT_MANUAL_FLOAT_TOGGLED, self.on_manual_float_toggled)
        event_bus.subscribe(EVENT_JOINT_SELECT_REQUESTED, self.on_joint_select_requested)
        event_bus.subscribe(EVENT_JOINT_VALUE_ADJUSTED, self.on_joint_value_adjusted)
        event_bus.subscribe(EVENT_SHUTDOWN_REQUESTED, self.on_shutdown_requested)

        # 【v4.5.0 修正】 仍然需要監聽 MODE_CHANGED 事件，以便在模式切換後進行物理狀態的清理
        # 注意：它不再監聽 MODE_CHANGE_REQUESTED
        event_bus.subscribe(EVENT_MODE_CHANGED, self.on_mode_changed)

        log.info("SimulationController 已訂閱其核心職責事件。")

    def _initialize_simulation_state(self) -> None:
        """【v4.5.0 修改】 此函式不再初始化視窗，只初始化物理狀態"""
        if isinstance(self.sim, MockSimulation):
            log.info("[MOCK] 跳過模擬狀態初始化。")
            return

        if self.terrain_manager.is_functional:
            # 初始啟動時重置地形管理器，以確保中心點與高度場為最新狀態
            self.terrain_manager.reset()
        self.hard_reset()
        print("\n--- 模擬已啟動 (SPACE: 暫停, N: 單步) ---")

    def start(self) -> None:
        """啟動模擬執行緒。"""
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, name="SimulationThread", daemon=True)
        self.thread.start()

    def run(self) -> None:
        """
        【v4.5.0 重大修改】執行緒主迴圈。
        此迴圈現在是一個精確的、固定頻率的控制迴圈，完全與渲染分離。
        """
        is_headless = isinstance(self.sim, MockSimulation)
        self._initialize_simulation_state()

        last_control_time = time.perf_counter()

        while self._running.is_set():
            current_time = time.perf_counter()
            elapsed = current_time - last_control_time
            
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt - elapsed)
            
            last_control_time = time.perf_counter()

            with self.state.lock:
                # 【v4.5.0 修正】 模式切換請求現在由 ApplicationManager 處理，此處不再需要
                # mode_change_req = self.state.mode_change_request
                # self.state.mode_change_request = None
                
                # 其他請求旗標的處理保持不變
                shutdown_req = self.state.shutdown_requested
                hard_reset_req = self.state.hard_reset_requested
                # 【v4.0.1 修正】正確讀取 soft_reset_requested
                soft_reset_req = self.state.soft_reset_requested
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
                self.stop() # 請求停止自身
                continue
            if hard_reset_req: self.hard_reset()
            # 【v4.0.1 修正】補上對軟重置請求的處理
            if soft_reset_req: self.soft_reset()
            # 【v4.5.0 刪除】 不再直接處理模式切換請求
            # if mode_change_req: self._handle_mode_change(mode_change_req)
            if float_toggle_req is not None: self._handle_float_toggle(float_toggle_req)
            
            # ======================== 主邏輯與模擬步驟 ========================
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
                # 【模擬活動模式】: 執行物理計算，然後更新狀態並渲染完整畫面
                self._simulation_step()
                self._update_derived_and_render_states()

    # 【v4.5.0 新增】 此函式用於響應模式已變更的通知，進行物理清理
    def on_mode_changed(self, old_mode: str, new_mode: str):
        """監聽模式已變更的通知，執行物理相關的狀態清理。"""
        log.debug(f"SimulationController 偵測到模式變更: {old_mode} -> {new_mode}")
        
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
            with self.state.lock:
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

    def on_simulation_reset_requested(self, type: str):
        """【v4.0.1 修復】處理模擬重置請求。只設定請求旗標。"""
        log.debug(f"接收到 '{type}' 重置請求，正在設定旗標。")
        with self.state.lock:
            if type == "hard":
                self.state.hard_reset_requested = True
            elif type == "soft":
                # 【v4.0.1 修正】確保 soft reset 旗標被正確設置為 True
                self.state.soft_reset_requested = True 

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
        log.info(f"SimulationController 偵測到關閉請求，正在停止...")
        self.stop()

    def _update_derived_and_render_states(self) -> None:
        """
        【v4.5.0 新增】 更新所有依賴於核心物理狀態的衍生狀態，並將數據寫入渲染緩衝區。
        """
        current_pos = self.sim.data.body('torso').xpos.copy()
        terrain_mode = self.state.terrain_mode
        if self.terrain_manager.is_functional:
            self.terrain_manager.update(current_pos, terrain_mode)
        
        with self.state.lock:
            self.state.raw_torso_quat = self.sim.data.body('torso').xquat.copy()
            self.state.raw_torso_linear_velocity_world = self.sim.data.cvel[self.sim.torso_id, 3:].copy()
            self.state.raw_torso_angular_velocity_world = self.sim.data.cvel[self.sim.torso_id, :3].copy()
            self.state.raw_joint_positions = self.sim.data.qpos[7:].copy()
            self.state.raw_joint_velocities = self.sim.data.qvel[6:].copy()
            if self.sim.accelerometer_id != -1:
                 start = self.sim.model.sensor_adr[self.sim.accelerometer_id]
                 end = start + self.sim.model.sensor_dim[self.sim.accelerometer_id]
                 self.state.raw_accelerometer = self.sim.data.sensordata[start:end].copy()
            else:
                 self.state.raw_accelerometer.fill(0.0)
            
            self.state.latest_pos = current_pos
            self.state.latest_quat = self.state.raw_torso_quat
            self.state.latest_joint_positions = self.state.raw_joint_positions

        with self.state.render_data_lock:
            if self.state.render_data_buffer is None: self.state.render_data_buffer = {}
            self.state.render_data_buffer['qpos'] = self.sim.data.qpos.copy()

    def _simulation_step(self) -> None:
        """
        【v4.5.0 修改】
        此函式現在除了執行模擬，還負責將原始物理數據寫入 SimulationState。
        物理步進迴圈已被修改為確定性的 for 迴圈。
        """
        # [保留] 讀取狀態和獲取 AI 動作的邏輯不變
        with self.state.lock:
            command = self.state.command.copy()
            control_mode = self.state.control_mode
            tuning_params = self.state.tuning_params

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
            self.state.latest_onnx_input = onnx_input.flatten()
            self.state.latest_action_raw = action_final
            self.state.latest_final_ctrl = final_ctrl

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
            # 【v4.5.0 修正】 模式切換應透過事件，但此處是內部重置，直接設定是可接受的
            if self.state.control_mode == "FLOATING":
                self.state.set_control_mode("WALKING") # 假設 set_control_mode 已被簡化
            
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
        """執行空中姿態重置。"""
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



