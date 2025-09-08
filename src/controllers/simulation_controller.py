# src/controllers/simulation_controller.py
"""
【v4.1.0 修改版】模擬與狀態協調控制器
【v4.12.3.1】（特殊版本）— 重構排列與註解強化（功能等價，不改變邏輯）

在獨立執行緒中運行模擬並處理所有狀態變更。
保留歷史版本註記（如 v4.0/v4.12.0/v4.3.1 等），以便追溯。
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from src.core.logger import log

import numpy as np
import random
from src.mock.mock_simulation import MockSimulation

from src.core.event_system import (
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
    # 【v4.0 新增】導入新的通知事件
    EVENT_MODE_CHANGED,
    EVENT_STATE_UPDATED,
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    from src.core.state import SimulationState


class SimulationController:
    """
    在獨立執行緒中運行模擬，並採用解耦的時間控制架構。

    職責：
    1. 協調「邏輯幀」(AI+物理) 和「渲染幀」的執行頻率。
    2. 監聽並處理來自 UI / 輸入層的系統級請求事件。
    3. 管理模擬生命週期（啟動、停止、重置）。
    """

    # =========================================================================
    # I. Public API & Thread Management (公開 API 與執行緒管理)
    # =========================================================================

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        self.sim = state.sim
        self.config = state.config

        # 參考其他核心模組
        self.policy_manager = state.policy_manager_ref
        self.terrain_manager = state.terrain_manager_ref
        self.floating_controller = state.floating_controller_ref
        self.xbox_handler = state.xbox_handler_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref

        # 執行緒控制
        self._running = threading.Event()
        self.thread: threading.Thread | None = None

        # 手動懸浮追蹤
        self._manual_float_active = False

        # 訂閱事件（僅一次，已移除重複）
        self._subscribe_to_events()

        # 初始化將在執行緒啟動後進行

    def start(self) -> None:
        """啟動模擬執行緒。"""
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """停止模擬執行緒（優雅停機）。"""
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    # =========================================================================
    # II. Main Loop & Coordinators (主迴圈與協調器)
    # =========================================================================

    def run(self) -> None:
        """
        【v4.0.2 修改版】執行緒主迴圈。
        【v4.6.0 修改】分離物理更新與數據處理，確保數據流在所有模式下暢通。
        【v4.12.0 重構】執行緒主迴圈，採用解耦的邏輯與渲染迴圈。
        【v4.12.4 重構】採用物理時間累加器架構，實現邏輯與渲染的完全解耦。
        """
        is_headless = isinstance(self.sim, MockSimulation)
        if not is_headless:
            self.sim.initialize_window_and_context()
        self._initialize_simulation_state()
        
        # --- 時間參數設定 ---
        logic_interval = 1.0 / self.config.control_freq
        render_interval = 1.0 / self.config.rendering_frequency if self.config.rendering_frequency > 0 else 0
        physics_timestep = self.config.physics_timestep
        
        # --- 計時器與累加器初始化 ---
        next_logic_update_time = time.perf_counter()
        next_render_update_time = time.perf_counter()
        last_frame_time = time.perf_counter()
        
        physics_accumulator = 0.0
        MAX_FRAME_TIME = 0.25 

        # --- 主迴圈 (以最高頻率運行) ---
        while self._running.is_set():
            # --- 步驟 1: 累積真實時間債務 ---
            current_time = time.perf_counter()
            frame_time = current_time - last_frame_time
            last_frame_time = current_time
            if frame_time > MAX_FRAME_TIME:
                frame_time = MAX_FRAME_TIME
            physics_accumulator += frame_time
            
            # --- 步驟 2: 處理掛起的 UI 請求 ---
            # 這段程式碼負責檢查由事件回呼設定的狀態旗標，並執行對應的操作。
            # 如果沒有這段，所有來自 UI 的請求都將被忽略。
            with self.state.lock:
                shutdown_req = self.state.shutdown_requested
                hard_reset_req = self.state.hard_reset_requested
                soft_reset_req = self.state.soft_reset_requested
                mode_change_req = self.state.mode_change_request
                float_toggle_req = self.state.manual_float_toggle_request

                # 執行後立即清除旗標，避免重複觸發
                self.state.shutdown_requested = False
                self.state.hard_reset_requested = False
                self.state.soft_reset_requested = False
                self.state.mode_change_request = None
                self.state.manual_float_toggle_request = None

            # 在鎖之外，安全地執行請求對應的函式
            if shutdown_req: self._handle_shutdown(); continue
            if hard_reset_req: self.hard_reset()
            if soft_reset_req: self.soft_reset()
            if mode_change_req: self._handle_mode_change(mode_change_req)
            if float_toggle_req is not None: self._handle_float_toggle(float_toggle_req)
            
            # --- 步驟 3: 償還物理時間債務 (物理+AI迴圈) ---
            is_sim_active = not is_headless and self.state.control_mode not in ["HARDWARE_MODE", "SERIAL_MODE"]
            if is_sim_active:
                while physics_accumulator >= physics_timestep:
                    # 在物理步進前，檢查是否到了 AI 決策的時刻
                    # 注意：這裡的 next_logic_update_time 是基於真實時鐘的
                    if current_time >= next_logic_update_time:
                        self._perform_ai_decision()
                        next_logic_update_time += logic_interval
                    
                    # 執行一次單步物理模擬
                    self._single_physics_step()
                    
                    # 償還一筆債務
                    physics_accumulator -= physics_timestep
            else: # 如果模擬不活躍，則重置累加器，避免時間債務無限累積
                physics_accumulator = 0.0

            # --- 步驟 4: 渲染迴圈 ---
            if not is_headless and current_time >= next_render_update_time:
                alpha = physics_accumulator / physics_timestep if physics_timestep > 0 else 1.0
                self._render_frame(alpha)
                
                if render_interval > 0:
                    next_render_update_time += render_interval
                else:
                    next_render_update_time = current_time

            # --- 步驟 5: 視窗事件與休眠 ---
            if not is_headless:
                self.sim.poll_window_events()
            time.sleep(0.001)

        log.info("模擬執行緒已優雅地停止。")

    # 舊的 _perform_logic_frame 將被新的 run() 迴圈邏輯取代
    # def _perform_logic_frame(self, action_from_previous_frame: np.ndarray):

    # =========================================================================
    # III. Core Logic Execution (核心邏輯執行)
    # =========================================================================

    def _update_recoil_warning_timer(self) -> None:
        """
        更新 Firearm Recoil Warning 計時器邏輯。
        - auto_inhibit = True 時完全中斷自動預警
        - 保留計時器循環節奏，避免破壞依賴 recoil_timer 的其他流程
        """
        frw_cfg = getattr(self.config, 'firearm_recoil_warming', None)
        if isinstance(frw_cfg, dict):
            auto_inhibit = frw_cfg.get('auto_inhibit', False)
        elif frw_cfg is not None:
            auto_inhibit = bool(getattr(frw_cfg, 'auto_inhibit', False))
        else:
            auto_inhibit = False

        WARNING_DURATION_S = 0.15
        MIN_INTERVAL_S = 2.5
        MAX_INTERVAL_S = 10.0

        with self.state.lock:
            self.state.recoil_timer -= self.config.control_dt

            if auto_inhibit:
                self.state.recoil_warning_active = False
                if self.state.recoil_timer <= 0:
                    self.state.recoil_interval = random.uniform(MIN_INTERVAL_S, MAX_INTERVAL_S)
                    self.state.recoil_timer = self.state.recoil_interval
                return

            if self.state.recoil_timer <= WARNING_DURATION_S:
                self.state.recoil_warning_active = True

            if self.state.recoil_timer <= 0:
                self.state.recoil_warning_active = False
                self.state.recoil_interval = random.uniform(MIN_INTERVAL_S, MAX_INTERVAL_S)
                self.state.recoil_timer = self.state.recoil_interval
                log.info(f"*** RECOIL EVENT *** Next in {self.state.recoil_interval:.2f}s")

    def _perform_ai_decision(self):
        """
        【v4.12.4 新增】只負責 AI 決策，並將結果存入 state。
        
        此函式封裝了 AI 決策的原子操作。它從 state 讀取感知信息，
        調用 PolicyManager，並將決策結果（action_raw, final_ctrl）寫回 state。
        """
        # 這段程式碼是從舊的 _perform_logic_frame 中抽離出來的
        with self.state.lock:
            command = self.state.command.copy()
            # 獲取 last_action 需要在鎖外，因為 get_action 可能耗時
        
        # 由於 get_action 內部有自己的鎖，這裡可以在鎖外安全調用
        onnx_input, action_final = self.policy_manager.get_action(command)
        
        with self.state.lock:
            control_mode = self.state.control_mode
            tuning_params = self.state.tuning_params
            if control_mode == "MANUAL_CTRL":
                final_ctrl = self.state.manual_final_ctrl.copy()
            elif control_mode == "JOINT_TEST":
                final_ctrl = self.sim.default_pose + self.state.joint_test_offsets
            else:
                final_ctrl = self.sim.default_pose + action_final * tuning_params.action_scale
            
            self.state.latest_onnx_input = onnx_input.flatten()
            self.state.latest_action_raw = action_final
            self.state.latest_final_ctrl = final_ctrl
            # 【重要】更新 last_action 的職責現在也歸於此處
            self.state.raw_last_action = action_final.copy()

    # 舊的 _simulation_step 將被 _single_physics_step 取代
    # def _simulation_step(self) -> None:
    
    def _single_physics_step(self):
        """
        【v4.12.4 新增】執行一次單步的物理模擬 + 狀態更新。
        """
        # 1. 應用控制指令
        with self.state.lock:
            final_ctrl = self.state.latest_final_ctrl.copy()
            tuning_params = self.state.tuning_params
        self.sim.apply_position_control(final_ctrl, tuning_params)
        
        # 2. 執行單步物理演進
        mujoco.mj_step(self.sim.model, self.sim.data)

        
        # 3. 將物理結果寫回 state.raw_...
        with self.state.lock:
            # ...[此處是將 sim.data 寫入 state.raw_... 的完整程式碼，與 v4.12.0 相同]...
            self.state.raw_torso_quat = self.sim.data.body('torso').xquat.copy()
            self.state.raw_torso_linear_velocity = self.sim.data.cvel[self.sim.torso_id, 3:].copy()
            self.state.raw_torso_angular_velocity = self.sim.data.cvel[self.sim.torso_id, :3].copy()
            self.state.raw_joint_positions = self.sim.data.qpos[7:].copy()
            self.state.raw_joint_velocities = self.sim.data.qvel[6:].copy()
            if self.sim.accelerometer_id != -1:
                 start = self.sim.model.sensor_adr[self.sim.accelerometer_id]
                 end = start + self.sim.model.sensor_dim[self.sim.accelerometer_id]
                 self.state.raw_accelerometer = self.sim.data.sensordata[start:end].copy()
            else:
                 self.state.raw_accelerometer.fill(0.0)
            
        # 4. 更新標準化觀測值
        if self.state.observation_manager_ref:
            self.state.observation_manager_ref.update_all_observations()

    def _render_frame(self, alpha: float):
        """
        【v4.12.4 新增】執行一次渲染。alpha 用於未來可能的插值渲染。
        """
        # 目前，我們暫不實現插值渲染，直接渲染最新狀態
        if self.state.control_mode == "HARDWARE_MODE":
            self._update_simulation_from_hardware_state()
        self.update_derived_states_and_render()

    def _update_simulation_from_hardware_state(self):
        """
        在硬體模式下，將從實體機器人收到的狀態同步到 MuJoCo 模擬器以進行視覺化。
        """
        if isinstance(self.sim, MockSimulation) or not self.state.hardware_is_running:
            return

        with self.state.lock:
            hw_joint_positions = self.state.raw_joint_positions.copy()

        self.sim.data.qpos[7:] = hw_joint_positions
        mujoco.mj_forward(self.sim.model, self.sim.data)

    # =========================================================================
    # IV. State Management & Resets (狀態管理與重置)
    # =========================================================================

    def _initialize_simulation_state(self) -> None:
        if isinstance(self.sim, MockSimulation):
            log.info("[MOCK] Skip simulation state initialization.")
            return

        if self.terrain_manager.is_functional:
            self.terrain_manager.reset()
        self.hard_reset()
        print("\n--- Simulation Started (SPACE: Pause, N: Step) ---")

    def _handle_shutdown(self):
        """處理關閉請求的邏輯。"""
        log.info("偵測到關閉請求，正在停止主迴圈...")
        self._running.clear()
        from nicegui import app
        app.shutdown()

    def _handle_mode_change(self, new_mode: str):
        """
        【v4.0 新增】安全地處理模式切換。
        此函式在主迴圈的安全上下文中被呼叫，可以安全地修改物理狀態。
        """
        with self.state.lock:
            old_mode = self.state.control_mode
            if old_mode == new_mode:
                return

            if new_mode == "HARDWARE_MODE":
                log.info(f"模式切換: 發出硬體啟動請求...")
                self.hardware_controller.request_start()
            elif old_mode == "HARDWARE_MODE":
                log.info(f"模式切換: 發出硬體停止請求...")
                self.hardware_controller.request_stop()

            self.state.set_control_mode(new_mode)

        self._handle_mode_change_physics(old_mode, new_mode)

        if new_mode in ["HARDWARE_MODE", "SERIAL_MODE"]:
            if not isinstance(self.sim, MockSimulation):
                log.info(f"渲染 '{new_mode}' 的凍結畫面...")
                self.sim.render_from_thread(self.state)

        event_bus.publish(EVENT_MODE_CHANGED, old_mode=old_mode, new_mode=new_mode)
        log.info(f"✅ 模式已成功從 '{old_mode}' 切換至 '{new_mode}'。")

    def _handle_mode_change_physics(self, old_mode: str, new_mode: str):
        """處理模式切換中涉及物理修改的部分。"""
        if old_mode == "FLOATING":
            self.floating_controller.disable()
            self._manual_float_active = False

        if new_mode == "FLOATING":
            current_pos = self.sim.data.body('torso').xpos.copy()
            self.floating_controller.enable(current_pos)
            self._manual_float_active = True

        if new_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
            log.info(f"進入 {new_mode}，重置關節姿態與速度。")
            self.sim.data.qpos[7:] = self.sim.default_pose.copy()
            self.sim.data.qvel[6:] = 0
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def _handle_float_toggle(self, is_floating: bool):
        """安全地處理手動懸浮請求。"""
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

        with self.state.lock:
            self.state.manual_mode_is_floating = self._manual_float_active
        log.info(f"手動懸浮物理狀態已切換為: {self._manual_float_active}")

    def update_derived_states_and_render(self) -> None:
        """
        【v4.0】更新所有依賴於核心物理狀態的衍生狀態（如地形），並渲染場景。
        """
        is_headless = isinstance(self.sim, MockSimulation)

        with self.state.lock:
            current_pos = self.state.latest_pos.copy()
            terrain_mode = self.state.terrain_mode

            if not is_headless:
                self.state.latest_pos = self.sim.data.body('torso').xpos.copy()
                self.state.latest_quat = self.sim.data.body('torso').xquat.copy()
                self.state.latest_joint_positions = self.sim.data.qpos[7:].copy()

        if self.terrain_manager.is_functional:
            self.terrain_manager.update(current_pos, terrain_mode)

        self.sim.render_from_thread(self.state)

    def hard_reset(self) -> None:
        """根據目前地形自動決定適當高度並執行硬重置。"""
        with self.state.lock:
            terrain_name = self.terrain_manager.get_current_terrain_name_simple(self.state)

        difficult = ["Pyramid", "Stepped Pyramid"]
        start_z_offset = 1.5 if terrain_name in difficult else 0.3

        print(f"\n--- 正在執行機器人硬重置 (地形: {terrain_name}, 高度偏移: {start_z_offset}m) ---")

        with self.state.lock:
            if self.state.control_mode == "HARDWARE_MODE":
                return

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

            # 重置座力計時器狀態
            self.state.recoil_interval = random.uniform(2.5, 10.0)
            self.state.recoil_timer = self.state.recoil_interval
            self.state.recoil_warning_active = False

            self.state.hard_reset_requested = False
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def soft_reset(self) -> None:
        """【v4.0.1 確認】此函式邏輯是正確的，無需修改。"""
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

    # =========================================================================
    # V. Event Callback Handlers (事件回呼處理器)
    # =========================================================================

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

        log.info("SimulationController 已訂閱所有核心請求事件。")

    def on_mode_change_requested(self, mode: str):
        """處理模式切換請求。只設定請求旗標。"""
        log.debug(f"接收到模式切換請求 -> {mode}，正在設定請求旗標。")
        with self.state.lock:
            self.state.mode_change_request = mode

    def on_simulation_reset_requested(self, type: str):
        """處理模擬重置請求。只設定請求旗標。"""
        log.debug(f"接收到 '{type}' 重置請求，正在設定旗標。")
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
        處理調整參數值的請求。
        """
        with self.state.lock:
            if param_name is None:
                param_keys = ['kp', 'kd', 'action_scale', 'bias']
                if 0 <= self.state.tuning_param_index < len(param_keys):
                    param_name = param_keys[self.state.tuning_param_index]
                else:
                    log.error(f"無效的調校參數索引: {self.state.tuning_param_index}")
                    return

            if value is not None:
                setattr(self.state.tuning_params, param_name, value)
            elif direction is not None:
                step = self.config.param_adjust_steps.get(param_name, 0.1)
                current_value = getattr(self.state.tuning_params, param_name)
                new_value = current_value + step * direction
                setattr(self.state.tuning_params, param_name, new_value)
            else:
                log.warning(f"接收到無效的參數調整請求: param_name={param_name}, value={value}, direction={direction}")
                return

            self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
            self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
            self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)
            log.info(f"參數 '{param_name}' 已調整為: {getattr(self.state.tuning_params, param_name):.2f}")

    def on_input_mode_change_requested(self, mode: str):
        """處理輸入模式切換請求。"""
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
        """處理序列埠命令發送請求。"""
        if self.serial_comm and self.serial_comm.is_connected:
            try:
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
        處理地形切換請求。
        """
        if not self.terrain_manager or not self.terrain_manager.is_functional:
            return

        log.info(f"接收到切換地形請求: {name}")

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

        if needs_reset:
            log.info("地形已變更，請求硬重置以應用...")
            with self.state.lock:
                self.state.hard_reset_requested = True

    def on_manual_float_toggled(self, is_floating: bool):
        """處理手動模式下的懸浮開關請求。只設定請求旗標。"""
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
                elif value is not None:
                    self.state.joint_test_offsets[idx] = value - self.state.sim.default_pose[idx]
                elif direction is not None:
                    self.state.joint_test_offsets[idx] += direction

            elif self.state.control_mode == "MANUAL_CTRL":
                idx = self.state.manual_ctrl_index
                if clear:
                    self.state.manual_final_ctrl[idx] = self.sim.default_pose[idx]
                elif value is not None:
                    self.state.manual_final_ctrl[idx] = value
                elif direction is not None:
                    self.state.manual_final_ctrl[idx] += direction

    def on_shutdown_requested(self):
        """處理全域關閉請求（僅設定旗標）。"""
        log.info("接收到全域關閉請求，正在設定旗標...")
        with self.state.lock:
            self.state.shutdown_requested = True

    # =========================================================================
    # VI. Utility Methods (輔助工具函式)
    # =========================================================================
    # 目前無其他輔助函式，保留為未來擴充用。