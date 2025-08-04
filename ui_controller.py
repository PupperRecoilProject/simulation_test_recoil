from nicegui import ui
import numpy as np
import time

from utils.logger import log, log_queue

from state import SimulationState, OperatingMode, ControlSubMode

class UIController:
    """管理 NiceGUI 介面與互動邏輯。"""
    def __init__(self, state: 'SimulationState'):
        self.state = state
        self.policy_manager = state.policy_manager_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref
        self.xbox_handler = state.xbox_handler_ref

        self.status_labels = {}
        self.param_sliders = {}
        self.onnx_input_labels = {}
        self.log_area = None
        self.serial_command_buffer = None
        # 關節控制滑桿 (僅在關節測試與手動控制模式下啟用)
        self.joint_control_slider = None

        # 儲存 UI 下拉選單的地形選擇值，避免與後端狀態互相觸發
        if self.state.terrain_mode == 'SINGLE':
            self.ui_terrain_selection = self.state.terrain_manager_ref.single_terrain_names[self.state.single_terrain_index]
        else:
            self.ui_terrain_selection = 'INFINITE'

        self._setup_ui()

    def _setup_ui(self):
        ui.dark_mode().enable()
        with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
            ui.label('Pupper 機器人控制台').classes('text-lg')

        with ui.row().classes('w-full no-wrap'):
            # 左側欄：控制相關的面板
            with ui.column().classes('w-1/3'):
                # 使用垂直 Tabs 組織面板
                with ui.tabs().props('vertical').classes('w-full') as tabs:
                    # 名稱作為第一參數，label 是顯示文字，避免重複傳 name 造成錯誤
                    ui.tab('control', label='控制')
                    ui.tab('tuning', label='參數')
                    ui.tab('hardware', label='硬體')

                with ui.tab_panels(tabs, value='control').props('vertical').classes('w-full'):
                    with ui.tab_panel('control'):
                        ui.label('主控制項').classes('text-lg font-bold mb-4')
                        self._create_main_control_panel()
                    with ui.tab_panel('tuning'):
                        ui.label('AI 與物理').classes('text-lg font-bold mb-4')
                        self._create_tuning_panel()
                    with ui.tab_panel('hardware'):
                        ui.label('設備連接').classes('text-lg font-bold mb-4')
                        self._create_device_panel()

                # 關節微調與搖桿控制仍在主頁面下方
                self._create_joint_control_panel()
                self._create_joystick_panel()

            # 右側欄：狀態與日誌顯示
            with ui.column().classes('w-2/3'):
                self._create_status_display()
                self._create_onnx_display()
                self._create_log_panel()

        ui.timer(0.1, self.update_ui_elements)

    # 主要控制面板：模式切換與播放控制
    def _create_main_control_panel(self):
        with ui.card():
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                # 使用新的子模式切換函式
                ui.button('走路 (Walking)', on_click=lambda: self.state.request_sub_mode_change(ControlSubMode.WALKING))
                ui.button('懸浮 (Floating)', on_click=lambda: self.state.request_sub_mode_change(ControlSubMode.FLOATING))
                ui.button('硬體 (Hardware)', on_click=self._toggle_operating_mode)
            with ui.row():
                ui.button('關節測試 (Joint Test)', on_click=lambda: self.state.request_sub_mode_change(ControlSubMode.JOINT_TEST))
                ui.button('手動控制 (Manual Ctrl)', on_click=lambda: self.state.request_sub_mode_change(ControlSubMode.MANUAL_CTRL))

            ui.separator()
            ui.label('模擬播放 (Playback)').classes('text-lg')
            with ui.row():
                # 建立「暫停/播放」按鈕，文字會依狀態變化
                ui.button(on_click=self._toggle_pause) \
                    .bind_text_from(self.state, 'single_step_mode',
                                    lambda is_paused: '播放 (Play)' if is_paused else '暫停 (Pause)')

                # 建立「下一步」按鈕，只在暫停時啟用
                ui.button('下一步 (Next Step)', on_click=self._request_one_step) \
                    .bind_enabled_from(self.state, 'single_step_mode')

            ui.separator()
            ui.label('重置').classes('text-lg')
            with ui.row():
                ui.button('軟重置 (X)', on_click=lambda: self._request_flag_change('soft_reset_requested'))
                ui.button('硬重置 (R)', on_click=lambda: self._request_flag_change('hard_reset_requested'))

    def _create_tuning_panel(self):
        with ui.card().classes('w-full'):
            ui.label('參數調整 (Tuning)').classes('text-lg')
            params = self.state.tuning_params
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    slider = ui.slider(min=min_val, max=max_val, step=0.01, value=getattr(params, key)).bind_value(params, key).classes('w-48')
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')

            ui.separator()
        ui.label('策略選擇 (Policy)').classes('text-lg')
        self.status_labels['policy_selector'] = ui.select(
            options=self.state.available_policies,
            label='Active Policy',
            value=self.policy_manager.primary_policy_name,
            on_change=lambda e: self.policy_manager.select_target_policy(e.value)
        ).classes('w-full')

        # 【核心修正】地形選擇下拉選單，綁定到本地狀態以避免循環觸發
        terrain_options = ['INFINITE'] + self.state.terrain_manager_ref.single_terrain_names
        self.terrain_selector = ui.select(
            options=terrain_options,
            label='Terrain Mode',
            on_change=self._on_terrain_change
        ).bind_value(self, 'ui_terrain_selection').classes('w-full')

    # 設備與系統相關控制
    def _create_device_panel(self):
        with ui.card():
            ui.label('硬體 AI 控制').classes('text-lg')
            ui.button('啟用/停用 AI (K)', on_click=self._toggle_hardware_ai).bind_enabled_from(
                self.state, 'operating_mode', lambda m: m == OperatingMode.HARDWARE)

            ui.separator()
            ui.label('設備連接').classes('text-lg')
            with ui.row():
                ui.button('連接序列埠 (U)', on_click=self._connect_serial)
                ui.button('連接搖桿 (J)', on_click=self._connect_gamepad)

            ui.separator()
            ui.label('系統').classes('text-lg')
            ui.button('退出程式', on_click=self._request_shutdown, color='red')

    def _create_joystick_panel(self):
        with ui.card().classes('w-full'):
            ui.label('手動駕駛 (Manual Driving)').classes('text-lg')
            ui.joystick(
                color='blue', 
                size=100, 
                on_move=self._update_command_from_joystick,  # 直接傳遞回呼函式
                on_end=self._on_joystick_end
            ).props('throttle')
            ui.button('清除命令 (Clear Command)', on_click=self.state.clear_command).props('outline')

    def _create_joint_control_panel(self):
        """在 JOINT_TEST 或 MANUAL_CTRL 模式下顯示的關節微調面板。"""
        with ui.card().bind_visibility_from(self.state, 'control_sub_mode', lambda m: m in [ControlSubMode.JOINT_TEST, ControlSubMode.MANUAL_CTRL]).classes('w-full'):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            # 懸浮開關，適用於手動相關模式
            with ui.row().classes('items-center'):
                ui.label('啟用懸浮')
                ui.switch(on_change=self._on_manual_float_toggle).bind_value(self.state, 'manual_mode_is_floating')
            joint_names = {
                0: 'FR_Abduction', 1: 'FR_Hip', 2: 'FR_Knee', 3: 'FL_Abduction', 4: 'FL_Hip', 5: 'FL_Knee',
                6: 'RR_Abduction', 7: 'RR_Hip', 8: 'RR_Knee', 9: 'RL_Abduction', 10: 'RL_Hip', 11: 'RL_Knee'
            }
            # 保存選擇框以便之後更新數值
            self.joint_selector = ui.select(
                joint_names,
                label='選擇關節',
                on_change=lambda e: self._set_joint_index(int(e.value))
            )

            self.status_labels['joint_info'] = ui.label('')
            # 滑桿在使用者拖動時會觸發回呼，其值將在 update_ui_elements 中同步
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01, on_change=self._on_joint_slider_change).props('label-always')
            with ui.row():
                ui.button('-0.1', on_click=lambda: self._adjust_joint_value(-0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: self._adjust_joint_value(0.1)).props('dense')
                ui.button('歸零 (Clear)', on_click=lambda: self._adjust_joint_value(0, clear=True)).props('dense')

    def _create_status_display(self):
        with ui.card():
            ui.label('即時狀態 (Real-time Status)').classes('text-lg')
            with ui.grid(columns=3):
                self.status_labels['mode'] = ui.label('模式: WALKING')
                self.status_labels['input_mode'] = ui.label('輸入: KEYBOARD')
                self.status_labels['sim_time'] = ui.label('時間: 0.00s')
                self.status_labels['serial_status'] = ui.label('序列埠: Disconnected')
                self.status_labels['gamepad_status'] = ui.label('搖桿: Disconnected')
                self.status_labels['hardware_ai'] = ui.label('硬體AI: N/A')
                self.status_labels['policy_status'] = ui.label(f'策略: {self.policy_manager.primary_policy_name}')
            ui.separator()
            ui.label('運動指令 (Command)').classes('font-bold')
            self.status_labels['command'] = ui.label('vy: 0.00, vx: 0.00, wz: 0.00')
            ui.label('機器人狀態 (Robot State)').classes('font-bold')
            self.status_labels['robot_pos'] = ui.label('位置: [0.0, 0.0, 0.0]')
            # 顯示硬體延遲與 CRC 錯誤數
            self.status_badge = ui.badge('Delay -- | CRC 0')
            self.status_labels['robot_vel'] = ui.label('速度: [0.0, 0.0, 0.0]')

    def _create_onnx_display(self):
        """建立 ONNX 觀察向量區域，並設定最小高度避免畫面跳動。"""
        # 【修正】設定卡片的最小高度，避免文字長度變化造成版面跳動
        with ui.card().style('min-height: 220px;'):
            ui.label('ONNX 觀察向量 (Observation Vector)').classes('text-lg')
            with ui.grid(columns=2):
                obs_components = [
                    'linear_velocity', 'angular_velocity', 'gravity_vector', 'commands',
                    'accelerometer', 'joint_positions', 'joint_velocities', 'last_action'
                ]
                for comp in obs_components:
                    self.onnx_input_labels[comp] = ui.label(f'{comp}: N/A')

    def _create_log_panel(self):
        with ui.card().classes('w-full'):
            ui.label('系統日誌與序列埠控制台').classes('text-lg')
            self.log_area = ui.textarea(label='Log').props('readonly outlined rows=10').style('width: 100%;')
            with ui.row().classes('w-full items-center'):
                # 輸入框綁定 Enter 鍵事件，按下 Enter 即送出指令
                self.serial_command_buffer = ui.input(label='Serial Command')\
                    .props('outlined dense').classes('flex-grow')\
                    .on('keydown.enter', self._send_serial_command)
                ui.button('Send', on_click=self._send_serial_command)

    def update_ui_elements(self):
        """更新所有 UI 元件，先鎖定狀態取得資料，再在鎖外更新。"""
        # --- 在鎖內快速複製所有需要的狀態值 ---
        with self.state.lock:
            op_mode = self.state.operating_mode
            sub_mode = self.state.control_sub_mode
            input_mode = self.state.input_mode
            sim_time = self.state.sim.data.time if self.state.sim else None

            # 使用實際物件狀態，避免資訊延遲
            serial_connected = self.serial_comm.is_connected if self.serial_comm else False
            gamepad_connected = (
                self.state.xbox_handler_ref.controller.is_connected()
                if self.state.xbox_handler_ref else False
            )

            hw_ai_active = self.state.hardware.ai_is_active
            command = self.state.command.copy()

            # 根據操作模式決定位置資料來源
            if op_mode == OperatingMode.SIMULATION:
                pos = self.state.sim_latest_pos.copy()
            else:  # HARDWARE 模式下無全域位置資訊
                pos = np.zeros(3)

            pm = self.policy_manager
            transitioning = pm.is_transitioning
            alpha = pm.transition_alpha
            src_policy = pm.source_policy_name
            tgt_policy = pm.target_policy_name
            primary_policy = pm.primary_policy_name

            terrain_name = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state)

            joint_info = None
            # 只有在關節測試或手動模式下才需要關節資訊
            if sub_mode in [ControlSubMode.JOINT_TEST, ControlSubMode.MANUAL_CTRL]:
                if op_mode == OperatingMode.SIMULATION:
                    actual_joint_positions = self.state.sim_latest_joint_positions
                else:
                    actual_joint_positions = self.state.hardware.joint_positions_rad

                if sub_mode == ControlSubMode.JOINT_TEST:
                    idx = self.state.joint_test_index
                    offset = self.state.joint_test_offsets[idx]
                    default_pos = self.state.sim.default_pose[idx]
                    target_abs = default_pos + offset
                    actual_abs = actual_joint_positions[idx]
                    joint_info = {
                        "mode": "offset",
                        "index": idx,
                        "target_abs": target_abs,
                        "actual_abs": actual_abs,
                        "offset": offset,
                    }
                elif sub_mode == ControlSubMode.MANUAL_CTRL:
                    idx = self.state.manual_ctrl_index
                    target_abs = self.state.manual_final_ctrl[idx]
                    actual_abs = actual_joint_positions[idx]
                    joint_info = {
                        "mode": "absolute",
                        "index": idx,
                        "target_abs": target_abs,
                        "actual_abs": actual_abs,
                    }

        # --- 在鎖外更新 UI 元件 ---
        self.status_labels['mode'].set_text(f"模式: {op_mode.name} / {sub_mode.name}")
        self.status_labels['input_mode'].set_text(f"輸入: {input_mode}")
        if sim_time is not None:
            self.status_labels['sim_time'].set_text(f"時間: {sim_time:.2f}s")
        else:
            self.status_labels['sim_time'].set_text("時間: N/A")

        self.status_labels['serial_status'].set_text(
            '序列埠: Connected' if serial_connected else '序列埠: Disconnected'
        )
        self.status_labels['gamepad_status'].set_text(
            '搖桿: Connected' if gamepad_connected else '搖桿: Disconnected'
        )

        if op_mode == OperatingMode.HARDWARE:
            self.status_labels['hardware_ai'].set_text(
                '硬體AI: Active' if hw_ai_active else '硬體AI: Disabled'
            )
        else:
            self.status_labels['hardware_ai'].set_text('硬體AI: N/A')

        self.status_labels['command'].set_text(
            f"vy: {command[0]:.2f}, vx: {command[1]:.2f}, wz: {command[2]:.2f}"
        )
        self.status_labels['robot_pos'].set_text(
            f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        )

        if transitioning:
            policy_text = (
                f"策略: Blending {src_policy} -> {tgt_policy} ({alpha*100:.0f}%)"
            )
        else:
            policy_text = f"策略: {primary_policy}"
        self.status_labels['policy_status'].set_text(policy_text)
        if self.status_labels['policy_selector'].value != primary_policy:
            self.status_labels['policy_selector'].set_value(primary_policy)

        if self.ui_terrain_selection != terrain_name:
            self.terrain_selector.set_value(terrain_name)

        # --- 關節控制資訊的安全更新 ---
        if joint_info:
            idx = joint_info['index']
            if self.joint_selector.value != idx:
                self.joint_selector.set_value(idx)

            target_abs = joint_info['target_abs']
            actual_abs = joint_info['actual_abs']

            if (
                self.joint_control_slider is not None
                and self.joint_control_slider.value is not None
                and abs(self.joint_control_slider.value - target_abs) > 1e-3
            ):
                self.joint_control_slider.set_value(target_abs)

            error = target_abs - actual_abs
            if joint_info['mode'] == 'offset':
                offset = joint_info['offset']
                text = (
                    f"模式: 偏移 | Offset={offset:+.2f} | Target={target_abs:+.2f} | "
                    f"Actual={actual_abs:+.2f} | Err={error:+.2f}"
                )
            else:
                text = (
                    f"模式: 絕對 | Target={target_abs:+.2f} | Actual={actual_abs:+.2f} | "
                    f"Err={error:+.2f}"
                )
            self.status_labels['joint_info'].set_text(text)
        else:
            # 非相關模式清空標籤內容
            self.status_labels['joint_info'].set_text('')

        # 更新 ONNX 標籤與日誌
        self._update_onnx_labels()
        log_content = "\n".join(log_queue)
        self.log_area.set_value(log_content)

        # 顯示 CRC 與延遲資訊
        if self.state.operating_mode == OperatingMode.HARDWARE:
            delay = time.time() - self.state.hardware.last_update_time
            crc_err = self.state.hardware.crc_error_count
            self.status_badge.set_text(f"Delay {delay:.2f}s | CRC {crc_err}")
        else:
            self.status_badge.set_text('')

    def _update_onnx_labels(self):
        """依據目前模式更新 ONNX 輸入顯示"""
        if self.state.operating_mode == OperatingMode.SIMULATION:
            obs_vec = self.state.sim_latest_onnx_input
        else:
            obs_vec = self.state.hardware.latest_onnx_input
        if obs_vec.size == 0 or not self.policy_manager.get_active_recipe():
            return
        recipe = self.policy_manager.get_active_recipe()
        current_idx = 0
        # 從已註冊的 policy_manager 取得各觀察元件的維度
        component_dims = self.policy_manager.obs_builder.component_dims
        for comp_name in recipe:
            dim = component_dims.get(comp_name, 0)
            if dim > 0 and comp_name in self.onnx_input_labels:
                end_idx = current_idx + dim
                if end_idx <= len(obs_vec):
                    value_slice = obs_vec[current_idx:end_idx]
                    vec_str = np.array2string(value_slice, precision=2, suppress_small=True, max_line_width=30)
                    self.onnx_input_labels[comp_name].set_text(f'{comp_name}: {vec_str}')
                current_idx = end_idx

    # 【新增】「暫停/播放」按鈕的回呼函式
    def _toggle_pause(self):
        """切換模擬的暫停/播放狀態。"""
        with self.state.lock:
            self.state.single_step_mode = not self.state.single_step_mode
        status = 'PAUSED' if self.state.single_step_mode else 'PLAYING'
        log.info(f"--- SIMULATION {status} (toggled from UI) ---")

    # 【新增】「下一步」按鈕的回呼函式
    def _request_one_step(self):
        """請求在暫停模式下執行單步模擬。"""
        with self.state.lock:
            if self.state.single_step_mode:
                self.state.execute_one_step = True

    def _toggle_operating_mode(self) -> None:
        """在模擬與硬體模式間切換"""
        with self.state.lock:
            current_op = self.state.operating_mode

        if current_op == OperatingMode.SIMULATION:
            self.state.request_mode_change(OperatingMode.HARDWARE, ControlSubMode.IDLE)
        else:
            self.state.request_mode_change(OperatingMode.SIMULATION, ControlSubMode.WALKING)

    def _toggle_hardware_ai(self):
        """切換硬體端 AI 的啟用狀態"""
        if self.state.operating_mode != OperatingMode.HARDWARE:
            return
        if self.state.hardware.ai_is_active:
            self.state.request_sub_mode_change(ControlSubMode.IDLE)
        else:
            self.state.request_sub_mode_change(ControlSubMode.WALKING)

    def _request_flag_change(self, flag_name: str):
        """非阻塞地請求一次性操作，如重置。"""
        with self.state.lock:
            setattr(self.state, flag_name, True)
        log.info(f"UI請求旗標設定: {flag_name}")

    def _connect_serial(self):
        if self.serial_comm:
            is_connected = self.serial_comm.scan_and_connect()
            # 只更新連線狀態，暫不將控制權交給硬體控制器
            # keep serial port managed by SerialCommunicator until HW mode
            with self.state.lock:
                self.state.serial_is_connected = is_connected

    def _connect_gamepad(self):
        if self.xbox_handler:
            is_connected = self.xbox_handler.scan_and_connect()
            with self.state.lock:
                self.state.gamepad_is_connected = is_connected

    def _request_shutdown(self) -> None:
        """請求關閉程式，由模擬執行緒處理"""
        log.info("UI 請求關閉程式")
        with self.state.lock:
            self.state.shutdown_requested = True

    def _set_joint_index(self, index: int):
        """設定目前選中的關節索引。"""
        with self.state.lock:
            if self.state.control_sub_mode == ControlSubMode.JOINT_TEST:
                self.state.joint_test_index = index
            elif self.state.control_sub_mode == ControlSubMode.MANUAL_CTRL:
                self.state.manual_ctrl_index = index

    def _on_joint_slider_change(self, event):
        """滑桿改變時即時更新目標值。"""
        value = event.value
        with self.state.lock:
            if self.state.control_sub_mode == ControlSubMode.JOINT_TEST:
                idx = self.state.joint_test_index
                # 滑桿給的是絕對角度，轉成偏移量存回 state
                self.state.joint_test_offsets[idx] = value - self.state.sim.default_pose[idx]
            elif self.state.control_sub_mode == ControlSubMode.MANUAL_CTRL:
                idx = self.state.manual_ctrl_index
                self.state.manual_final_ctrl[idx] = value

    def _adjust_joint_value(self, value: float, clear: bool = False):
        """依目前模式調整關節值或歸零。"""
        with self.state.lock:
            if self.state.control_sub_mode == ControlSubMode.JOINT_TEST:
                idx = self.state.joint_test_index
                if clear:
                    self.state.joint_test_offsets[idx] = 0.0
                else:
                    self.state.joint_test_offsets[idx] += value
            elif self.state.control_sub_mode == ControlSubMode.MANUAL_CTRL:
                idx = self.state.manual_ctrl_index
                if clear:
                    self.state.manual_final_ctrl[idx] = 0.0
                else:
                    self.state.manual_final_ctrl[idx] += value

        # 滑桿的實際更新由 update_ui_elements 進行，避免鎖重複取得

    def _on_manual_float_toggle(self, event) -> None:
        """手動模式懸浮開關變化時呼叫，僅更新狀態由模擬執行緒處理"""
        is_floating = bool(event.value)
        with self.state.lock:
            self.state.manual_mode_is_floating = is_floating
        log.info(f"手動懸浮狀態切換為: {is_floating}")

    def _update_command_from_joystick(self, event):
        """虛擬搖桿移動時的回呼函式，根據 x、y 更新指令。"""
        x_val = -event.y #/ 50.0  # y 值對應機器人前後速度，方向相反
        y_val = event.x #/ 50.0   # x 值對應左右橫移速度
        # 切換到虛擬搖桿輸入模式，但保留當前指令
        self.state.toggle_input_mode("VJOY", clear_cmd=False)
        with self.state.lock:
            self.state.command[0] = y_val * self.state.config.gamepad_sensitivity['vy']
            self.state.command[1] = x_val * self.state.config.gamepad_sensitivity['vx']
            self.state.command[2] = 0.0

    def _on_joystick_end(self, event):
        """虛擬搖桿釋放時的回呼函式。"""
        self.state.clear_command()
        self.state.toggle_input_mode("KEYBOARD")


    def _on_terrain_change(self, event):
        """當地形下拉選單改變時，更新後端狀態並生成新的地形。"""
        # 有些情況(如初始化)會傳入 None，此時直接忽略
        if event.value is None:
            return

        terrain_name = event.value
        terrain_manager = self.state.terrain_manager_ref

        # 若 terrain_manager 在無頭模式下可能為 mock，需先檢查屬性
        if not hasattr(terrain_manager, 'single_terrain_names'):
            return

        with self.state.lock:
            # 若選擇與目前狀態相同，則不進行任何操作
            current_real = terrain_manager.get_current_terrain_name_simple(self.state)
            if terrain_name == current_real:
                return

            if terrain_name == 'INFINITE':
                self.state.terrain_mode = 'INFINITE'
                if terrain_manager.is_functional:
                    terrain_manager.reset()
            else:
                self.state.terrain_mode = 'SINGLE'
                if terrain_name in terrain_manager.single_terrain_names:
                    self.state.single_terrain_index = terrain_manager.single_terrain_names.index(terrain_name)
                if terrain_manager.is_functional:
                    terrain_manager.set_single_terrain(terrain_name)

            if terrain_manager.is_functional:
                # 請求硬重置以適應新的地形
                self.state.hard_reset_requested = True

    def _send_serial_command(self):
        command_text = self.serial_command_buffer.value
        if self.serial_comm and command_text:
            self.serial_comm.send_command(command_text)
            self.serial_command_buffer.set_value('')
            log.info(f"> {command_text}")

    def run(self):
        ui.run(title="Pupper Robot Console", port=8080)
