from nicegui import ui
import numpy as np
import threading
from typing import TYPE_CHECKING, List

from logger import log, log_queue

if TYPE_CHECKING:
    from state import SimulationState

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
            with ui.column().classes('w-1/3'):
                self._create_control_panel()
                self._create_tuning_panel()
                self._create_joystick_panel()
                self._create_joint_control_panel()  # 新增關節微調面板
            with ui.column().classes('w-2/3'):
                self._create_status_display()
                self._create_onnx_display()
                self._create_log_panel()

        ui.timer(0.1, self.update_ui_elements)

    def _create_control_panel(self):
        with ui.card():
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                ui.button('走路 (Walking)', on_click=lambda: self.state.set_control_mode("WALKING"))
                ui.button('懸浮 (Floating)', on_click=lambda: self.state.set_control_mode("FLOATING"))
                ui.button('硬體 (Hardware)', on_click=lambda: self.state.set_control_mode("HARDWARE_MODE"))
            with ui.row():
                ui.button('關節測試 (Joint Test)', on_click=lambda: self.state.set_control_mode("JOINT_TEST"))
                ui.button('手動控制 (Manual Ctrl)', on_click=lambda: self.state.set_control_mode("MANUAL_CTRL"))

            ui.separator()
            ui.label('硬體 AI 控制').classes('text-lg')
            ui.button('啟用/停用 AI (K)', on_click=self._toggle_hardware_ai).bind_enabled_from(self.state, 'control_mode', lambda mode: mode == "HARDWARE_MODE")

            ui.separator()
            ui.label('設備與重置').classes('text-lg')
            with ui.row():
                ui.button('連接序列埠 (U)', on_click=self._connect_serial)
                ui.button('連接搖桿 (J)', on_click=self._connect_gamepad)
            with ui.row():
                ui.button('軟重置 (X)', on_click=lambda: self.set_request_flag('soft_reset_requested'))
                ui.button('硬重置 (R)', on_click=lambda: self.set_request_flag('hard_reset_requested'))

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
        with ui.card().bind_visibility_from(self.state, 'control_mode', lambda m: m in ["JOINT_TEST", "MANUAL_CTRL"]).classes('w-full'):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            joint_names = {
                0: 'FR_Abduction', 1: 'FR_Hip', 2: 'FR_Knee', 3: 'FL_Abduction', 4: 'FL_Hip', 5: 'FL_Knee',
                6: 'RR_Abduction', 7: 'RR_Hip', 8: 'RR_Knee', 9: 'RL_Abduction', 10: 'RL_Hip', 11: 'RL_Knee'
            }
            ui.select(joint_names, label='選擇關節', on_change=lambda e: self._set_joint_index(int(e.value)))
            self.status_labels['joint_info'] = ui.label('')
            # 可以精確拖曳的滑桿
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
                self.serial_command_buffer = ui.input(label='Serial Command').props('outlined dense').classes('flex-grow')
                ui.button('Send', on_click=self._send_serial_command)

    def update_ui_elements(self):
        with self.state.lock:
            self.status_labels['mode'].set_text(f"模式: {self.state.control_mode}")
            self.status_labels['input_mode'].set_text(f"輸入: {self.state.input_mode}")
            self.status_labels['sim_time'].set_text(f"時間: {self.state.sim.data.time:.2f}s" if self.state.sim else "時間: N/A")
            self.status_labels['serial_status'].set_text('序列埠: Connected' if self.state.serial_is_connected else '序列埠: Disconnected')
            self.status_labels['gamepad_status'].set_text('搖桿: Connected' if self.state.gamepad_is_connected else '搖桿: Disconnected')
            if self.state.control_mode == 'HARDWARE_MODE':
                self.status_labels['hardware_ai'].set_text('硬體AI: Active' if self.state.hardware_ai_is_active else '硬體AI: Disabled')
            else:
                self.status_labels['hardware_ai'].set_text('硬體AI: N/A')
            cmd = self.state.command
            self.status_labels['command'].set_text(f"vy: {cmd[0]:.2f}, vx: {cmd[1]:.2f}, wz: {cmd[2]:.2f}")
            pos = self.state.latest_pos
            self.status_labels['robot_pos'].set_text(f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            # 若處於關節相關模式，更新面板顯示
            if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                self._update_joint_control_display()
            pm = self.policy_manager
            if pm.is_transitioning:
                alpha_percent = pm.transition_alpha * 100
                policy_text = f"策略: Blending {pm.source_policy_name} -> {pm.target_policy_name} ({alpha_percent:.0f}%)"
            else:
                policy_text = f"策略: {pm.primary_policy_name}"
            self.status_labels['policy_status'].set_text(policy_text)
            self.status_labels['policy_selector'].set_value(pm.primary_policy_name)

            # 從後端狀態更新本地地形選擇值，但不直接 set_value
            current_terrain_name = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state)
            if self.ui_terrain_selection != current_terrain_name:
                self.ui_terrain_selection = current_terrain_name

            self._update_onnx_labels()
        log_content = "\n".join(log_queue)
        self.log_area.set_value(log_content)

    def _update_onnx_labels(self):
        if self.state.latest_onnx_input.size == 0 or not self.policy_manager.get_active_recipe():
            return
        recipe = self.policy_manager.get_active_recipe()
        obs_vec = self.state.latest_onnx_input
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

    def _toggle_hardware_ai(self):
        if self.hardware_controller and self.state.control_mode == 'HARDWARE_MODE':
            if self.state.hardware_ai_is_active:
                self.hardware_controller.disable_ai()
            else:
                self.hardware_controller.enable_ai()

    def set_request_flag(self, flag_name: str):
        with self.state.lock:
            setattr(self.state, flag_name, True)

    def _connect_serial(self):
        if self.serial_comm:
            is_connected = self.serial_comm.scan_and_connect()
            with self.state.lock:
                self.state.serial_is_connected = is_connected

    def _connect_gamepad(self):
        if self.xbox_handler:
            is_connected = self.xbox_handler.scan_and_connect()
            with self.state.lock:
                self.state.gamepad_is_connected = is_connected

    def _set_joint_index(self, index: int):
        """設定目前選中的關節索引。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                self.state.joint_test_index = index
            elif self.state.control_mode == "MANUAL_CTRL":
                self.state.manual_ctrl_index = index

    def _on_joint_slider_change(self, event):
        """滑桿改變時即時更新目標值。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                self.state.joint_test_offsets[self.state.joint_test_index] = event.value
            elif self.state.control_mode == "MANUAL_CTRL":
                self.state.manual_final_ctrl[self.state.manual_ctrl_index] = event.value

    def _adjust_joint_value(self, value: float, clear: bool = False):
        """依目前模式調整關節值或歸零。"""
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                idx = self.state.joint_test_index
                if clear:
                    self.state.joint_test_offsets[idx] = 0.0
                else:
                    self.state.joint_test_offsets[idx] += value
            elif self.state.control_mode == "MANUAL_CTRL":
                idx = self.state.manual_ctrl_index
                if clear:
                    self.state.manual_final_ctrl[idx] = 0.0
                else:
                    self.state.manual_final_ctrl[idx] += value

        # 更新滑桿顯示值
        if self.joint_control_slider is not None:
            with self.state.lock:
                if self.state.control_mode == "JOINT_TEST":
                    self.joint_control_slider.set_value(self.state.joint_test_offsets[self.state.joint_test_index])
                elif self.state.control_mode == "MANUAL_CTRL":
                    self.joint_control_slider.set_value(self.state.manual_final_ctrl[self.state.manual_ctrl_index])

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
        with self.state.lock:
            if self.state.input_mode == "VJOY":
                self.state.clear_command()
        # 切回鍵盤輸入模式，但不要再次清除指令
        self.state.toggle_input_mode("KEYBOARD", clear_cmd=False)

    def _update_joint_control_display(self):
        """根據當前模式更新滑桿與資訊標籤顯示。"""
        if self.joint_control_slider is None:
            return
        with self.state.lock:
            if self.state.control_mode == "JOINT_TEST":
                idx = self.state.joint_test_index
                target = self.state.joint_test_offsets[idx]
                actual = self.state.sim.data.qpos[7 + idx] - self.state.sim.default_pose[idx]
                self.joint_control_slider.set_value(target)
                text = f"模式: 偏移 | 目標: {target:.2f} | 實際: {actual:.2f}"
            elif self.state.control_mode == "MANUAL_CTRL":
                idx = self.state.manual_ctrl_index
                target = self.state.manual_final_ctrl[idx]
                actual = self.state.sim.data.qpos[7 + idx]
                self.joint_control_slider.set_value(target)
                text = f"模式: 絕對 | 目標: {target:.2f} | 實際: {actual:.2f}"
            else:
                text = ""
        self.status_labels['joint_info'].set_text(text)

    def _on_terrain_change(self, event):
        """當地形下拉選單改變時，更新後端狀態並生成新的地形。"""
        terrain_name = event.value
        terrain_manager = self.state.terrain_manager_ref
        with self.state.lock:
            # 若選擇與目前狀態相同，則不進行任何操作
            current_real = terrain_manager.get_current_terrain_name_simple(self.state)
            if terrain_name == current_real:
                return

            if terrain_name == 'INFINITE':
                self.state.terrain_mode = 'INFINITE'
                # 切回無限模式時需要重新產生地形
                terrain_manager.reset()
            else:
                self.state.terrain_mode = 'SINGLE'
                self.state.single_terrain_index = terrain_manager.single_terrain_names.index(terrain_name)
                terrain_manager.set_single_terrain(terrain_name)

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
