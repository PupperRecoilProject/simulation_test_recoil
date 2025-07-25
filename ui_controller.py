"""NiceGUI based control panel."""

from typing import TYPE_CHECKING

from nicegui import ui
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type hints
    from state import SimulationState


class UIController:
    """Encapsulate all NiceGUI layout and callbacks."""

    def __init__(self, state: 'SimulationState') -> None:
        self.state = state
        self.policy_manager = state.policy_manager_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref
        self.xbox_handler = state.xbox_handler_ref

        self.status_labels: dict[str, ui.label] = {}
        self.param_sliders: dict[str, ui.slider] = {}
        self.onnx_input_labels: dict[str, ui.label] = {}
        self.log_area: ui.textarea | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        ui.dark_mode().enable()
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('Pupper 機器人控制台').classes('text-lg')

        with ui.row().classes('w-full no-wrap'):
            with ui.column().classes('w-1/3'):
                self._create_control_panel()
                self._create_tuning_panel()
            with ui.column().classes('w-2/3'):
                self._create_status_display()
                self._create_onnx_display()
                self._create_log_panel()

        ui.timer(0.1, self.update_ui_elements)

    def _create_control_panel(self) -> None:
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
            ui.button('啟用/停用 AI (K)', on_click=self._toggle_hardware_ai).bind_enabled_from(
                self.state, 'control_mode', lambda m: m == "HARDWARE_MODE")

            ui.separator()
            ui.label('設備與重置').classes('text-lg')
            with ui.row():
                ui.button('連接序列埠 (U)', on_click=self._connect_serial)
                ui.button('連接搖桿 (J)', on_click=self._connect_gamepad)
            with ui.row():
                ui.button('軟重置 (X)', on_click=lambda: self.set_request_flag('soft_reset_requested'))
                ui.button('硬重置 (R)', on_click=lambda: self.set_request_flag('hard_reset_requested'))

    def _create_tuning_panel(self) -> None:
        with ui.card().classes('w-full'):
            ui.label('參數調整 (Tuning)').classes('text-lg')
            params = self.state.tuning_params
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            for key, (mn, mx) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    ui.slider(min=mn, max=mx, step=0.01, value=getattr(params, key)).bind_value(params, key).classes('w-48')
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')

            ui.separator()
            ui.label('策略選擇 (Policy)').classes('text-lg')
            self.status_labels['policy_selector'] = ui.select(
                options=self.state.available_policies,
                label='Active Policy',
                value=self.policy_manager.primary_policy_name,
                on_change=lambda e: self.policy_manager.select_target_policy(e.value)
            ).classes('w-full')

    def _create_status_display(self) -> None:
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

    def _create_onnx_display(self) -> None:
        with ui.card().classes('w-full'):
            ui.label('ONNX 觀察向量 (Observation Vector)').classes('text-lg')
            with ui.grid(columns=2):
                obs_components = [
                    'linear_velocity', 'angular_velocity', 'gravity_vector', 'commands',
                    'accelerometer', 'joint_positions', 'joint_velocities', 'last_action'
                ]
                for comp in obs_components:
                    self.onnx_input_labels[comp] = ui.label(f'{comp}: N/A')

    def _create_log_panel(self) -> None:
        with ui.card().classes('w-full'):
            ui.label('日誌輸出 (Log Output)').classes('text-lg')
            # "rows=10" 用於設定顯示高度，取代錯誤的 .lines()
            self.log_area = ui.textarea(label='Log').props('readonly outlined rows=10').style('width: 100%;')

    # ------------------------------------------------------------------
    def update_ui_elements(self) -> None:
        with self.state.lock:
            self.status_labels['mode'].text = f"模式: {self.state.control_mode}"
            self.status_labels['input_mode'].text = f"輸入: {self.state.input_mode}"
            if self.state.sim:
                self.status_labels['sim_time'].text = f"時間: {self.state.sim.data.time:.2f}s"
            self.status_labels['serial_status'].text = (
                '序列埠: Connected' if self.state.serial_is_connected else '序列埠: Disconnected'
            )
            self.status_labels['gamepad_status'].text = (
                '搖桿: Connected' if self.state.gamepad_is_connected else '搖桿: Disconnected'
            )
            if self.state.control_mode == 'HARDWARE_MODE':
                self.status_labels['hardware_ai'].text = '硬體AI: Active' if self.state.hardware_ai_is_active else '硬體AI: Disabled'
            else:
                self.status_labels['hardware_ai'].text = '硬體AI: N/A'

            cmd = self.state.command
            self.status_labels['command'].text = f"vy: {cmd[0]:.2f}, vx: {cmd[1]:.2f}, wz: {cmd[2]:.2f}"

            pos = self.state.latest_pos
            self.status_labels['robot_pos'].text = f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"

            pm = self.policy_manager
            if pm.is_transitioning:
                alpha_percent = pm.transition_alpha * 100
                policy_text = f"策略: Blending {pm.source_policy_name} -> {pm.target_policy_name} ({alpha_percent:.0f}%)"
            else:
                policy_text = f"策略: {pm.primary_policy_name}"
            self.status_labels['policy_status'].text = policy_text
            self.status_labels['policy_selector'].value = pm.primary_policy_name

            self._update_onnx_labels()

            if self.state.control_mode == 'SERIAL_MODE':
                self.log_area.value = "\n".join(self.state.serial_latest_messages)
            else:
                self.log_area.value = self.state.hardware_status_text if self.state.control_mode == 'HARDWARE_MODE' else ''

    def _update_onnx_labels(self) -> None:
        if self.state.latest_onnx_input.size == 0 or not self.policy_manager.get_active_recipe():
            return

        recipe = self.policy_manager.get_active_recipe()
        obs_vec = self.state.latest_onnx_input
        current_idx = 0
        component_dims = self.policy_manager.obs_builder.component_dims

        for comp_name in recipe:
            dim = component_dims.get(comp_name, 0)
            if dim > 0 and comp_name in self.onnx_input_labels:
                end_idx = current_idx + dim
                if end_idx <= len(obs_vec):
                    value_slice = obs_vec[current_idx:end_idx]
                    vec_str = np.array2string(value_slice, precision=2, suppress_small=True, max_line_width=30)
                    self.onnx_input_labels[comp_name].text = f'{comp_name}: {vec_str}'
                current_idx = end_idx

    # ------------------------------------------------------------------
    def _toggle_hardware_ai(self) -> None:
        if self.hardware_controller and self.state.control_mode == 'HARDWARE_MODE':
            if self.state.hardware_ai_is_active:
                self.hardware_controller.disable_ai()
            else:
                self.hardware_controller.enable_ai()

    def set_request_flag(self, flag_name: str) -> None:
        with self.state.lock:
            setattr(self.state, flag_name, True)

    def _connect_serial(self) -> None:
        if self.serial_comm:
            connected = self.serial_comm.scan_and_connect()
            with self.state.lock:
                self.state.serial_is_connected = connected

    def _connect_gamepad(self) -> None:
        if self.xbox_handler:
            connected = self.xbox_handler.scan_and_connect()
            with self.state.lock:
                self.state.gamepad_is_connected = connected

    def run(self) -> None:
        ui.run(title="Pupper Robot Console", port=8080)

