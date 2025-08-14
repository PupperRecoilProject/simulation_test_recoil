from nicegui import ui, app
import numpy as np
import threading
from typing import TYPE_CHECKING, List
from src.core.logger import log, log_queue
from src.controllers.hardware_controller import HWState  # 導入硬體狀態枚舉

# [新增] 導入我們新創建的事件系統模組和所有UI會用到的事件名稱
# 解釋:
#   - event_bus: 這是我們發布事件需要用到的全域事件匯流排實例。
#   - EVENT_MODE_CHANGE_REQUESTED: 當用戶點擊模式切換按鈕時，我們會發布這個事件。
#   - EVENT_SIMULATION_RESET_REQUESTED: 用戶點擊重置按鈕時發布。
#   - EVENT_HARDWARE_CONNECT_REQUESTED: 用戶請求連接硬體時發布。
#   - EVENT_HARDWARE_AI_TOGGLE_REQUESTED: 用戶請求啟用/停用硬體AI時發布。
#   - EVENT_SHUTDOWN_REQUESTED: 用戶請求關閉整個應用程式時發布。
#
#   通過從一個統一的地方導入這些"事件契約"，我們可以確保UI和核心邏輯
#   使用的是完全相同的事件名稱，避免因拼寫錯誤導致的通信失敗。
from src.core.event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_TUNING_PARAM_ADJUSTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_COMMAND_UPDATED, # 用於虛擬搖桿
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_SERIAL_COMMAND_SEND,
)

if TYPE_CHECKING:
    from src.core.state import SimulationState



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


    # --- UI 佈局函式 ---
    # 【修改】這些 _create_..._panel 函式的主要變更是它們的 on_click 回呼。
    #         它們不再呼叫 self._request... 這樣的內部函式，而是直接發布事件。

    def _create_main_control_panel(self):
        with ui.card():
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                # 【修改】點擊按鈕時，直接發布一個包含目標模式的'請求'事件。
                ui.button('走路 (Walking)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING"))
                ui.button('懸浮 (Floating)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="FLOATING"))

                # 綁定到 state.serial_is_connected，提供清晰的使用者引導
                ui.button('硬體 (Hardware)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE")) \
                  .bind_enabled_from(self.state, 'serial_is_connected')

            with ui.row():
                ui.button('關節測試 (Joint Test)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="JOINT_TEST"))
                ui.button('手動控制 (Manual Ctrl)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="MANUAL_CTRL"))

            ui.separator()

            ui.label('重置').classes('text-lg')
            with ui.row():
                # 【修改】重置按鈕發布帶有 'type' 參數的事件，以便 SimulationController 區分。
                ui.button('軟重置 (X)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft"))
                ui.button('硬重置 (R)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"))

    def _create_tuning_panel(self):
        with ui.card().classes('w-full'):
            ui.label('參數調整 (Tuning)').classes('text-lg')
            params = self.state.tuning_params
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    # [修改] 滑桿不再直接綁定 state，而是通過 on_change 發布事件
                    slider = ui.slider(min=min_val, max=max_val, step=0.01, value=getattr(params, key),
                                       on_change=lambda e, k=key: event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, param_name=k, value=e.value))
                    self.param_sliders[key] = slider # 保存滑桿參考以便更新
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')

            ui.separator()

        ui.label('策略選擇 (Policy)').classes('text-lg')
        # [修改] on_change 回呼發布策略變更請求事件
        self.status_labels['policy_selector'] = ui.select(
            options=self.state.available_policies,
            label='Active Policy',
            value=self.policy_manager.primary_policy_name,
            on_change=lambda e: event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=e.value)
        ).classes('w-full')

        terrain_options = ['INFINITE'] + self.state.terrain_manager_ref.single_terrain_names
        # [修改] on_change 回呼發布地形變更請求事件
        self.terrain_selector = ui.select(
            options=terrain_options,
            label='Terrain Mode',
            on_change=lambda e: event_bus.publish(EVENT_TERRAIN_CHANGE_REQUESTED, name=e.value)
        ).bind_value(self, 'ui_terrain_selection').classes('w-full')


    # 設備與系統相關控制
    def _create_device_panel(self):
        with ui.card():
            ui.label('硬體 AI 控制').classes('text-lg')
            # 現在綁定到 self.hardware_controller.is_running
            # 只有在硬體控制器成功啟動後，這個按鈕才能被點擊
            ui.button('啟用/停用 AI (K)', on_click=lambda: event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED)) \
              .bind_enabled_from(self.state, 'hardware_is_running')

            ui.separator()
            ui.label('設備連接').classes('text-lg')
            with ui.row():
                # [修改] 連接按鈕發布設備連接請求事件
                ui.button('連接序列埠 (U)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial"))
                ui.button('連接搖桿 (J)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad"))

            ui.separator()
            ui.label('系統').classes('text-lg')
            # [保留] 退出按鈕已經是發布事件，無需修改
            ui.button('退出程式', on_click=lambda: event_bus.publish(EVENT_SHUTDOWN_REQUESTED), color='red')


    def _create_joystick_panel(self):
        with ui.card().classes('w-full'):
            ui.label('手動駕駛 (Manual Driving)').classes('text-lg')
            
            # [修改] on_move 和 on_end 的 lambda 函式現在包含了輸入模式切換的邏輯
            # 並且使用了正確的事件參數 e.x 和 e.y
            ui.joystick(
                color='blue', 
                size=100, 
                on_start=lambda: event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="VJOY"),
                on_move=lambda e: event_bus.publish(
                    EVENT_COMMAND_UPDATED, 
                    command=np.array([
                        e.x * self.state.config.gamepad_sensitivity['vy'], # 使用 e.x
                        -e.y * self.state.config.gamepad_sensitivity['vx'], # 使用 e.y
                        0.0
                    ])
                ),
                on_end=lambda e: (
                    event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(3)),
                    event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="KEYBOARD") # 釋放後切回鍵盤模式
                )
            ).props('throttle')
            
            ui.button('清除命令 (Clear Command)', on_click=lambda: event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(3))).props('outline')


    def _create_joint_control_panel(self):
        # 僅當控制模式為 "JOINT_TEST" 或 "MANUAL_CTRL" 時，此卡片才可見
        with ui.card().bind_visibility_from(self.state, 'control_mode', lambda m: m in ["JOINT_TEST", "MANUAL_CTRL"]):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            
            with ui.row().classes('items-center'):
                ui.label('啟用懸浮')
                
                # 【v4.0 核心修改】將雙向綁定拆為 "on_change" 和 "bind_value_from"
                ui.switch(
                    # 1. 控制流: 當用戶操作開關時，發布一個請求事件
                    on_change=lambda e: event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=e.value)
                ).bind_value_from(
                    # 2. 數據流: 開關的 "開/關" 狀態，單向地從 state.manual_mode_is_floating 讀取
                    self.state, 'manual_mode_is_floating'
                )

            joint_names = {i: name for i, name in enumerate([
                "FR_Abduction", "FR_Hip", "FR_Knee", "FL_Abduction", "FL_Hip", "FL_Knee",
                "RR_Abduction", "RR_Hip", "RR_Knee", "RL_Abduction", "RL_Hip", "RL_Knee"
            ])}
            
            # [修改] on_change 發布關節選擇事件
            self.joint_selector = ui.select(
                joint_names,
                label='選擇關節',
                on_change=lambda e: event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, index=int(e.value))
            )

            self.status_labels['joint_info'] = ui.label('')
            # [修改] 滑桿 on_change 發布關節值調整事件
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01,
                                                 on_change=lambda e: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, value=e.value)
                                                 ).props('label-always')
            with ui.row():
                # [修改] 按鈕發布關節值調整事件
                ui.button('-0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=-0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=0.1)).props('dense')
                ui.button('歸零 (Clear)', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, clear=True)).props('dense')


    def _create_status_display(self):
        with ui.card():
            ui.label('即時狀態 (Real-time Status)').classes('text-lg')
            with ui.grid(columns=3):
                self.status_labels['mode'] = ui.label('模式: WALKING')
                self.status_labels['input_mode'] = ui.label('輸入: KEYBOARD')
                self.status_labels['sim_time'] = ui.label('時間: 0.00s')
                self.status_labels['serial_status'] = ui.label('序列埠: Disconnected')
                # 透過 Gamepad Presence Guard 綁定 UI 顯示
                self.status_labels['gamepad_status'] = ui.label().bind_text_from(
                    self.state,
                    'ui_gamepad_connected',
                    lambda v: '搖桿: Connected' if v else '搖桿: Disconnected',
                )
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
                # 輸入框綁定 Enter 鍵事件，按下 Enter 即送出指令
                self.serial_command_buffer = ui.input(label='Serial Command')\
                    .props('outlined dense').classes('flex-grow')\
                    .on('keydown.enter', self._send_serial_command)
                ui.button('Send', on_click=self._send_serial_command)

    def _send_serial_command(self):
        """
        [v3.0.1] 從 UI 輸入框獲取命令，並發布序列埠命令發送請求事件。
        不再直接調用 serial_comm。
        """
        command_text = self.serial_command_buffer.value
        if command_text:
            event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command=command_text)
            self.serial_command_buffer.set_value('') # 清空輸入框
            log.info(f"> {command_text}")


    def update_ui_elements(self):
        """
        [v3.0.1] 定期從 SimulationState 讀取最新數據，並更新所有UI元件。
        這是一個單向的數據流：State -> UI。
        """
        # ============================ 步驟 1: 在鎖內快速複製所有需要的狀態值 ============================
        with self.state.lock:
            # --- 通用狀態 ---
            mode = self.state.control_mode
            input_mode = self.state.input_mode
            sim_time = self.state.sim.data.time if self.state.sim else None
            serial_connected = self.state.serial_is_connected

            hw_running = self.state.hardware_is_running
            hw_ai_active = self.state.hardware_ai_is_active
            hw_internal = self.hardware_controller.internal_state if self.hardware_controller else None
            
            command = self.state.command.copy()
            pos = self.state.latest_pos.copy()

            # --- AI 策略狀態 ---
            pm = self.policy_manager
            transitioning = pm.is_transitioning
            alpha = pm.transition_alpha
            src_policy = pm.source_policy_name
            tgt_policy = pm.target_policy_name
            primary_policy = pm.primary_policy_name

            # --- 地形狀態 ---
            terrain_name = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state)

            # --- 參數調校狀態 ---
            tuning_params_copy = {
                'kp': self.state.tuning_params.kp,
                'kd': self.state.tuning_params.kd,
                'action_scale': self.state.tuning_params.action_scale,
                'bias': self.state.tuning_params.bias
            }

            # --- 關節控制狀態 ---
            joint_info = None
            if mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                if mode == "JOINT_TEST":
                    idx = self.state.joint_test_index
                    target_abs = self.state.sim.default_pose[idx] + self.state.joint_test_offsets[idx]
                    offset = self.state.joint_test_offsets[idx]
                    joint_info = {"mode": "offset", "index": idx, "target_abs": target_abs, "offset": offset}
                else: # MANUAL_CTRL
                    idx = self.state.manual_ctrl_index
                    target_abs = self.state.manual_final_ctrl[idx]
                    joint_info = {"mode": "absolute", "index": idx, "target_abs": target_abs}
                
                # 共享的資訊
                joint_info["actual_abs"] = self.state.latest_joint_positions[joint_info['index']]

        # 直接從 hardware_controller 讀取其內部狀態來更新 UI
        hw_mode_active = self.state.control_mode == 'HARDWARE_MODE'
        ai_status_text = '硬體AI: N/A'
        if hw_mode_active:
            if hw_running:
                ai_status_text = '硬體AI: Active' if hw_ai_active else '硬體AI: Disabled'
            elif hw_internal == HWState.FAILED:
                ai_status_text = '硬體AI: Failed'
            else:
                ai_status_text = '硬體AI: Starting...'

        self.status_labels['hardware_ai'].set_text(ai_status_text)


        # ============================ 步驟 2: 在鎖外安全地更新所有 UI 元件 ============================

        # --- 更新通用狀態標籤 ---
        self.status_labels['mode'].set_text(f"模式: {mode}")
        self.status_labels['input_mode'].set_text(f"輸入: {input_mode}")
        self.status_labels['sim_time'].set_text(f"時間: {sim_time:.2f}s" if sim_time is not None else "時間: N/A")
        self.status_labels['serial_status'].set_text('序列埠: Connected' if serial_connected else '序列埠: Disconnected')
        # hardware_ai 的文字已在上方統一處理，這裡不再覆寫
        self.status_labels['command'].set_text(f"vy: {command[0]:.2f}, vx: {command[1]:.2f}, wz: {command[2]:.2f}")
        self.status_labels['robot_pos'].set_text(f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

        # --- 更新 AI 策略相關 UI ---
        policy_text = f"策略: Blending {src_policy} -> {tgt_policy} ({alpha*100:.0f}%)" if transitioning else f"策略: {primary_policy}"
        self.status_labels['policy_status'].set_text(policy_text)
        if self.status_labels['policy_selector'].value != primary_policy:
            self.status_labels['policy_selector'].set_value(primary_policy)

        # --- 更新地形選擇 UI ---
        if self.ui_terrain_selection != terrain_name:
            self.ui_terrain_selection = terrain_name

        # --- 更新參數調校滑桿 ---
        for key, slider in self.param_sliders.items():
            state_value = tuning_params_copy[key]
            # 只有在值有顯著差異時才更新，避免浮點數誤差導致的無限循環
            if abs(slider.value - state_value) > 1e-4:
                slider.set_value(state_value)

        # --- 更新關節控制 UI ---
        if joint_info and self.joint_control_slider is not None:
            idx = joint_info['index']
            if self.joint_selector.value != idx:
                self.joint_selector.set_value(idx)

            target_abs = joint_info['target_abs']
            if abs(self.joint_control_slider.value - target_abs) > 1e-4:
                self.joint_control_slider.set_value(target_abs)

            # 更新顯示文字
            actual_abs = joint_info['actual_abs']
            error = target_abs - actual_abs
            if joint_info['mode'] == 'offset':
                text = f"模式: 偏移 | Offset={joint_info['offset']:+.2f} | Target={target_abs:+.2f} | Actual={actual_abs:+.2f} | Err={error:+.2f}"
            else:
                text = f"模式: 絕對 | Target={target_abs:+.2f} | Actual={actual_abs:+.2f} | Err={error:+.2f}"
            self.status_labels['joint_info'].set_text(text)

        # --- 更新 ONNX 觀察向量和日誌 ---
        self._update_onnx_labels()
        log_content = "\n".join(log_queue)
        if self.log_area.value != log_content:
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


    def run(self):
        ui.run(title="Pupper Robot Console", port=8080)
