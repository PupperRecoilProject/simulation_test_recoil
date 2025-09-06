# src/controllers/ui_controller.py

from nicegui import ui, app
import numpy as np
import threading
from typing import TYPE_CHECKING, List, Dict  # 【修改】從 typing 導入 Dict
from src.controllers.global_keyboard_driver import GlobalKeyboardDriver  # 【新增】全域鍵盤駕駛

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
    EVENT_COMMAND_UPDATED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_SERIAL_COMMAND_SEND,
    # ↓↓↓ 這四行是新增的 ↓↓↓
    EVENT_FRW_AUTO_INHIBIT_SET,
    EVENT_FRW_AUTO_INHIBIT_CLEAR,
    EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED,
    EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED,
)
from src.core.logger import log, log_queue
# 【v4.10.1 新增】導入 State 和 Enum 以進行類型提示
from src.core.state import SimulationState, HardwareLinkStatus

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

        # 【v4.10.1 新增】為靜默開關宣告一個屬性
        # 【v4.10.5 修改】將 mute_switch 重命名為 enable_motors_switch
        self.enable_motors_switch = None
        self.param_sliders: Dict[str, ui.slider] = {}
        self.onnx_input_labels: Dict[str, ui.element] = {}
        self.log_area = None
        self.serial_command_buffer = None
        # 關節控制滑桿 (僅在關節測試與手動控制模式下啟用)
        self.joint_control_slider = None
        # 【整合 NanoOwl】為 NanoOwl 的 prompt 輸入框創建一個屬性
        self.prompt_input = None

        # 儲存 UI 下拉選單的地形選擇值，避免與後端狀態互相觸發
        if self.state.terrain_mode == 'SINGLE':
            self.ui_terrain_selection = self.state.terrain_manager_ref.single_terrain_names[self.state.single_terrain_index]
        else:
            self.ui_terrain_selection = 'INFINITE'

        # ===================================================================
        # === 【v4.7.0 新增】UI 元素定義區 (Single Source of Truth) ===
        # ===================================================================
        # 存放最終創建的 ui.label 物件
        self.status_labels: Dict[str, ui.label] = {}

        # 【v4.7.0 新增】狀態標籤描述字典
        # 這是 UI 狀態顯示的唯一真相來源。它定義了每個標籤的:
        # - title: 在 UI 上顯示的靜態標題。
        # - getter: 一個從 state 快照中提取對應數據的函式。
        # - formatter: 一個將提取出的數據轉換為顯示字串的函式。
        # 【手冊實作】移除 action_raw 和 final_ctrl，它們將由新的儀表板處理
        self._label_descriptors = {
            'mode': {
                'title': '模式',
                'getter': lambda s: s.control_mode,
                'formatter': str
            },
            'input_mode': {
                'title': '輸入',
                'getter': lambda s: s.input_mode,
                'formatter': str
            },
            'sim_time': {
                'title': '時間',
                'getter': lambda s: s.sim.data.time if s.sim and hasattr(s.sim, 'data') else 0.0,
                'formatter': lambda v: f"{v:.2f}s"
            },
            'serial_status': {
                'title': '序列埠',
                'getter': lambda s: s.serial_is_connected,
                'formatter': lambda v: 'Connected' if v else 'Disconnected'
            },
            'gamepad_status': {
                'title': '搖桿',
                'getter': lambda s: s.gamepad_is_connected,
                'formatter': lambda v: 'Connected' if v else 'Disconnected'
            },
            'hardware_ai': {
                'title': '硬體AI',
                'getter': lambda s: (s.control_mode, s.hardware_is_running, s.hardware_ai_is_active),
                'formatter': lambda v: 'Active' if v[2] else 'Standby' if v[1] else 'Starting...' if v[0] == 'HARDWARE_MODE' else 'N/A'
            },
            'policy_status': {
                'title': '策略',
                'getter': lambda s: (s.policy_manager_ref.is_transitioning, s.policy_manager_ref.source_policy_name, s.policy_manager_ref.target_policy_name, s.policy_manager_ref.transition_alpha, s.policy_manager_ref.primary_policy_name),
                'formatter': lambda v: f"Blending {v[1]}->{v[2]} ({v[3]*100:.0f}%)" if v[0] else v[4]
            },
            'command': {
                'title': '運動指令',
                'getter': lambda s: s.command,
                'formatter': lambda v: f"vy: {v[0]:.2f}, vx: {v[1]:.2f}, wz: {v[2]:.2f}, pitch: {v[3]:.2f}"  # 【修改】顯示4D指令
            },
            'robot_pos': {
                'title': '位置',
                'getter': lambda s: s.latest_pos,
                'formatter': lambda v: f"[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]"
            },
            # 【v4.11.0-w 新增】新增硬體頻率的 UI 描述符
            'hw_io_freq': {
                'title': '硬體 I/O 頻率',
                'getter': lambda s: s.hw_data_freq if s.control_mode == 'HARDWARE_MODE' else -1.0,
                'formatter': lambda v: f"{v:.1f} Hz" if v >= 0 else "N/A"
            },
            'hw_ai_freq': {
                'title': '硬體 AI 頻率',
                'getter': lambda s: s.hw_ai_freq if s.control_mode == 'HARDWARE_MODE' else -1.0,
                'formatter': lambda v: f"{v:.1f} Hz" if v >= 0 else "N/A"
            },
        }
        
        # 【新增】掛上全域鍵盤駕駛（無修飾鍵＋忽略輸入焦點＋超時關閉）
        self.global_kbd = GlobalKeyboardDriver(self.state)  # 功能：啟用 WASD、A/D 旋轉、Q/E 平移、I/K 俯仰 與 FRW/Reset/F

        # 這兩行沒有必要，會在初始化階段無意義來回切換，已移除
        # self.global_kbd.on_input_mode_change('VJOY')
        # self.global_kbd.on_input_mode_change('KEYBOARD')

        self._setup_ui()

    def _format_12d_vector_for_ui(self, vector: np.ndarray) -> str:
        if not isinstance(vector, np.ndarray) or vector.shape != (12,):
            return "`無效數據`"
        lines = []
        legs = ['FR', 'FL', 'RR', 'RL']
        for i in range(4):
            leg_data = vector[i * 3: i * 3 + 3]
            formatted_numbers = [f"{x: 7.3f}" for x in leg_data]
            lines.append(f"  {legs[i]}: [ {' '.join(formatted_numbers)} ]")
        content = "\n".join(lines)
        return f"```\n{content}\n```"

    def _setup_ui(self):
        ui.add_head_html('''
            <style>
                .nicegui-card { margin: 4px !important; padding: 8px !important; box-shadow: none !important; border: 1px solid #333; }
                .q-btn-group .q-btn { margin: 0 !important; }
                .nicegui-markdown pre { margin: 0 !important; padding: 4px !important; background-color: #222 !important; border-radius: 4px; }
                .nicegui-markdown code { font-size: 0.8rem !important; }
                .q-expansion-item__container .q-item { padding: 0 8px !important; min-height: 40px !important; }

                /* 通用滾動條樣式 - 適用於 column 等直接滾動的容器 */
                .custom-scrollbar::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                /* 【手冊實作 v1.19 - 修正】簡化滑塊樣式，確保可見性 */
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: #888;
                    border-radius: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background-color: #AAA;
                }

                /* 【手冊實作 v1.19 - 修正】為 textarea 內部滾動條設定相同的簡化樣式 */
                .custom-scrollbar .q-field__native::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                .custom-scrollbar .q-field__native::-webkit-scrollbar-track {
                    background: transparent;
                }
                .custom-scrollbar .q-field__native::-webkit-scrollbar-thumb {
                    background-color: #888;
                    border-radius: 4px;
                }
                .custom-scrollbar .q-field__native::-webkit-scrollbar-thumb:hover {
                    background-color: #AAA;
                }
            </style>
        ''')
        ui.dark_mode().enable()
        with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
            ui.label('Pupper 機器人控制台').classes('text-lg')

        with ui.row().classes('w-full no-wrap').style('height: calc(100vh - 100px);'):
            # 【手冊實作 v1.18】為左欄添加 'custom-scrollbar' class
            with ui.column().classes('w-1/3 custom-scrollbar').style('height: 100%; overflow:hidden auto; min-height: 0; min-width: 0;'):
                ui.label('主控制項').classes('text-xl font-bold mt-2 mb-1')
                self._create_main_control_panel()
                ui.label('策略與地形').classes('text-xl font-bold mt-2 mb-1')
                self._create_policy_and_terrain_selectors()
                self._create_recoil_warning_panel()
                self._create_joystick_panel()
                self._create_joint_control_panel()
                self._create_device_panel()
                # 【v4.11.3-w 移除】刪除左側欄的 Teensy 指令面板呼叫
                ui.label('參數微調 (Tuning)').classes('text-xl font-bold mt-2 mb-1')
                self._create_tuning_sliders()
            # 【手冊實作 v1.18】為右欄添加 'custom-scrollbar' class
            with ui.column().classes('grow custom-scrollbar').style('height: 100%; overflow:hidden auto; min-height: 0; min-width: 0;'):
                self._create_status_display()
                self._create_core_dashboard()
                self._create_onnx_display()
                self._create_vision_panel()  # 【整合】新增視覺面板
                # 【v4.11.3-w 修改】將舊的 _create_log_panel() 呼叫替換為新的 _create_interactive_console()
                self._create_interactive_console()

        ui.timer(0.1, self.update_ui_elements)

    def _create_main_control_panel(self):
        with ui.card().classes('w-full'):
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                ui.button('走路 (Walking)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING"))
                ui.button('懸浮 (Floating)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="FLOATING"))
                ui.button('硬體 (Hardware)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE")).bind_enabled_from(self.state, 'serial_is_connected')
            with ui.row():
                ui.button('關節測試 (Joint Test)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="JOINT_TEST"))
                ui.button('手動控制 (Manual Ctrl)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="MANUAL_CTRL"))
            ui.separator().classes('my-2')
            ui.label('模擬控制').classes('text-lg')
            with ui.row():
                ui.button().bind_text_from(self.state, 'single_step_mode', lambda is_paused: '▶️ 播放 (SPACE)' if is_paused else '⏸️ 暫停 (SPACE)').on('click', self._toggle_pause)
                ui.button('⏭️ 步進 (N)', on_click=lambda: setattr(self.state, 'execute_one_step', True)).bind_enabled_from(self.state, 'single_step_mode')
            ui.separator().classes('my-2')
            ui.label('重置').classes('text-lg')
            with ui.row():
                ui.button('軟重置 (X)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft"))
                ui.button('硬重置 (R)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"))

    # 【v4.11.6-w 關鍵修正】
    # 新增這個被遺漏的方法
    def _toggle_pause(self):
        """安全地切換模擬的暫停/播放狀態。"""
        # 使用 state.lock 確保線程安全
        with self.state.lock:
            self.state.single_step_mode = not self.state.single_step_mode
            # 當從暫停模式恢復時，確保 execute_one_step 被重置
            if not self.state.single_step_mode:
                self.state.execute_one_step = False
        log.info(f"模擬暫停狀態已切換為: {self.state.single_step_mode}")

    def _create_policy_and_terrain_selectors(self):
        with ui.card().classes('w-full'):
            ui.label('策略選擇 (Policy)').classes('text-lg')
            self.status_labels['policy_selector'] = ui.select(
                options=self.state.available_policies,
                label='Active Policy',
                value=self.policy_manager.primary_policy_name,
                on_change=lambda e: event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=e.value)
            ).classes('w-full')
            ui.label('地形選擇 (Terrain)').classes('text-lg mt-2')
            terrain_options = ['INFINITE'] + self.state.terrain_manager_ref.single_terrain_names
            self.terrain_selector = ui.select(
                options=terrain_options,
                label='Terrain Mode',
                on_change=lambda e: event_bus.publish(EVENT_TERRAIN_CHANGE_REQUESTED, name=e.value)
            ).bind_value(self, 'ui_terrain_selection').classes('w-full')

    def _create_tuning_sliders(self):
        with ui.card().classes('w-full'):
            params = self.state.tuning_params
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    slider = ui.slider(
                        min=min_val, max=max_val, step=0.01, value=getattr(params, key),
                        on_change=lambda e, k=key: event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, param_name=k, value=e.value)
                    )
                    self.param_sliders[key] = slider
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')

    def _create_device_panel(self):
        with ui.card().classes('w-full'):
            ui.label('硬體 AI 控制').classes('text-lg')

            # 【v4.10.5 修改】全面更新 UI 標籤和邏輯
            with ui.row().classes('items-center'):
                ui.label('啟用馬達 (Enable Motors)').classes('text-sm')
                # 1. 重新命名 self.mute_switch -> self.enable_motors_switch
                # 2. 更新 on_change 的處理函式
                # 3. 更新 tooltip 的說明文字
                self.enable_motors_switch = ui.switch(on_change=self._handle_enable_motors_switch_change) \
                    .tooltip('手動啟用或禁用 AI 對馬達的控制。僅在硬體通訊驗證成功後可用。')

            # 現在綁定到 self.hardware_controller.is_running
            # 只有在硬體控制器成功啟動後，這個按鈕才能被點擊
            ui.button('啟用或停用 AI (K)', on_click=lambda: event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED)) \
                .bind_enabled_from(self.state, 'hardware_is_running')

            ui.separator().classes('my-2')
            ui.label('設備連接').classes('text-lg')
            with ui.row():
                ui.button('連接序列埠 (U)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial"))
                ui.button('連接搖桿 (J)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad"))
            ui.separator().classes('my-2')
            ui.label('系統').classes('text-lg')
            ui.button('退出程式', on_click=lambda: event_bus.publish(EVENT_SHUTDOWN_REQUESTED), color='red')

    def _create_joystick_panel(self):
        with ui.card().classes('w-full'):
            ui.label('手動駕駛 (Manual Driving)').classes('text-lg')
            with ui.row().classes('w-full items-center justify-center'):
                ui.joystick(
                    color='blue',
                    size=100,
                    # 同時呼叫 publish 與 global_kbd，同步輸入來源狀態
                    on_start=lambda: (
                        event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="VJOY"),
                        self.global_kbd.on_input_mode_change('VJOY')
                    ),
                    on_move=lambda e: event_bus.publish(
                        EVENT_COMMAND_UPDATED,
                        command=np.array([e.x * self.state.config.gamepad_sensitivity['vy'],
                                          -e.y * self.state.config.gamepad_sensitivity['vx'],
                                          0.0, 0.0])
                    ),
                    on_end=lambda e: (
                        event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(4)),
                        event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="KEYBOARD"),
                        self.global_kbd.on_input_mode_change('KEYBOARD')
                    )
                ).props('throttle')
            ui.button('清除命令 (Clear Command)', on_click=lambda: event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(4))).props('outline').classes('w-full mt-2')

    def _create_joint_control_panel(self):
        with ui.card().bind_visibility_from(self.state, 'control_mode', lambda m: m in ["JOINT_TEST", "MANUAL_CTRL"]).classes('w-full'):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            with ui.row().classes('items-center'):
                ui.label('啟用懸浮')
                ui.switch(on_change=lambda e: event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=e.value)).bind_value_from(self.state, 'manual_mode_is_floating')
            joint_names = {i: name for i, name in enumerate(["FR_Abduction", "FR_Hip", "FR_Knee", "FL_Abduction", "FL_Hip", "FL_Knee", "RR_Abduction", "RR_Hip", "RR_Knee", "RL_Abduction", "RL_Hip", "RL_Knee"])}
            self.joint_selector = ui.select(joint_names, label='選擇關節', on_change=lambda e: event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, index=int(e.value)))
            self.status_labels['joint_info'] = ui.label('')
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01, value=0.0, on_change=lambda e: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, value=e.value)).props('label-always')
            with ui.row():
                ui.button('-0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=-0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=0.1)).props('dense')
                ui.button('歸零 (Clear)', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, clear=True)).props('dense')

    def _create_vector_grid_display(self, title: str, key_prefix: str, show_row_labels: bool = True):
        with ui.column().classes('gap-0'):
            ui.label(title).classes('text-lg font-bold')
            with ui.grid(columns=4).classes('w-full'):
                if not show_row_labels:
                    ui.element('div')
                else:
                    ui.label('')
                ui.label('X/Abd').classes('text-sm text-gray-400 font-mono text-center')
                ui.label('Y/Hip').classes('text-sm text-gray-400 font-mono text-center')
                ui.label('Z/Knee').classes('text-sm text-gray-400 font-mono text-center')
                for i, leg in enumerate(['FR', 'FL', 'RR', 'RL']):
                    if show_row_labels:
                        ui.label(leg).classes('text-base font-bold')
                    else:
                        ui.element('div')
                    for j in range(3):
                        label_key = f"{key_prefix}_{i}_{j}"
                        self.status_labels[label_key] = ui.label('0.000').classes('font-mono text-right w-full')

    def _create_status_display(self):
        with ui.card().classes('w-full'):
            ui.label('即時狀態 (Real-time Status)').classes('text-xl font-bold')
            # 【v4.11.0-w 修改】增加網格的列數以容納新的顯示項目
            with ui.grid(columns=4):  # 從 3 改為 4
                for key, desc in self._label_descriptors.items():
                    # 這裡的邏輯不需要改變，它會自動創建新的標籤
                    self.status_labels[key] = ui.label(f"{desc['title']}: N/A")

    def _create_core_dashboard(self):
        """【手冊實作 v1.16】儀表板現在只包含 AI 的直接輸出。"""
        with ui.expansion('核心數據儀表板 (Core Dashboard)', icon='insights').classes('w-full').props('value=true'):
            with ui.card().classes('w-full'):
                with ui.row().classes('w-full no-wrap'):
                    with ui.element('div').classes('w-1/2'):
                        self._create_vector_grid_display('原始動作 (Raw Action)', 'action_raw', show_row_labels=True)
                    with ui.element('div').classes('w-1/2'):
                        self._create_vector_grid_display('最終控制 (Final Ctrl)', 'final_ctrl', show_row_labels=False)

    # 在 UIController 類別中，找到這個函式並用下面的版本替換它
    def _create_vision_panel(self):
        """【手冊實作 v1.20】將功能正常的視覺面板整合進可折疊佈局中。"""
        # 1. 最外層使用 ui.expansion 來實現折疊功能
        with ui.expansion('即時視覺 (NanoOwl)', icon='visibility').classes('w-full').props('value=true'):
            # 2. 在 expansion 內部使用 ui.card 來保持統一的邊框和背景風格
            with ui.card().classes('w-full'):
                # 3. 這裡的內容完全複製自您提供的「沒問題的版本」
                video_server_url = "http://localhost:8081"
                ui.html(f'''
                    <iframe src="{video_server_url}"
                            width="100%" height="480" frameborder="0" scrolling="no" tabindex="-1"
                            style="max-width: 640px; aspect-ratio: 640 / 480; display: block; margin: auto; pointer-events: none;">
                    </iframe>
                ''')
                with ui.row().classes('w-full items-center'):
                    # 【注意】這裡我們不再需要 self.prompt_input，因為 prompt_input 的作用域只在這個函式內部
                    prompt_input = ui.input(placeholder='輸入識別提示, 例如: [a person]').props('outlined dense').classes('flex-grow')

                    def handle_send_event():
                        prompt_value = prompt_input.value

                        if not prompt_value:
                            ui.notify('請先輸入識別提示', color='warning')
                            return

                        import json
                        escaped_prompt = json.dumps(prompt_value)

                        js_script = f'''
                            const prompt = {escaped_prompt};
                            console.log(`接收到來自 Python 的 Prompt: "${{prompt}}"`);

                            const ws_name = 'nanoowl_ws_connection';

                            if (!window[ws_name] || window[ws_name].readyState > 1) {{
                                console.log("WebSocket 未連接，正在創建...");
                                window[ws_name] = new WebSocket('ws://localhost:8081/ws');

                                window[ws_name].onopen = () => {{
                                    console.log('%c[WebSocket] 連接成功，現在發送消息', 'color: green;');
                                    window[ws_name].send(prompt);
                                }};
                                window[ws_name].onerror = (err) => console.error("[WebSocket] 連接錯誤:", err);
                                window[ws_name].onclose = () => console.warn("[WebSocket] 連接已關閉");
                            }} else {{
                                console.log("WebSocket 已連接，直接發送消息");
                                window[ws_name].send(prompt);
                            }}
                        '''

                        ui.run_javascript(js_script)
                        prompt_input.set_value(None)

                    ui.button('辨識', on_click=handle_send_event)
                    prompt_input.on('keydown.enter', handle_send_event)

    def _create_onnx_display(self):
        """【手冊實作 v1.16】此面板現在包含所有作為 AI 輸入的觀測數據。"""
        with ui.expansion('ONNX 觀察向量 (Observation Vector)', icon='schema').classes('w-full').props('value=true'):
            with ui.card().classes('w-full'):
                all_components = sorted(self.state.observation_manager_ref.ALL_OBS_DIMS.items())

                # --- 創建 12 維長向量的並排儀表板 ---
                vectors_to_display_12d = ['joint_positions', 'joint_velocities', 'last_action']
                ui.label("12-D Vectors").classes('text-lg font-bold')
                with ui.row().classes('w-full no-wrap'):
                    for idx, comp_name in enumerate(vectors_to_display_12d):
                        if self.state.observation_manager_ref.ALL_OBS_DIMS.get(comp_name) == 12:
                            with ui.element('div').classes('w-1/3'):
                                is_first_in_row = (idx == 0)
                                self._create_vector_grid_display(
                                    comp_name.replace('_', ' ').title(),
                                    f"onnx_{comp_name}",
                                    show_row_labels=is_first_in_row
                                )

                # --- 創建短向量的顯示區域 ---
                ui.separator().classes('my-2')
                ui.label("Short Vectors").classes('text-lg font-bold')
                with ui.grid(columns=3):
                    for comp_name, dim in all_components:
                        if dim != 12:
                            with ui.column().classes('gap-0'):
                                ui.label(comp_name).classes('text-xs font-bold text-gray-400')
                                self.onnx_input_labels[comp_name] = ui.markdown('`N/A`').classes('text-sm')

    # 【v4.11.3-w 刪除】
    # def _create_teensy_command_panel(self):
    #     ...

    # 【v4.11.3-w 刪除】
    # def _create_log_panel(self):
    #     ...

    def _create_interactive_console(self):
        """
        【v4.11.5-w+expansion】互動式控制台改為可收合面板。
        修正點：
        1) 外層改為 ui.expansion，與 Core Dashboard / ONNX / NanoOwl 一致
        2) 仍維持只對實際可互動元件做 bind_enabled_from 的安全綁定
        """
        with ui.expansion('互動式控制台 (Interactive Console)', icon='terminal').classes('w-full').props('value=true'):
            with ui.card().classes('w-full'):
                # 標題放在卡片內，視覺與其他面板一致
                ui.label('互動式控制台 (Interactive Console)').classes('text-xl font-bold')

                # --- 日誌顯示區域 (非互動) ---
                self.log_area = ui.textarea(label='系統日誌 / 硬體回饋') \
                                   .props('readonly outlined rows=10') \
                                   .style('width: 100%') \
                                   .classes('custom-scrollbar')

                # --- 收集所有需要被統一控制狀態的互動元件 ---
                interactive_elements = []

                # --- 指令輸入與發送 ---
                with ui.row().classes('w-full items-center'):
                    self.serial_command_buffer = ui.input(placeholder='手動輸入指令') \
                        .props('outlined dense') \
                        .classes('flex-grow') \
                        .on('keydown.enter', self._send_serial_command)
                    send_button = ui.button('發送', on_click=self._send_serial_command)

                interactive_elements.append(self.serial_command_buffer)
                interactive_elements.append(send_button)

                ui.separator().classes('my-2')
                ui.label('快捷指令').classes('text-lg')

                # --- 快捷指令按鈕 ---
                with ui.row().classes('items-center'):
                    with ui.button_group().props('outline'):
                        b1 = ui.button('校準 (cal)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='cal')).tooltip('執行馬達校準程序')
                        b2 = ui.button('停止 (stop)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='stop')).tooltip('立即停止所有馬達運動')
                        b3 = ui.button('站立 (stand)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='stand')).tooltip('讓機器人進入站立姿態')
                        b4 = ui.button('歸零 (zero)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='zero')).tooltip('將所有馬達移動到零位')
                        interactive_elements.extend([b1, b2, b3, b4])

                with ui.row().classes('items-center mt-2'):
                    with ui.button_group().props('outline'):
                        b5 = ui.button('人類模式 (h)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='monitor h')).tooltip('切換到人類可讀的遙測模式')
                        b6 = ui.button('AI 模式 (p)', on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command='monitor p')).tooltip('切換到高頻 AI 數據流模式')
                        interactive_elements.extend([b5, b6])

                    freq_options = [1, 2, 10, 50]
                    freq_select = ui.select(freq_options, value=50) \
                                     .props('outlined dense options-dense') \
                                     .style('width: 80px; margin-left: 20px')
                    freq_button = ui.button(
                        '設定頻率',
                        on_click=lambda: event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command=f'monitor freq {freq_select.value}')
                    ).props('outline')

                    interactive_elements.append(freq_select)
                    interactive_elements.append(freq_button)

                # 僅對互動元件做狀態綁定，避免 ButtonGroup 綁定錯誤
                for element in interactive_elements:
                    element.bind_enabled_from(self.state, 'serial_is_connected')
                    element.bind_enabled_from(self.state, 'control_mode', lambda mode: mode != 'HARDWARE_MODE')

    def _create_recoil_warning_panel(self):
        with ui.card().classes('w-full'):
            ui.label('開火預警控制 (Firearm Recoil Warning)').classes('text-lg font-bold')

            # 讀取 config 的初始值，兼容 dict 或 dataclass，預設 False
            frw_cfg = getattr(self.state.config, 'firearm_recoil_warming', {})
            if isinstance(frw_cfg, dict):
                initial_inhibit = bool(frw_cfg.get('auto_inhibit', False))
            else:
                initial_inhibit = bool(getattr(frw_cfg, 'auto_inhibit', False))

            # 建立唯一的 switch 並以 config 值初始化
            inhibit_switch = ui.switch('抑制自動預警', value=initial_inhibit)

            # 正確事件：on_value_change 會提供 e.value；同時備援讀取 widget.value
            def on_toggle_inhibit(e):
                new_val = bool(getattr(e, 'value', inhibit_switch.value))
                event_bus.publish(EVENT_FRW_AUTO_INHIBIT_SET if new_val else EVENT_FRW_AUTO_INHIBIT_CLEAR)

            # 只綁這一個，避免重複廣播或拿到 GenericEventArguments
            inhibit_switch.on_value_change(on_toggle_inhibit)

            # 啟動時依 config 值同步一次事件，確保後端狀態一致
            event_bus.publish(EVENT_FRW_AUTO_INHIBIT_SET if initial_inhibit else EVENT_FRW_AUTO_INHIBIT_CLEAR)

            with ui.row():
                ui.button('手動觸發', on_click=lambda: event_bus.publish(
                    EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED
                ))
                ui.button('手動重置', on_click=lambda: event_bus.publish(
                    EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED
                ))

            # 狀態顯示
            self.frw_status_label = ui.label('狀態：OFF')

    def _send_serial_command(self):
        """
        [v3.0.1] 從 UI 輸入框獲取命令，並發布序列埠命令發送請求事件。
        不再直接調用 serial_comm。
        """
        command_text = self.serial_command_buffer.value
        if command_text:
            event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command=command_text)
            self.serial_command_buffer.set_value('')  # 清空輸入框
            log.info(f"> {command_text}")

    def update_ui_elements(self):
        with self.state.lock:
            state_snapshot = self.state
            hw_link_status = self.state.hardware_link_status
            tuning_params_copy = {
                'kp': self.state.tuning_params.kp,
                'kd': self.state.tuning_params.kd,
                'action_scale': self.state.tuning_params.action_scale,
                'bias': self.state.tuning_params.bias
            }
            joint_info_data = None
            mode = self.state.control_mode
            if mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                num_motors = self.state.config.num_motors
                default_pose = self.state.sim.default_pose.copy() if self.state.sim else np.zeros(num_motors)
                if mode == "JOINT_TEST":
                    idx = self.state.joint_test_index
                    joint_info_data = {
                        "mode": "offset", "index": idx,
                        "offset": self.state.joint_test_offsets[idx],
                        "default_angle": default_pose[idx],
                        "actual_angle": self.state.latest_joint_positions[idx]
                    }
                else:
                    idx = self.state.manual_ctrl_index
                    joint_info_data = {
                        "mode": "absolute", "index": idx,
                        "target_angle": self.state.manual_final_ctrl[idx],
                        "actual_angle": self.state.latest_joint_positions[idx]
                    }
            terrain_name = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state)
            std_obs_snapshot = self.state.std_obs.copy()

        for key, desc in self._label_descriptors.items():
            if label_widget := self.status_labels.get(key):
                try:
                    value = desc['getter'](state_snapshot)
                    formatted_text = desc['formatter'](value)
                    label_widget.set_text(f"{desc['title']}: {formatted_text}")
                except Exception as e:
                    log.warning(f"UI 更新失敗 for key '{key}': {e}")

        # 【手冊實作 v1.12】統一更新所有 5 個 12 維儀表板
        core_12d_keys_to_update = {
            'action_raw': state_snapshot.latest_action_raw,
            'final_ctrl': state_snapshot.latest_final_ctrl,
            'onnx_joint_positions': std_obs_snapshot.get('joint_positions'),
            'onnx_joint_velocities': std_obs_snapshot.get('joint_velocities'),
            'onnx_last_action': std_obs_snapshot.get('last_action')
        }
        for key_prefix, vector_data in core_12d_keys_to_update.items():
            if vector_data is not None:
                for i in range(4):
                    for j in range(3):
                        label_key = f"{key_prefix}_{i}_{j}"
                        if label_key in self.status_labels:
                            value = vector_data[i * 3 + j]
                            self.status_labels[label_key].set_text(f"{value: 7.3f}")

        # 【v4.10.5 修改】更新開關的狀態和可用性邏輯
        if self.enable_motors_switch:
            # 開關的可操作性：只有在 VERIFIED 或 MUTED 狀態下才可操作
            is_operable = hw_link_status in [HardwareLinkStatus.VERIFIED, HardwareLinkStatus.MUTED]

            if is_operable:
                self.enable_motors_switch.enable()
            else:
                self.enable_motors_switch.disable()

            # 開關的視覺狀態
            should_be_on = (hw_link_status == HardwareLinkStatus.VERIFIED)
            current_switch_value = self.enable_motors_switch.value
            if current_switch_value != should_be_on:
                self.enable_motors_switch.set_value(should_be_on)

        primary_policy = state_snapshot.policy_manager_ref.primary_policy_name
        if self.status_labels['policy_selector'].value != primary_policy:
            self.status_labels['policy_selector'].set_value(primary_policy)

        if self.ui_terrain_selection != terrain_name:
            self.ui_terrain_selection = terrain_name

        for key, slider in self.param_sliders.items():
            state_value = tuning_params_copy[key]
            if abs(slider.value - state_value) > 1e-4:
                slider.set_value(state_value)

        if joint_info_data and self.joint_control_slider is not None:
            idx = joint_info_data['index']
            actual_angle = joint_info_data['actual_angle']
            if joint_info_data['mode'] == 'offset':
                target_abs = joint_info_data['default_angle'] + joint_info_data['offset']
                text = f"模式: 偏移 | Offset={joint_info_data['offset']:+.2f} | Target={target_abs:+.2f} | Actual={actual_angle:+.2f} | Err={target_abs - actual_angle:+.2f}"
            else:
                target_abs = joint_info_data['target_angle']
                text = f"模式: 絕對 | Target={target_abs:+.2f} | Actual={actual_angle:+.2f} | Err={target_abs - actual_angle:+.2f}"

            if self.joint_selector.value != idx:
                self.joint_selector.set_value(idx)

            target_abs_float = float(target_abs)
            if self.joint_control_slider.value is not None and abs(self.joint_control_slider.value - target_abs_float) > 1e-4:
                self.joint_control_slider.set_value(target_abs_float)

            self.status_labels['joint_info'].set_text(text)

        self._update_onnx_short_vector_labels(std_obs_snapshot)
        log_content = "\n".join(log_queue)
        if self.log_area.value != log_content:
            self.log_area.set_value(log_content)

        # 避免在面板尚未建立或被重建時觸發例外
        if hasattr(self, 'frw_status_label') and self.frw_status_label is not None:
            self.frw_status_label.set_text(f"狀態：{'ON' if self.state.recoil_warning_active else 'OFF'}")

    def _update_onnx_short_vector_labels(self, std_obs_snapshot: Dict):
        dims_dict = self.state.observation_manager_ref.ALL_OBS_DIMS if self.state.observation_manager_ref else {}
        for comp_name, md_widget in self.onnx_input_labels.items():
            value_slice = std_obs_snapshot.get(comp_name)
            if value_slice is not None:
                vec_str = np.array2string(value_slice, precision=3, suppress_small=True, formatter={'float_kind': lambda x: f"{x:7.3f}"})
                md_content = f"`{vec_str}`"
                md_widget.set_content(md_content)
            else:
                md_widget.set_content('`N/A`')

    def _handle_enable_motors_switch_change(self, event):
        """
        【v4.10.5 新增】處理「啟用馬達」開關變更的事件。
        """
        with self.state.lock:
            # 只在連結已驗證或已靜默的狀態下進行切換
            if self.state.hardware_link_status in [HardwareLinkStatus.VERIFIED, HardwareLinkStatus.MUTED]:
                if event.value:  # 如果開關被使用者打開
                    self.state.hardware_link_status = HardwareLinkStatus.VERIFIED
                    log.info("🟢 馬達已由使用者啟用。")
                else:  # 如果開關被使用者關閉
                    self.state.hardware_link_status = HardwareLinkStatus.MUTED
                    log.info("🟡 馬達已由使用者靜默。")

    def run(self):
        ui.run(title="Pupper Robot Console", port=8080)
