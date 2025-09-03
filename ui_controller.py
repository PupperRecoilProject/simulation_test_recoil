from nicegui import ui, app
import numpy as np
import threading
import httpx
import asyncio
from typing import TYPE_CHECKING, List
from src.core.logger import log, log_queue

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
        self.mute_switch = None
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

        # ===================================================================
        # === 【v4.7.0 新增】UI 元素定義區 (Single Source of Truth) ===
        # ===================================================================
        # 存放最終創建的 ui.label 物件
        self.status_labels = {}

        # 【v4.7.0 新增】狀態標籤描述字典
        # 這是 UI 狀態顯示的唯一真相來源。它定義了每個標籤的:
        # - title: 在 UI 上顯示的靜態標題。
        # - getter: 一個從 state 快照中提取對應數據的函式。
        # - formatter: 一個將提取出的數據轉換為顯示字串的函式。
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
                'formatter': lambda v: f"vy: {v[0]:.2f}, vx: {v[1]:.2f}, wz: {v[2]:.2f}"
            },
            'robot_pos': {
                'title': '位置',
                'getter': lambda s: s.latest_pos,
                'formatter': lambda v: f"[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]"
            },
            # --- 【v4.7.0 目標】新增 AI 輸出顯示 ---
            'action_raw': {
                'title': '原始動作 (Raw)',
                'getter': lambda s: s.latest_action_raw,
                'formatter': lambda v: np.array2string(v, precision=3, suppress_small=True, formatter={'float_kind': lambda x: f"{x:7.3f}"})
            },
            'final_ctrl': {
                'title': '最終控制 (Ctrl)',
                'getter': lambda s: s.latest_final_ctrl,
                'formatter': lambda v: np.array2string(v, precision=3, suppress_small=True, formatter={'float_kind': lambda x: f"{x:7.3f}"})
            }
        }

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
                self._create_vision_panel()

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

            # 【v4.7.4 新增】在 UI 上添加模擬控制（暫停/播放/步進）功能
            ui.separator()
            ui.label('模擬控制 (Simulation Control)').classes('text-lg')
            with ui.row():
                # 按鈕的文字會根據 state.single_step_mode 的值動態變化 ('▶️ 播放' 或 '⏸️ 暫停')
                ui.button().bind_text_from(self.state, 'single_step_mode', 
                                           lambda is_paused: '▶️ 播放 (SPACE)' if is_paused else '⏸️ 暫停 (SPACE)') \
                           .on('click', self._toggle_pause) # 使用輔助函式確保線程安全
                
                # 步進按鈕只有在暫停模式下才可點擊
                ui.button('⏭️ 步進 (N)', on_click=lambda: setattr(self.state, 'execute_one_step', True)) \
                   .bind_enabled_from(self.state, 'single_step_mode')
                

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

            # 【v4.10.1 新增】"Dry Run" 安全模式開關
            with ui.row().classes('items-center'):
                ui.label('指令靜默 (Mute)').classes('text-sm')
                self.mute_switch = ui.switch(on_change=self._handle_mute_switch_change) \
                    .tooltip('手動靜默/解除靜默馬達指令。僅在硬體通訊驗證成功後可用。')
                
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
                    # 【v4.9.1 修正】確保送出的指令為 4 維向量，以符合系統目前的設計。
                    # 虛擬搖桿不控制俯仰角 (pitch)，因此第四個元素硬編碼為 0.0。
                    command=np.array([
                        e.x * self.state.config.gamepad_sensitivity['vy'],
                        -e.y * self.state.config.gamepad_sensitivity['vx'],
                        0.0,
                        0.0  # <--- 新增的第四個元素 (pitch)
                    ])
                ),
                on_end=lambda e: (
                    # 【v4.9.1 修正】確保清除指令時，同樣送出 4 維向量。
                    event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(4)),
                    event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="KEYBOARD")
                )
            ).props('throttle')
            
            # 【v4.9.1 修正】確保清除命令按鈕同樣送出 4 維向量。
            ui.button('清除命令 (Clear Command)', on_click=lambda: event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(4))).props('outline')
            

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
            # 【v4.3.4 修改】 為滑桿增加初始值，避免 value 屬性為 None
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01,
                                                 value=0.0, # 【新增】設定初始值為 0.0
                                                 on_change=lambda e: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, value=e.value)
                                                 ).props('label-always')
            with ui.row():
                # [修改] 按鈕發布關節值調整事件
                ui.button('-0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=-0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, direction=0.1)).props('dense')
                ui.button('歸零 (Clear)', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUSTED, clear=True)).props('dense')


    def _create_status_display(self):
        # 【v4.7.0 重構】數據驅動的 UI 元素創建
        # 此函式現在遍歷 _label_descriptors 字典，動態地創建所有狀態標籤。
        # 這使得未來新增或修改狀態顯示變得極為簡單，只需修改字典即可。
        # 【v4.7.4 修改】為卡片設定最小高度以穩定佈局
        with ui.card().style('min-height: 320px'):
            ui.label('即時狀態 (Real-time Status)').classes('text-lg')
            with ui.grid(columns=3): # 可根據標籤數量調整佈局
                for key, desc in self._label_descriptors.items():
                    # 為字典中的每一個描述符，創建一個 ui.label。
                    # 初始文本從 state 中獲取，確保 UI 在創建時就顯示真實數據。
                    with self.state.lock:
                        initial_value = desc['getter'](self.state)
                        initial_text = desc['formatter'](initial_value)
                    
                    # 將創建的 label 物件存儲起來，以便後續更新。
                    self.status_labels[key] = ui.label(f"{desc['title']}: {initial_text}")

    def _create_vision_panel(self):
        """【最終穩定版】在點擊事件中動態生成並執行完整的 JS 邏輯。"""
        with ui.card():
            ui.label('即時視覺 (NanoOwl)').classes('text-lg font-bold')
            
            video_server_url = "http://localhost:8081"
            ui.html(f'''
                <iframe src="{video_server_url}" 
                        width="100%" height="480" frameborder="0" scrolling="no"
                        style="max-width: 640px; aspect-ratio: 640 / 480;">
                </iframe>
            ''')

            with ui.row().classes('w-full items-center'):
                prompt_input = ui.input(placeholder='輸入識別提示, e.g., a person').props('outlined dense').classes('flex-grow')
                
                def handle_send_prompt():
                    # 1. 從 Python 端獲取輸入框的當前值
                    prompt_value = prompt_input.value
                    if not prompt_value:
                        # 如果輸入框為空，則什麼都不做
                        ui.notify('請輸入識別提示！', color='warning')
                        return

                    # 2. 動態生成一段完整的 JavaScript 腳本
                    #    這段腳本將負責所有事情：連接、檢查狀態、發送消息
                    javascript_code = f'''
                        // 將 Python 變數的值安全地傳遞給 JavaScript
                        const prompt_text = `{prompt_value}`;

                        console.log(`[Python -> JS] 準備處理 prompt: "${{prompt_text}}"`);

                        // 檢查 window 對象上是否已經有我們的 WebSocket 實例
                        if (!window.nanoowl_ws || window.nanoowl_ws.readyState === WebSocket.CLOSED) {{
                            console.log("WebSocket 未連接或已關閉，正在創建新連接...");
                            window.nanoowl_ws = new WebSocket('ws://localhost:8081/ws');
                            
                            window.nanoowl_ws.onopen = () => {{
                                console.log('%c[WebSocket] 连接成功！现在发送消息...', 'color: green; font-weight: bold;');
                                // 連接成功後，立即發送消息
                                window.nanoowl_ws.send(prompt_text);
                                // 發送後清空 Python 端的輸入框
                                document.getElementById('{prompt_input.id}').value = '';
                            }};
                            
                            window.nanoowl_ws.onclose = () => {{
                                console.warn('[WebSocket] 連接已斷開。');
                            }};
                            
                            window.nanoowl_ws.onerror = (error) => {{
                                console.error('[WebSocket] 發生錯誤:', error);
                            }};
                        }} 
                        // 如果 WebSocket 已經存在且處於連接狀態
                        else if (window.nanoowl_ws.readyState === WebSocket.OPEN) {{
                             console.log('[WebSocket] 連接已存在，直接發送消息...');
                             window.nanoowl_ws.send(prompt_text);
                             // 發送後清空 Python 端的輸入框
                             document.getElementById('{prompt_input.id}').value = '';
                        }}
                        // 如果正在連接中，則等待連接成功後再發送
                        else if (window.nanoowl_ws.readyState === WebSocket.CONNECTING) {{
                            console.log('[WebSocket] 正在連接中... 將在連接成功後發送消息。');
                            window.nanoowl_ws.addEventListener('open', () => {{
                                console.log('[WebSocket] 連接成功，現在發送延遲的消息...');
                                window.nanoowl_ws.send(prompt_text);
                                document.getElementById('{prompt_input.id}').value = '';
                            }}, {{ once: true }}); // once: true 確保這個監聽器只執行一次
                        }}
                    '''
                    
                    # 3. 執行這段動態生成的腳本
                    ui.run_javascript(javascript_code)
                    
                    # 4. 在 Python 端也清空輸入框的值，以保持同步
                    prompt_input.set_value(None)

                # 將同一個處理函數綁定到按鈕點擊和輸入框回車事件
                ui.button('識別', on_click=handle_send_prompt)
                prompt_input.on('keydown.enter', handle_send_prompt)
    
    def _create_onnx_display(self):
        """
        【v4.6.0 重構】建立 ONNX 觀察向量區域。
        此版本不再使用硬編碼列表，而是動態地從 ObservationManager 的
        權威維度字典 (ALL_OBS_DIMS) 中創建所有 UI 標籤，確保 UI
        能夠自動適應未來新增的觀測元件。
        """
        # 【v4.7.4 修改】為卡片設定最小高度以穩定佈局
        with ui.card().style('min-height: 240px'):
            ui.label('ONNX 觀察向量 (Observation Vector)').classes('text-lg')
            with ui.grid(columns=2):
                # 【v4.6.0 修改】 核心修改：動態生成 UI 標籤
                # 我們直接從 observation_manager 實例中獲取 ALL_OBS_DIMS 字典。
                # 這是觀測元件的「單一事實來源」。
                # .items() 會返回 (鍵, 值) 對，即 (元件名, 維度)。
                # 我們對其進行排序，以確保 UI 上的顯示順序是固定的、可預測的。
                if self.state.observation_manager_ref:
                    all_components = sorted(self.state.observation_manager_ref.ALL_OBS_DIMS.items())
                    
                    for comp_name, dim in all_components:
                        # 為字典中的每一個元件，都創建一個 ui.label，並將其
                        # 存儲在 self.onnx_input_labels 字典中，以便後續更新。
                        # 初始文字設置為 'N/A'，等待 update_ui_elements 填充真實數據。
                        self.onnx_input_labels[comp_name] = ui.label(f'{comp_name}: N/A')
                else:
                    # 這是一個防禦性程式碼，如果 observation_manager 尚未初始化，
                    # UI 會顯示一條清晰的錯誤訊息，而不是崩潰。
                    ui.label("錯誤: ObservationManager 未初始化!")


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
        【v3.0.1】從 UI 輸入框獲取命令，並發布序列埠命令發送請求事件。
        不再直接調用 serial_comm。
        """
        command_text = self.serial_command_buffer.value
        if command_text:
            event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command=command_text)
            self.serial_command_buffer.set_value('') # 清空輸入框
            log.info(f"> {command_text}")


    def update_ui_elements(self):
        """
        【v3.0.1】定期從 SimulationState 讀取最新數據，並更新所有UI元件。
        【v4.6.0 重構】重構關節控制 UI 的更新邏輯，並將數據獲取與 UI 更新分離。
        【v4.7.0 重構】採用數據驅動模型，自動化更新所有由描述字典管理的標籤。
        """
        # =================================================================
        # === 階段一：原子性地從 State 中獲取數據快照 (Atomic Data Snapshot) ===
        # =================================================================
        with self.state.lock:
            # 我們傳遞整個 state 物件的參考，讓 getter 函式自己去解析。
            # 這比手動複製每個變量更簡潔、更具擴展性。
            state_snapshot = self.state

            # 【v4.10.1 新增】獲取連結狀態
            hw_link_status = self.state.hardware_link_status

            # 【保留】非描述性數據仍需手動獲取
            # 參數調校狀態 (用於滑桿)
            tuning_params_copy = {
                'kp': self.state.tuning_params.kp,
                'kd': self.state.tuning_params.kd,
                'action_scale': self.state.tuning_params.action_scale,
                'bias': self.state.tuning_params.bias
            }
            # 關節控制狀態
            joint_info_data = None
            mode = self.state.control_mode # 模式判斷仍然需要
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
                else: # MANUAL_CTRL
                    idx = self.state.manual_ctrl_index
                    joint_info_data = {
                        "mode": "absolute", "index": idx,
                        "target_angle": self.state.manual_final_ctrl[idx],
                        "actual_angle": self.state.latest_joint_positions[idx]
                    }
            # 地形狀態
            terrain_name = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state)

        # =================================================================
        # === 階段二：使用數據快照安全地更新所有 UI 元件 (UI Update) ===
        # =================================================================
        
        # --- 【v4.7.0 新增】自動化更新所有由描述字典管理的標籤 ---
        for key, desc in self._label_descriptors.items():
            # 使用 .get() 進行最終防禦，確保即使描述字典和創建的標籤不匹配也不會崩潰。
            if label_widget := self.status_labels.get(key):
                try:
                    value = desc['getter'](state_snapshot)
                    formatted_text = desc['formatter'](value)
                    label_widget.set_text(f"{desc['title']}: {formatted_text}")
                except Exception as e:
                    # 防禦性程式碼：如果 getter 或 formatter 出錯，只打印警告，不讓 UI 崩潰。
                    log.warning(f"UI 更新失敗 for key '{key}': {e}")
        
        # 【v4.10.1 新增】根據連結狀態更新 Mute 開關的狀態和可用性
        if self.mute_switch:
            is_verified_or_muted = hw_link_status in [HardwareLinkStatus.VERIFIED, HardwareLinkStatus.MUTED]
            
            # 【v4.10.2 修正】使用正確的 enable()/disable() API
            # 根據 is_verified_or_muted 的布林值，呼叫對應的不帶參數的方法。
            if is_verified_or_muted:
                self.mute_switch.enable()
            else:
                self.mute_switch.disable()
            
            # 根據狀態更新開關的顯示 (on/off)
            current_switch_value = self.mute_switch.value
            should_be_on = (hw_link_status == HardwareLinkStatus.MUTED)
            if current_switch_value != should_be_on:
                self.mute_switch.set_value(should_be_on)

        # --- 更新非描述性/複雜的 UI 元件 (保留現有邏輯) ---
        # 更新 AI 策略下拉選單
        primary_policy = state_snapshot.policy_manager_ref.primary_policy_name
        if self.status_labels['policy_selector'].value != primary_policy:
            self.status_labels['policy_selector'].set_value(primary_policy)
        
        # 更新地形選擇 UI
        if self.ui_terrain_selection != terrain_name:
            self.ui_terrain_selection = terrain_name

        # 更新參數調校滑桿
        for key, slider in self.param_sliders.items():
            state_value = tuning_params_copy[key]
            if abs(slider.value - state_value) > 1e-4:
                slider.set_value(state_value)

        # 更新關節控制 UI
        if joint_info_data and self.joint_control_slider is not None:
            idx = joint_info_data['index']
            actual_angle = joint_info_data['actual_angle']
            if joint_info_data['mode'] == 'offset':
                target_abs = joint_info_data['default_angle'] + joint_info_data['offset']
                text = f"模式: 偏移 | Offset={joint_info_data['offset']:+.2f} | Target={target_abs:+.2f} | Actual={actual_angle:+.2f} | Err={target_abs - actual_angle:+.2f}"
            else: # 'absolute'
                target_abs = joint_info_data['target_angle']
                text = f"模式: 絕對 | Target={target_abs:+.2f} | Actual={actual_angle:+.2f} | Err={target_abs - actual_angle:+.2f}"
            
            if self.joint_selector.value != idx: self.joint_selector.set_value(idx)
            target_abs_float = float(target_abs)
            if self.joint_control_slider.value is not None and abs(self.joint_control_slider.value - target_abs_float) > 1e-4:
                self.joint_control_slider.set_value(target_abs_float)
            self.status_labels['joint_info'].set_text(text)

        # 更新 ONNX 觀察向量和日誌
        self._update_onnx_labels()
        log_content = "\n".join(log_queue)
        if self.log_area.value != log_content:
            self.log_area.set_value(log_content)


    def _update_onnx_labels(self):
        """
        【v4.4.7 重構】
        從 state.std_obs 這個單一權威數據源讀取數據並更新 UI。
        【v4.7.0 修改】新增固定寬度數字格式化，提升可讀性。
        """
        with self.state.lock:
            std_obs_snapshot = self.state.std_obs.copy()

        for comp_name, label_widget in self.onnx_input_labels.items():
            value_slice = std_obs_snapshot.get(comp_name)

            if value_slice is not None:
                # 【v4.7.0 修改】使用固定寬度的數字格式化，類似 pyserial_console.py
                # 這個 formatter 確保每個數字都佔用 7 個字符寬度，小數點後保留 3 位。
                # 這將使得所有向量的排版都非常整齊，提升可讀性。
                vec_str = np.array2string(value_slice, 
                                          precision=3, 
                                          suppress_small=True, 
                                          formatter={'float_kind': lambda x: f"{x:7.3f}"})
                label_widget.set_text(f'{comp_name}: {vec_str}')
            else:
                label_widget.set_text(f'{comp_name}: N/A')

    def _toggle_pause(self):
        """【v4.7.4 新增】線程安全地切換暫停狀態。"""
        with self.state.lock:
            self.state.single_step_mode = not self.state.single_step_mode

    def _handle_mute_switch_change(self, event):
        """【v4.10.1 新增】處理靜默開關變更的事件，線程安全地更新狀態。"""
        with self.state.lock:
            # 只在連結已驗證或已靜默的狀態下進行切換
            if self.state.hardware_link_status in [HardwareLinkStatus.VERIFIED, HardwareLinkStatus.MUTED]:
                if event.value: # 如果開關被使用者打開 (要求靜默)
                    self.state.hardware_link_status = HardwareLinkStatus.MUTED
                else: # 如果開關被使用者關閉 (要求解除靜默)
                    self.state.hardware_link_status = HardwareLinkStatus.VERIFIED


    def run(self):
        ui.run(title="Pupper Robot Console", port=8080)
        
    # 在 class UIController 中加入這個新方法

    def inject_websocket_script(self):
        """
        在 UI 啟動後注入 WebSocket 通信的 JavaScript。
        這個方法應該由 app.on_connect 事件觸發。
        """
        # 確保 self.prompt_input 已經被創建
        if not self.prompt_input:
            log.error("無法注入 JS：prompt_input 尚未被初始化。")
            return

        log.info("正在向客戶端注入 WebSocket 連接腳本...")
        ui.run_javascript(f'''
            let ws_{self.prompt_input.id};
            
            function connect_{self.prompt_input.id}() {{
                ws_{self.prompt_input.id} = new WebSocket('ws://localhost:8081/ws');
                ws_{self.prompt_input.id}.onopen = () => console.log('成功连接到 NanoOwl 视频服务器的 WebSocket。');
                ws_{self.prompt_input.id}.onclose = () => {{
                    console.log('与视频服务器的 WebSocket 连接已断开。将在2秒后尝试重新连接...');
                    setTimeout(connect_{self.prompt_input.id}, 2000);
                }};
                ws_{self.prompt_input.id}.onerror = (error) => {{
                    console.error('WebSocket 发生错误: ', error);
                }};
            }}
            
            function send_prompt_{self.prompt_input.id}() {{
                const input_element = document.getElementById('{self.prompt_input.id}');
                const prompt_text = input_element.value;
                if (prompt_text && ws_{self.prompt_input.id} && ws_{self.prompt_input.id}.readyState === WebSocket.OPEN) {{
                    ws_{self.prompt_input.id}.send(prompt_text);
                    input_element.value = '';
                }} else {{
                    console.error('无法发送 prompt：WebSocket 未连接或输入为空。');
                }}
            }}
            
            connect_{self.prompt_input.id}();
        ''')
        log.info("✅ WebSocket 腳本注入完成。")
