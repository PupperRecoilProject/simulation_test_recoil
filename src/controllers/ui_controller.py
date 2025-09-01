from nicegui import ui, app
import numpy as np
import threading
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

if TYPE_CHECKING:
    from src.core.state import SimulationState



class UIController:
    """
    【v4.8.0 修改】管理符合 UX 原則的雙欄獨立滾動佈局。
    
    管理 NiceGUI 介面與互動邏輯。
    """

    def __init__(self, state: 'SimulationState'):
        self.state = state
        self.policy_manager = state.policy_manager_ref
        self.hardware_controller = state.hardware_controller_ref
        self.serial_comm = state.serial_communicator_ref
        self.xbox_handler = state.xbox_handler_ref

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

    # ===================================================================
    #                          UI 佈局創建 (v4.8.2)
    # ===================================================================

    def _setup_ui(self):
        """
        【v4.8.2 重構】引入 ui.splitter 實現健壯的雙欄獨立滾動佈局。

        此版本用 ui.splitter 取代了基於 Flexbox 的 column 佈局，從根本上解決了
        全頁滾動條的問題，並允許使用者自由調整左右欄的寬度。同時，對元件
        順序和樣式進行了精細打磨，以提升整體的使用者體驗。
        """
        ui.dark_mode().enable()
        with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
            ui.label('Pupper 機器人控制台').classes('text-lg')

        # --- 【v4.8.2 核心修改】使用 ui.splitter ---
        with ui.splitter(value=40).classes('w-full h-[calc(100vh-70px)]') as splitter:
            
            # --- 左側面板：互動與控制區 ---
            with splitter.before:
                # gap-y-4: 在卡片之間增加統一的垂直間距
                with ui.scroll_area().classes('w-full h-full p-4'):
                    with ui.column().classes('w-full gap-y-4'):
                        self._create_pinned_controls()
                        self._create_device_panel()
                        self._create_joint_control_panel()
                        self._create_joystick_panel()
                        # 【v4.8.2 修改】將參數微調移至最下方
                        self._create_tuning_panel()

            # --- 右側面板：監控與顯示區 ---
            with splitter.after:
                with ui.scroll_area().classes('w-full h-full p-4'):
                    with ui.column().classes('w-full gap-y-4'):
                        self._create_status_display()
                        self._create_ai_core_display()
                        self._create_log_panel()

        ui.timer(0.1, self.update_ui_elements)


    def _create_pinned_controls(self):
        """
        【v4.8.1 新增】創建置頂的「核心控制」卡片。

        此卡片整合了使用者最高頻的操作，確保它們永遠顯示在左欄頂部，
        無需滾動即可存取，極大地優化了核心工作流程的效率。
        """
        with ui.card().classes('w-full'):
            ui.label('核心控制 (Core Control)').classes('text-lg font-bold')
            ui.separator()

            # 模式控制
            ui.label('模式 (Mode)').classes('text-base font-bold mt-4 mb-2')
            with ui.row():
                ui.button('走路', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING"))
                ui.button('硬體', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE")) \
                  .bind_enabled_from(self.state, 'serial_is_connected')
                ui.button('關節測試', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="JOINT_TEST"))
                
            # 模擬控制
            ui.label('模擬 (Simulation)').classes('text-base font-bold mt-4 mb-2')
            with ui.row().classes('items-center'):
                ui.button().bind_text_from(self.state, 'single_step_mode', 
                                           lambda p: '▶️ 播放' if p else '⏸️ 暫停') \
                           .on('click', self._toggle_pause)
                ui.button('步進', on_click=lambda: setattr(self.state, 'execute_one_step', True)) \
                   .bind_enabled_from(self.state, 'single_step_mode')
                ui.button('硬重置', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"))

            # AI 與策略
            ui.label('AI 與策略 (AI & Policy)').classes('text-base font-bold mt-4 mb-2')
            # 【v4.8.1 修正】確保策略和地形選擇器只在這裡創建
            self.status_labels['policy_selector'] = ui.select(
                options=self.state.available_policies,
                label='選擇 AI 策略',
                value=self.policy_manager.primary_policy_name,
                on_change=lambda e: event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=e.value)
            ).classes('w-full')
            
            terrain_options = ['INFINITE'] + self.state.terrain_manager_ref.single_terrain_names
            self.terrain_selector = ui.select(
                options=terrain_options,
                label='選擇地形',
                on_change=lambda e: event_bus.publish(EVENT_TERRAIN_CHANGE_REQUESTED, name=e.value)
            ).bind_value(self, 'ui_terrain_selection').classes('w-full')


    def _create_tuning_panel(self):
        """
        【v4.8.1 修改】調整樣式並移除重複的元件。

        創建包含 KP、KD 等物理參數調整滑桿的卡片。
        """
        with ui.card().classes('w-full'):
            ui.label('參數微調 (Fine-Tuning)').classes('text-lg font-bold')
            ui.separator()
            params = self.state.tuning_params
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    slider = ui.slider(min=min_val, max=max_val, step=0.01, value=getattr(params, key),
                                       on_change=lambda e, k=key: event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, param_name=k, value=e.value))
                    self.param_sliders[key] = slider
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')
            
            # 【v4.8.1 移除】將策略和地形選擇器移至 _create_pinned_controls，解決重複問題
            # ui.separator() ... 
            # self.status_labels['policy_selector'] = ui.select(...)
            # self.terrain_selector = ui.select(...)


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
        """【v4.8.2 修改】精煉佈局和字體。"""
        with ui.card().classes('w-full'):
            ui.label('關鍵指標 (Key Metrics)').classes('text-lg font-bold')
            ui.separator()
            # 【v4.8.2 修改】使用更具彈性的 row-based 佈局代替 grid
            with ui.column().classes('w-full text-sm gap-y-1 mt-2'):
                keys_to_display = ['mode', 'input_mode', 'sim_time', 'policy_status', 'command', 'robot_pos']
                for key in keys_to_display:
                    desc = self._label_descriptors[key]
                    with self.state.lock:
                        initial_value = desc['getter'](self.state)
                        initial_text = desc['formatter'](initial_value)
                    # 使用 Markdown 的粗體來凸顯標題
                    self.status_labels[key] = ui.markdown(f"**{desc['title']}**: {initial_text}")


    def _create_ai_core_display(self):
        """
        【v4.8.2 重構】徹底重做 ONNX 向量的顯示方式，解決對齊與換行問題。
        """
        with ui.card().classes('w-full'):
            ui.label('AI 核心 (AI Core)').classes('text-lg font-bold')
            ui.separator()
            
            with ui.column().classes('w-full mt-2 gap-y-2'):
                ui.label('AI 策略輸出 (Policy Outputs)').classes('text-base font-bold')
                keys_to_display = ['action_raw', 'final_ctrl']
                for key in keys_to_display:
                    desc = self._label_descriptors[key]
                    with self.state.lock:
                        initial_value = desc['getter'](self.state)
                        initial_text = desc['formatter'](initial_value)
                    # 【v4.8.1 修改】使用 text-xs (最小) 和等寬字體來顯示密集的數字陣列
                    self.status_labels[key] = ui.label(f"**{desc['title']}**: {initial_text}").classes('text-xs font-mono')
            
            # ONNX 觀察向量（放入摺疊面板）
            # 【v4.8.2 修改】預設展開 ONNX 觀察向量
            with ui.expansion('ONNX 觀察向量 (Input)', icon='schema', value=True).classes('w-full mt-2'):
                with ui.column().classes('w-full gap-y-1'):
                    if self.state.observation_manager_ref:
                        all_components = sorted(self.state.observation_manager_ref.ALL_OBS_DIMS.items())
                        for comp_name, dim in all_components:
                            # 【v4.8.2 核心修改】為每一行創建一個 row
                            with ui.row().classes('w-full no-wrap items-center'):
                                # 固定寬度的標題標籤
                                ui.label(f'{comp_name}:').classes('w-48 text-xs font-mono text-right mr-2')
                                # 用於顯示數值的標籤
                                self.onnx_input_labels[comp_name] = ui.label('N/A').classes('text-xs font-mono')
                    else:
                        ui.label("錯誤: ObservationManager 未初始化!")


    # 【v4.8.0 移除】此函式的功能已被 _create_ai_core_display 取代。
    # def _create_onnx_display(self): ...


    # 【v4.8.0 重構】_create_log_panel 放入摺疊面板
    def _create_log_panel(self):
        """
        【v4.8.0 修改】將日誌區域放入一個預設收起的摺疊面板中。
        【v4.8.2 修改】預設展開日誌區域。

        創建系統日誌和序列埠命令輸入區域。為了優化主螢幕空間，
        整個面板被包裹在一個 ui.expansion 元件中，使用者需要時可手動展開。
        """
        # 【v4.8.0 修改】將日誌區域放入摺疊面板，預設關閉
        with ui.expansion('系統日誌與序列埠 (Logs & Serial)', icon='plagiarism', value=True).classes('w-full'):
            with ui.card().classes('w-full no-shadow border'): # 使用無陰影的卡片樣式
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
        【v4.8.2 修改】引入新的格式化函式，強制顯示正負號以確保完美對齊。
        """
        with self.state.lock:
            std_obs_snapshot = self.state.std_obs.copy()

        for comp_name, label_widget in self.onnx_input_labels.items():
            value_slice = std_obs_snapshot.get(comp_name)

            if value_slice is not None:
                # 【v4.8.2 核心修改】使用新的 formatter，"+7.3f" 會強制為正數也顯示 '+' 號
                # 這能確保所有數字（如 +0.123 和 -0.123）佔據完全相同的寬度。
                vec_str = np.array2string(value_slice, 
                                          precision=3, 
                                          suppress_small=True, 
                                          formatter={'float_kind': lambda x: f"{x:+7.3f}"})
                label_widget.set_text(vec_str)
            else:
                label_widget.set_text('N/A')

    def _toggle_pause(self):
        """【v4.7.4 新增】線程安全地切換暫停狀態。"""
        with self.state.lock:
            self.state.single_step_mode = not self.state.single_step_mode

    def run(self):
        ui.run(title="Pupper Robot Console", port=8080)
