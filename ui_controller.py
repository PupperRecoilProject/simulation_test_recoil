# ui_controller.py
import numpy as np
from nicegui import ui, app
import threading
from typing import TYPE_CHECKING, Any, List, Dict # 導入 Any 以用於類型提示不確定型別的物件
from logger import log, log_queue

# 從 typing 模組導入 TYPE_CHECKING 以用於類型提示。
if TYPE_CHECKING:
    from state import SimulationState
    from policy import PolicyManager
    from hardware_controller import HardwareController
    from serial_communicator import SerialCommunicator
    from xbox_input_handler import XboxInputHandler
    from terrain_manager import TerrainManager
    from simulation import Simulation # 假設在 update_ui_elements 中可能需要 sim 的屬性

# 導入事件系統模組和所有需要使用的事件類型。
from event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_TUNING_PARAM_ADJUST_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_COMMAND_UPDATED,
    EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUST_REQUESTED,
    EVENT_TERRAIN_MODE_CHANGE_REQUESTED
)


class UIController:
    """
    [表現層] NiceGUI 介面控制器。
    負責構建和管理 NiceGUI 網頁介面，捕捉使用者互動，
    並將這些互動翻譯成標準化的請求事件發布到事件匯流排 (`event_bus`)。
    它不直接修改核心狀態，只讀取 `SimulationState` 以更新介面顯示。
    """
    def __init__(self, state: 'SimulationState'):
        """
        [初始化] 初始化 UIController。
        Args:
            state (SimulationState): 全域模擬狀態的參考。
        """
        self.state = state
        
        # UIController 不再直接持有這些模組的引用，因為它只與 state 和 event_bus 交互。
        # 這些參考的唯一作用是為了初始化時獲取選項列表或綁定。
        # self.policy_manager = state.policy_manager_ref
        # self.hardware_controller = state.hardware_controller_ref
        # self.serial_comm = state.serial_communicator_ref
        # self.xbox_handler = state.xbox_handler_ref

        # UI 元件的參考字典，用於後續更新。
        self.status_labels: Dict[str, ui.label] = {}
        self.param_sliders: Dict[str, ui.slider] = {} # 實際上可能不再需要，因為直接使用 `getattr`
        self.onnx_input_labels: Dict[str, ui.label] = {}
        self.log_area: ui.textarea | None = None
        self.serial_command_buffer_input: ui.input | None = None # 將名稱改為 input 結尾，避免與 buffer 變數混淆
        self.joint_control_slider: ui.slider | None = None
        self.policy_selector_ui: ui.select | None = None # 新增策略選擇器 UI 元件參考
        self.terrain_selector_ui: ui.select | None = None # 新增地形選擇器 UI 元件參考
        self.joint_selector_ui: ui.select | None = None # 新增關節選擇器 UI 元件參考

        # 用於 UI 下拉選單的地形選擇值，避免與後端狀態互相觸發。
        # 在初始化時，如果 terrain_manager_ref 存在，則根據後端實際狀態初始化 UI 顯示值。
        if self.state.terrain_manager_ref and self.state.terrain_mode == 'SINGLE':
            self.ui_terrain_selection = self.state.terrain_manager_ref.single_terrain_names[self.state.single_terrain_index]
        else:
            self.ui_terrain_selection = 'INFINITE'

        self._setup_ui() # 呼叫設定 UI 的方法

    def _setup_ui(self):
        """
        [UI 佈局] 設定整個 NiceGUI 應用程式的佈局和初始元件。
        """
        ui.dark_mode().enable() # 啟用暗色模式
        with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
            ui.label('Pupper 機器人控制台').classes('text-lg')

        with ui.row().classes('w-full no-wrap'):
            # 左側欄：控制相關的面板
            with ui.column().classes('w-1/3'):
                # 使用垂直 Tabs 組織面板
                with ui.tabs().props('vertical').classes('w-full') as tabs:
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

                # 關節微調與搖桿控制面板，獨立於 Tabs 面板下方
                self._create_joint_control_panel()
                self._create_joystick_panel()

            # 右側欄：狀態與日誌顯示
            with ui.column().classes('w-2/3'):
                self._create_status_display()
                self._create_onnx_display()
                self._create_log_panel()

        # [重要修正] 調整 timer 的執行方式，讓它在 NiceGUI 應用程式完全啟動後才開始輪詢更新 UI。
        # 這是為了確保在 UI 元件嘗試綁定或讀取狀態時，後端核心模組（如 policy_manager_ref）已經被初始化。
        app.on_startup(lambda: ui.timer(0.1, self.update_ui_elements))

    def _create_main_control_panel(self):
        """
        [UI 佈局] 創建主控制面板，包含模式切換和重置按鈕。
        所有按鈕的回呼函式都發布相應的請求事件。
        """
        with ui.card():
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                ui.button('走路 (Walking)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING"))
                ui.button('懸浮 (Floating)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="FLOATING"))
                ui.button('硬體 (Hardware)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE"))
            with ui.row():
                ui.button('關節測試 (Joint Test)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="JOINT_TEST"))
                ui.button('手動控制 (Manual Ctrl)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="MANUAL_CTRL"))

            ui.separator()
            ui.label('重置').classes('text-lg')
            with ui.row():
                ui.button('軟重置 (X)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft"))
                ui.button('硬重置 (R)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"))

    def _create_tuning_panel(self):
        """
        [UI 佈局] 創建參數調整和 AI 策略選擇面板。
        滑桿和選擇框的變更都發布請求事件。
        """
        with ui.card().classes('w-full'):
            ui.label('參數調整 (Tuning)').classes('text-lg')
            params = self.state.tuning_params # 直接從 state 獲取參數物件的參考。
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    # 滑桿：不再使用 bind_value，而是通過 on_change 事件發布參數調整請求。
                    # `value` 屬性直接從 `params` 中讀取初始值。
                    slider = ui.slider(min=min_val, max=max_val, step=0.01,
                                       value=getattr(params, key), # 設置初始值
                                       on_change=lambda e, k=key: event_bus.publish(EVENT_TUNING_PARAM_ADJUST_REQUESTED, param_name=k, value=e.value))\
                                       .classes('w-48')
                    # 標籤：使用 bind_text_from 雙向綁定到 `params` 屬性，以顯示即時數值。這是安全的讀取綁定。
                    self.param_sliders[key] = slider # 保存對滑桿的引用以便後續同步（如果需要）
                    ui.label().bind_text_from(params, key, lambda v: f'{v:.2f}')
            
            ui.separator()
            ui.label('策略選擇 (Policy)').classes('text-lg')
            # 策略選擇下拉選單：通過 on_change 事件發布策略切換請求。
            # 初始選項來自 state.available_policies。
            # 初始值在 update_ui_elements 中設置，避免在初始化時 policy_manager_ref 可能為 None。
            self.policy_selector_ui = ui.select(
                options=self.state.available_policies, # 可用策略模型的名稱列表
                label='Active Policy',
                on_change=lambda e: event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=e.value)
            ).classes('w-full')

            # 地形模式選擇下拉選單
            terrain_options = ['INFINITE'] + (self.state.terrain_manager_ref.single_terrain_names if self.state.terrain_manager_ref else [])
            self.terrain_selector_ui = ui.select(
                options=terrain_options,
                label='Terrain Mode',
                on_change=self._on_terrain_change_publish_event # 自定義回呼函式，用於發布地形切換事件
            ).bind_value(self, 'ui_terrain_selection').classes('w-full') # 綁定到 UIController 自身屬性，是安全的。

    def _create_device_panel(self):
        """
        [UI 佈局] 創建設備連接和系統控制面板。
        所有按鈕的回呼函式都發布請求事件。
        """
        with ui.card():
            ui.label('硬體 AI 控制').classes('text-lg')
            ui.button('啟用/停用 AI (K)', on_click=lambda: event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED)).bind_enabled_from(
                self.state, 'control_mode', lambda mode: mode == "HARDWARE_MODE")

            ui.separator()
            ui.label('設備連接').classes('text-lg')
            with ui.row():
                # 發布設備連接請求事件，指明要連接的設備類型。
                ui.button('連接序列埠 (U)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial"))
                ui.button('連接搖桿 (J)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad"))

            ui.separator()
            ui.label('系統').classes('text-lg')
            ui.button('退出程式', on_click=lambda: event_bus.publish(EVENT_SHUTDOWN_REQUESTED), color='red')

    def _create_joystick_panel(self):
        """
        [UI 佈局] 創建虛擬搖桿控制面板。
        虛擬搖桿的移動和結束都發布命令更新事件。
        """
        with ui.card().classes('w-full'):
            ui.label('手動駕駛 (Manual Driving)').classes('text-lg')
            ui.joystick(
                color='blue',
                size=100,
                on_move=self._update_command_from_joystick,  # 搖桿移動時的回呼
                on_end=self._on_joystick_end # 搖桿釋放時的回呼
            ).props('throttle')
            ui.button('清除命令 (Clear Command)', on_click=lambda: event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(3, dtype=np.float32))).props('outline')

    def _create_joint_control_panel(self):
        """
        [UI 佈局] 創建關節微調控制面板。
        在 JOINT_TEST 或 MANUAL_CTRL 模式下可見。
        所有關節相關的 UI 互動都發布請求事件。
        """
        # 面板可見性綁定到控制模式。
        with ui.card().bind_visibility_from(self.state, 'control_mode', lambda m: m in ["JOINT_TEST", "MANUAL_CTRL"]).classes('w-full'):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            
            # 手動模式懸浮開關
            with ui.row().classes('items-center'):
                ui.label('啟用懸浮')
                # 開關狀態變化時，發布懸浮切換請求事件。
                ui.switch(on_change=lambda e: event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED, value=e.value))\
                  .bind_value(self.state, 'manual_mode_is_floating') # 綁定到 state 自身屬性，安全。
            
            # 關節名稱字典。
            joint_names = {i: name for i, name in enumerate([
                "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee", "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
                "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee", "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
            ])}
            
            # 關節選擇器下拉選單。
            # 選擇變化時，發布關節選擇請求事件。
            self.joint_selector_ui = ui.select(
                joint_names,
                label='選擇關節',
                on_change=lambda e: event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, index=int(e.value))
            )
            # 初始值和實時顯示綁定到 state.joint_test_index。
            self.joint_selector_ui.bind_value_from(self.state, 'joint_test_index')

            self.status_labels['joint_info'] = ui.label('') # 顯示關節詳細資訊的標籤

            # 關節控制滑桿。
            # 滑桿值變化時，發布關節數值調整請求事件。
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01,
                on_change=lambda e: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, value=e.value))\
                .props('label-always')

            with ui.row():
                # 調整按鈕：發布關節數值調整請求事件（方向性調整）。
                ui.button('-0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=-1, step=0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=1, step=0.1)).props('dense')
                # 歸零按鈕：發布關節數值調整請求事件（清除）。
                ui.button('歸零 (Clear)', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, clear=True)).props('dense')

    def _create_status_display(self):
        """
        [UI 佈局] 創建即時狀態顯示面板。
        這些標籤的內容都通過 `update_ui_elements` 周期性地從 `state` 中讀取並更新。
        """
        with ui.card():
            ui.label('即時狀態 (Real-time Status)').classes('text-lg')
            with ui.grid(columns=3):
                self.status_labels['mode'] = ui.label('模式: WALKING')
                self.status_labels['input_mode'] = ui.label('輸入: KEYBOARD')
                self.status_labels['sim_time'] = ui.label('時間: 0.00s')
                self.status_labels['serial_status'] = ui.label('序列埠: Disconnected')
                self.status_labels['gamepad_status'] = ui.label('搖桿: Disconnected')
                self.status_labels['hardware_ai'] = ui.label('硬體AI: N/A')
                self.status_labels['policy_status'] = ui.label(f'策略: N/A') # 初始化時可能還沒有策略
            ui.separator()
            ui.label('運動指令 (Command)').classes('font-bold')
            self.status_labels['command'] = ui.label('vy: 0.00, vx: 0.00, wz: 0.00')
            ui.label('機器人狀態 (Robot State)').classes('font-bold')
            self.status_labels['robot_pos'] = ui.label('位置: [0.0, 0.0, 0.0]')
            self.status_labels['robot_vel'] = ui.label('速度: [0.0, 0.0, 0.0]') # 這個可能需要從 sim.data 或 observation 中計算

    def _create_onnx_display(self):
        """
        [UI 佈局] 創建 ONNX 觀察向量顯示面板。
        顯示 AI 模型輸入各分量的即時數值。
        """
        with ui.card().style('min-height: 220px;'): # 設定最小高度避免畫面跳動
            ui.label('ONNX 觀察向量 (Observation Vector)').classes('text-lg')
            with ui.grid(columns=2):
                obs_components = [
                    'linear_velocity', 'angular_velocity', 'gravity_vector', 'commands',
                    'accelerometer', 'joint_positions', 'joint_velocities', 'last_action',
                    'z_angular_velocity', 'foot_contact_states', 'phase_signal' # 可能包含更多
                ]
                for comp in obs_components:
                    self.onnx_input_labels[comp] = ui.label(f'{comp}: N/A')

    def _create_log_panel(self):
        """
        [UI 佈局] 創建系統日誌和序列埠控制台面板。
        顯示應用程式日誌，並提供序列埠指令輸入功能。
        """
        with ui.card().classes('w-full'):
            ui.label('系統日誌與序列埠控制台').classes('text-lg')
            self.log_area = ui.textarea(label='Log').props('readonly outlined rows=10').style('width: 100%;')
            with ui.row().classes('w-full items-center'):
                # 輸入框綁定 Enter 鍵事件，按下 Enter 即送出指令。
                self.serial_command_buffer_input = ui.input(label='Serial Command')\
                    .props('outlined dense').classes('flex-grow')\
                    .on('keydown.enter', self._send_serial_command)
                ui.button('Send', on_click=self._send_serial_command)

    def _update_command_from_joystick(self, event: Any):
        """
        [事件回呼] 虛擬搖桿移動時的回呼函式。
        將搖桿的 x, y 座標轉換為機器人的 vx, vy 命令，並發布更新事件。
        Args:
            event (Any): 虛擬搖桿事件物件，包含 x, y 座標。
        """
        x_val = -event.y # Y 值通常對應前後方向，這裡可能需要反向
        y_val = event.x  # X 值對應左右方向

        new_command = np.zeros(3, dtype=np.float32)
        new_command[0] = y_val * self.state.config.gamepad_sensitivity['vy'] # 左右速度
        new_command[1] = x_val * self.state.config.gamepad_sensitivity['vx'] # 前後速度
        # wz (轉向速度) 預設為 0，除非虛擬搖桿有特殊轉向功能或由其他輸入控制。
        # 如果需要轉向，則需要另一個虛擬搖桿。

        event_bus.publish(EVENT_COMMAND_UPDATED, command=new_command)

    def _on_joystick_end(self, event: Any):
        """
        [事件回呼] 虛擬搖桿釋放（停止拖動）時的回呼函式。
        發布一個清除運動指令的事件，使機器人停止移動。
        Args:
            event (Any): 虛擬搖桿事件物件。
        """
        # 清除所有運動指令。
        event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(3, dtype=np.float32))

    def _on_terrain_change_publish_event(self, event: Any):
        """
        [事件回呼] 當地形下拉選單改變時，發布地形模式切換請求事件。
        Args:
            event (Any): NiceGUI 事件物件，包含選擇的值。
        """
        if event.value is not None:
            # 發布地形模式切換請求，包含目標模式名稱。
            event_bus.publish(EVENT_TERRAIN_MODE_CHANGE_REQUESTED, mode_name=event.value)

    def _send_serial_command(self):
        """
        [本地操作] 從序列埠輸入框獲取命令並發送。
        這是一個 UI 自身的本地操作，不影響核心狀態，故不發布事件。
        """
        if self.serial_command_buffer_input:
            command_text = self.serial_command_buffer_input.value
            if command_text and self.state.serial_communicator_ref:
                self.state.serial_communicator_ref.send_command(command_text)
                self.serial_command_buffer_input.set_value('') # 清空輸入框
                log.info(f"> {command_text}")

    def update_ui_elements(self):
        """
        [UI 更新] 定時器回呼函式，用於從 SimulationState 讀取最新狀態並更新所有 UI 元件。
        這是 UI 層的核心渲染邏輯，負責將後端狀態可視化。
        """
        # --- 1. 從共享狀態安全地獲取數據 ---
        # 使用狀態鎖確保在多執行緒環境下讀取數據的一致性。
        with self.state.lock:
            # 基本狀態資訊
            mode = self.state.control_mode
            input_mode = self.state.input_mode
            sim_time = self.state.sim.data.time if self.state.sim else None
            serial_connected = self.state.serial_is_connected
            gamepad_connected = self.state.gamepad_is_connected
            hw_ai_active = self.state.hardware_ai_is_active
            command = self.state.command.copy()
            
            # 機器人位置與姿態 (從模擬器獲取)
            pos = self.state.latest_pos.copy()
            # 機器人速度 (從模擬器獲取，如果需要，可以從 cvel[torso_id, 3:] 或 observation 計算)
            # 由於 state.robot_vel 暫時沒有直接更新，這裡先用佔位符。
            robot_vel_display = np.zeros(3) # 假設，實際應從 state 或 sim.data 獲取

            # AI 策略狀態
            pm = self.state.policy_manager_ref
            transitioning = pm.is_transitioning if pm else False
            alpha = pm.transition_alpha if pm else 0.0
            src_policy = pm.source_policy_name if pm else "N/A"
            tgt_policy = pm.target_policy_name if pm else "N/A"
            primary_policy = pm.primary_policy_name if pm else "N/A"

            # 地形資訊
            terrain_name_simple = self.state.terrain_manager_ref.get_current_terrain_name_simple(self.state) if self.state.terrain_manager_ref else "N/A"
            
            # 關節測試/手動控制模式資訊
            joint_info = None
            if mode == "JOINT_TEST":
                # 確保 sim 和 default_pose 存在，避免 mock_simulation 下報錯
                if self.state.sim and hasattr(self.state.sim, 'default_pose'):
                    idx = self.state.joint_test_index
                    offset = self.state.joint_test_offsets[idx]
                    default_pos = self.state.sim.default_pose[idx]
                    target_abs = default_pos + offset
                    actual_abs = self.state.latest_joint_positions[idx]
                    joint_info = {
                        "mode": "offset",
                        "index": idx,
                        "target_abs": target_abs,
                        "actual_abs": actual_abs,
                        "offset": offset,
                    }
            elif mode == "MANUAL_CTRL":
                if self.state.sim and hasattr(self.state.sim, 'default_pose'):
                    idx = self.state.manual_ctrl_index
                    target_abs = self.state.manual_final_ctrl[idx]
                    actual_abs = self.state.latest_joint_positions[idx]
                    joint_info = {
                        "mode": "absolute",
                        "index": idx,
                        "target_abs": target_abs,
                        "actual_abs": actual_abs,
                    }

        # --- 2. 在鎖外安全地更新 UI 元件 ---
        # 更新核心狀態標籤
        self.status_labels['mode'].set_text(f"模式: {mode}")
        self.status_labels['input_mode'].set_text(f"輸入: {input_mode}")
        self.status_labels['sim_time'].set_text(f"時間: {sim_time:.2f}s" if sim_time is not None else "時間: N/A")
        self.status_labels['serial_status'].set_text('序列埠: Connected' if serial_connected else '序列埠: Disconnected')
        self.status_labels['gamepad_status'].set_text('搖桿: Connected' if gamepad_connected else '搖桿: Disconnected')
        self.status_labels['hardware_ai'].set_text('硬體AI: Active' if hw_ai_active else '硬體AI: Disabled' if mode == 'HARDWARE_MODE' else '硬體AI: N/A')

        # 更新命令和機器人位置
        self.status_labels['command'].set_text(f"vy: {command[0]:.2f}, vx: {command[1]:.2f}, wz: {command[2]:.2f}")
        self.status_labels['robot_pos'].set_text(f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        self.status_labels['robot_vel'].set_text(f"速度: [{robot_vel_display[0]:.2f}, {robot_vel_display[1]:.2f}, {robot_vel_display[2]:.2f}]")

        # 更新策略狀態和選擇器
        if transitioning:
            policy_text = f"策略: Blending {src_policy} -> {tgt_policy} ({alpha*100:.0f}%)"
        else:
            policy_text = f"策略: {primary_policy}"
        self.status_labels['policy_status'].set_text(policy_text)
        # [修正] 確保 policy_selector_ui 已經被實例化，並只在值不同時才更新
        if self.policy_selector_ui and self.policy_selector_ui.value != primary_policy:
            self.policy_selector_ui.set_value(primary_policy)

        # 更新地形選擇器狀態
        if self.terrain_selector_ui and self.ui_terrain_selection != terrain_name_simple:
            self.ui_terrain_selection = terrain_name_simple
            self.terrain_selector_ui.set_value(self.ui_terrain_selection) # 確保 UI 顯示與後端狀態一致

        # 更新關節控制資訊和滑桿/選擇器
        if self.joint_selector_ui:
            # 同步關節選擇器
            current_joint_idx = self.state.joint_test_index if mode == "JOINT_TEST" else self.state.manual_ctrl_index
            if self.joint_selector_ui.value != current_joint_idx:
                self.joint_selector_ui.set_value(current_joint_idx)

        if joint_info and self.joint_control_slider:
            idx = joint_info['index']
            target_abs = joint_info['target_abs']
            actual_abs = joint_info['actual_abs']

            # 同步關節控制滑桿的值
            if abs(self.joint_control_slider.value - target_abs) > 1e-3:
                self.joint_control_slider.set_value(target_abs)

            error = target_abs - actual_abs
            if joint_info['mode'] == 'offset':
                offset = joint_info['offset']
                text = f"模式: 偏移 | Offset={offset:+.2f} | Target={target_abs:+.2f} | Actual={actual_abs:+.2f} | Err={error:+.2f}"
            else: # mode == 'absolute'
                text = f"模式: 絕對 | Target={target_abs:+.2f} | Actual={actual_abs:+.2f} | Err={error:+.2f}"
            self.status_labels['joint_info'].set_text(text)
        else:
            self.status_labels['joint_info'].set_text('') # 非相關模式下清空資訊

        # 更新 ONNX 標籤與日誌
        self._update_onnx_labels()
        log_content = "\n".join(log_queue)
        self.log_area.set_value(log_content)

    def _update_onnx_labels(self):
        """
        [UI 更新] 更新 ONNX 觀察向量的顯示標籤。
        從 `state` 中讀取最新的觀察數據，並根據當前 AI 策略的配方進行解析顯示。
        """
        # 獲取當前活躍策略的觀察配方和組件維度信息
        pm_ref = self.state.policy_manager_ref
        if not pm_ref or not pm_ref.get_active_recipe():
            # 如果 policy_manager_ref 為 None 或沒有活躍配方，則清空顯示
            for comp_name in self.onnx_input_labels:
                self.onnx_input_labels[comp_name].set_text(f'{comp_name}: N/A')
            return

        recipe = pm_ref.get_active_recipe()
        obs_builder = pm_ref.obs_builder # 獲取 ObservationBuilder 的參考
        component_dims = obs_builder.component_dims # 獲取觀察分量的維度信息

        obs_vec = self.state.latest_onnx_input # 從 state 獲取最新的 ONNX 輸入向量
        
        # 遍歷觀察配方，解析並顯示每個分量。
        current_idx = 0
        for comp_name_in_recipe in recipe:
            dim = component_dims.get(comp_name_in_recipe, 0) # 獲取該分量的維度
            if dim > 0:
                end_idx = current_idx + dim
                # 確保切片索引在向量範圍內，避免 IndexError。
                if end_idx <= len(obs_vec):
                    value_slice = obs_vec[current_idx:end_idx]
                    # 將向量格式化為字串，限制精度和行寬，並抑制小數。
                    vec_str = np.array2string(value_slice, precision=2, suppress_small=True, max_line_width=30)
                    # 更新對應的 UI 標籤。
                    if comp_name_in_recipe in self.onnx_input_labels:
                        self.onnx_input_labels[comp_name_in_recipe].set_text(f'{comp_name_in_recipe}: {vec_str}')
                else:
                    # 如果向量長度不足，顯示 N/A。
                    if comp_name_in_recipe in self.onnx_input_labels:
                        self.onnx_input_labels[comp_name_in_recipe].set_text(f'{comp_name_in_recipe}: N/A (數據不足)')
                current_idx = end_idx
            else:
                # 如果分量維度為 0 或不在 component_dims 中，顯示 N/A。
                if comp_name_in_recipe in self.onnx_input_labels:
                    self.onnx_input_labels[comp_name_in_recipe].set_text(f'{comp_name_in_recipe}: N/A (無定義)')

    def run(self):
        """
        [主入口] 啟動 NiceGUI 應用程式。
        這個方法通常由 `main_nicegui.py` 調用。
        """
        ui.run(title="Pupper Robot Console", port=8080)