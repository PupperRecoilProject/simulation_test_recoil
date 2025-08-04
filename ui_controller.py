# ui_controller.py
from nicegui import ui, app
import numpy as np
import threading
from typing import TYPE_CHECKING, List
from logger import log, log_queue

# [修改] 導入所有需要的事件
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

if TYPE_CHECKING:
    from state import SimulationState

class UIController:
    """[保留] 管理 NiceGUI 介面與互動邏輯。"""
    def __init__(self, state: 'SimulationState'):
        # [保留] 初始化邏輯
        self.state = state
        # [刪除] 以下引用不再需要由 UIController 直接調用
        # self.policy_manager = state.policy_manager_ref
        # self.hardware_controller = state.hardware_controller_ref
        # self.serial_comm = state.serial_communicator_ref
        # self.xbox_handler = state.xbox_handler_ref

        self.status_labels = {}
        self.param_sliders = {}
        self.onnx_input_labels = {}
        self.log_area = None
        self.serial_command_buffer = None
        self.joint_control_slider = None

        if self.state.terrain_mode == 'SINGLE':
            self.ui_terrain_selection = self.state.terrain_manager_ref.single_terrain_names[self.state.single_terrain_index]
        else:
            self.ui_terrain_selection = 'INFINITE'
            
        self._setup_ui()

    def _setup_ui(self):
        """[保留] UI 佈局的整體結構不變。"""
        # ... (此處佈局程式碼與原版相同，僅修改回呼函式) ...
        # 我將只展示被修改的回呼函式部分
        pass # 實際程式碼未省略

    def _create_main_control_panel(self):
        with ui.card():
            ui.label('模式控制 (Control Mode)').classes('text-lg')
            with ui.row():
                # [保留] 模式切換已是事件驅動
                ui.button('走路 (Walking)', on_click=lambda: event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING"))
                # ... 其他模式按鈕
            
            ui.separator()
            ui.label('重置').classes('text-lg')
            with ui.row():
                # [保留] 重置已是事件驅動
                ui.button('軟重置 (X)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft"))
                ui.button('硬重置 (R)', on_click=lambda: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"))

    def _create_tuning_panel(self):
        with ui.card().classes('w-full'):
            ui.label('參數調整 (Tuning)').classes('text-lg')
            p_keys = {'kp': (0, 50), 'kd': (0, 5), 'action_scale': (0, 2), 'bias': (-20, 20)}
            for key, (min_val, max_val) in p_keys.items():
                with ui.row().classes('w-full items-center'):
                    ui.label(key.upper()).classes('w-20')
                    # [修改] 移除 bind_value，改用 on_change 發布事件
                    slider = ui.slider(min=min_val, max=max_val, step=0.01,
                                       on_change=lambda e, k=key: event_bus.publish(EVENT_TUNING_PARAM_ADJUST_REQUESTED, param_name=k, value=e.value))\
                                       .bind_value_from(self.state.tuning_params, key)\
                                       .classes('w-48')
                    ui.label().bind_text_from(self.state.tuning_params, key, lambda v: f'{v:.2f}')
            
            ui.separator()
            ui.label('策略選擇 (Policy)').classes('text-lg')
            # [修改] on_change 發布事件
            ui.select(
                options=self.state.available_policies,
                label='Active Policy',
                on_change=lambda e: event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=e.value)
            ).bind_value_from(self.state.policy_manager_ref, 'primary_policy_name').classes('w-full')
            
            terrain_options = ['INFINITE'] + self.state.terrain_manager_ref.single_terrain_names
            # [修改] on_change 發布事件
            ui.select(
                options=terrain_options,
                label='Terrain Mode',
                on_change=self._on_terrain_change_publish_event
            ).bind_value(self, 'ui_terrain_selection').classes('w-full')

    def _create_device_panel(self):
        with ui.card():
            ui.label('硬體 AI 控制').classes('text-lg')
            # [保留] AI 切換已是事件驅動
            ui.button('啟用/停用 AI (K)', on_click=lambda: event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED)).bind_enabled_from(
                self.state, 'control_mode', lambda mode: mode == "HARDWARE_MODE")

            ui.separator()
            ui.label('設備連接').classes('text-lg')
            with ui.row():
                # [修改] 連接按鈕發布事件
                ui.button('連接序列埠 (U)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial"))
                ui.button('連接搖桿 (J)', on_click=lambda: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad"))

            ui.separator()
            ui.label('系統').classes('text-lg')
            # [保留] 退出按鈕已是事件驅動
            ui.button('退出程式', on_click=lambda: event_bus.publish(EVENT_SHUTDOWN_REQUESTED), color='red')

    def _create_joint_control_panel(self):
        with ui.card().bind_visibility_from(self.state, 'control_mode', lambda m: m in ["JOINT_TEST", "MANUAL_CTRL"]).classes('w-full'):
            ui.label('關節微調 (Joint Fine-Tuning)').classes('text-lg')
            with ui.row().classes('items-center'):
                ui.label('啟用懸浮')
                # [修改] 開關發布事件
                ui.switch(on_change=lambda e: event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED, value=e.value))\
                  .bind_value(self.state, 'manual_mode_is_floating')
            
            joint_names = {i: name for i, name in enumerate([
                "FR_Abduction", "FR_Hip", "FR_Knee", "FL_Abduction", "FL_Hip", "FL_Knee",
                "RR_Abduction", "RR_Hip", "RR_Knee", "RL_Abduction", "RL_Hip", "RL_Knee"
            ])}
            
            # [修改] 選擇器發布事件
            ui.select(
                joint_names, label='選擇關節',
                on_change=lambda e: event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, index=e.value)
            ).bind_value_from(self.state, 'joint_test_index') # 仍綁定以顯示當前值

            self.status_labels['joint_info'] = ui.label('')

            # [修改] 滑桿發布事件
            self.joint_control_slider = ui.slider(min=-np.pi, max=np.pi, step=0.01,
                on_change=lambda e: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, value=e.value))\
                .props('label-always')

            with ui.row():
                # [修改] 按鈕發布事件
                ui.button('-0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=-1, step=0.1)).props('dense')
                ui.button('+0.1', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=1, step=0.1)).props('dense')
                ui.button('歸零 (Clear)', on_click=lambda: event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, clear=True)).props('dense')

    # [重構] _update_command_from_joystick，現在只發布事件
    def _update_command_from_joystick(self, event):
        x_val = -event.y
        y_val = event.x
        # [修改] 不再切換輸入模式，只發布指令。模式切換由使用者明確點擊按鈕完成。
        new_command = np.zeros(3, dtype=np.float32)
        new_command[0] = y_val * self.state.config.gamepad_sensitivity['vy']
        new_command[1] = x_val * self.state.config.gamepad_sensitivity['vx']
        event_bus.publish(EVENT_COMMAND_UPDATED, command=new_command)

    # [重構] _on_joystick_end，現在只發布清除指令的事件
    def _on_joystick_end(self, event):
        event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(3, dtype=np.float32))

    # [新增] 用於地形選擇器的新回呼函式
    def _on_terrain_change_publish_event(self, event):
        if event.value is not None:
            event_bus.publish(EVENT_TERRAIN_MODE_CHANGE_REQUESTED, mode_name=event.value)

    # [刪除] 所有舊的、直接操作 state 的方法，如 _set_joint_index, _on_joint_slider_change,
    #        _adjust_joint_value, _on_manual_float_toggle, _on_terrain_change 等。
    #        它們的功能已經被上面匿名函式中的事件發布所取代。

    # [保留] UI 更新和日誌顯示的邏輯，它們是 UI 的核心職責
    def update_ui_elements(self):
        # ... (此處程式碼與原版類似，僅做必要的調整以適應綁定變化) ...
        # 例如，滑桿的值現在從 state 中讀取，而不是由 UI 直接控制
        if self.joint_control_slider:
             with self.state.lock:
                if self.state.control_mode == "JOINT_TEST":
                    target_abs = self.state.sim.default_pose[self.state.joint_test_index] + self.state.joint_test_offsets[self.state.joint_test_index]
                elif self.state.control_mode == "MANUAL_CTRL":
                    target_abs = self.state.manual_final_ctrl[self.state.manual_ctrl_index]
                else:
                    target_abs = 0

             if abs(self.joint_control_slider.value - target_abs) > 1e-3:
                self.joint_control_slider.set_value(target_abs)
        # ... (其他 UI 更新邏輯) ...
        pass # 實際程式碼未省略
    
    def _send_serial_command(self):
        # [保留] 此功能屬於 UI 自身邏輯，不影響核心狀態
        command_text = self.serial_command_buffer.value
        if self.state.serial_communicator_ref and command_text:
            self.state.serial_communicator_ref.send_command(command_text)
            self.serial_command_buffer.set_value('')
            log.info(f"> {command_text}")