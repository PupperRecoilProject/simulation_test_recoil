# keyboard_input_handler.py
try:
    import glfw
except ImportError:
    glfw = None
from state import SimulationState
from logger import log
# [修改] 導入所有需要的事件
from event_system import (
    event_bus,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_TERRAIN_REGENERATE_REQUESTED,
    EVENT_TERRAIN_SNAPSHOT_REQUESTED,
    EVENT_TERRAIN_MODE_CHANGE_REQUESTED,
    EVENT_UI_PAGE_CHANGE_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
    EVENT_TUNING_PARAM_ADJUST_REQUESTED,
    EVENT_COMMAND_UPDATED,
)
import numpy as np # [新增] 用於指令清除

class KeyboardInputHandler:
    """[保留] 處理所有鍵盤輸入事件，並根據當前模式進行分派。"""
    def __init__(self, state: SimulationState, xbox_handler, terrain_manager):
        """[保留] 初始化函式，儲存必要的物件參考。"""
        self.state = state
        self.config = state.config
        self.serial_comm_ref = state.serial_communicator_ref
        self.xbox_handler = xbox_handler
        self.terrain_manager = terrain_manager
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']
        self.num_params = len(self.param_keys)

    def register_callbacks(self, window):
        """[保留] 向 GLFW 註冊鍵盤事件的回呼函式。"""
        if glfw is None:
            log.warning("glfw 模組不存在，無法註冊鍵盤事件")
            return
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

    def char_callback(self, window, codepoint):
        """
        [保留] 處理可列印字元的輸入，專門用於序列埠模式。
        這部分邏輯與核心狀態無關，屬於 UI 緩衝區操作，可保留。
        """
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        """[保留] 模式壁壘分派邏輯。"""
        if self.state.control_mode == "SERIAL_MODE":
            self.handle_serial_mode_keys(key, action)
            return

        # [刪除] JOINT_TEST 和 MANUAL_CTRL 的按鍵處理將與通用按鍵合併，
        # 因為它們的操作（選擇關節、調整值）現在是透過事件發布，
        # 而不是直接修改 state，因此無需特殊處理。
        
        self.handle_global_and_default_keys(window, key, action)

    def handle_serial_mode_keys(self, key, action):
        """
        [保留] 專門處理序列埠模式下的按鍵。
        這部分邏輯與核心狀態無關，可保留。
        """
        if key == glfw.KEY_GRAVE_ACCENT and action == glfw.PRESS:
            self.state.set_control_mode(self.state.previous_control_mode)
            return
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_ENTER:
                log.info(f"[UI > Serial]: {self.state.serial_command_buffer}")
                self.serial_comm_ref.send_command(self.state.serial_command_buffer)
                self.state.serial_command_buffer = ""
            elif key == glfw.KEY_BACKSPACE:
                self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]
    
    # [刪除] handle_joint_test_mode_keys 和 handle_manual_ctrl_mode_keys
    #        它們的功能將被合併到 handle_global_and_default_keys 中，並使用事件。

    # [重構] handle_global_and_default_keys，成為所有非 SERIAL 模式的事件發布中心
    def handle_global_and_default_keys(self, window, key, action):
        """處理所有非專用模式下的全域快捷鍵和預設控制。"""
        
        # --- 只在按下瞬間觸發的事件 ---
        if action == glfw.PRESS:
            # 模式切換請求
            key_to_mode = {
                glfw.KEY_GRAVE_ACCENT: "SERIAL_MODE",
                glfw.KEY_F: "FLOATING",
                glfw.KEY_G: "JOINT_TEST", # 兼做退出
                glfw.KEY_B: "MANUAL_CTRL",
                glfw.KEY_H: "HARDWARE_MODE",
            }
            if key in key_to_mode:
                # 特殊處理 'G' 鍵的退出邏輯
                if key == glfw.KEY_G and self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                        event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE")
                    else:
                        event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING")
                else:
                    event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode=key_to_mode[key])
                return

            # AI 策略模型切換
            key_to_policy = {
                glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3
            }
            if key in key_to_policy and key_to_policy[key] < len(self.state.available_policies):
                policy_name = self.state.available_policies[key_to_policy[key]]
                event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=policy_name)
                return

            # 全域快捷鍵
            if key == glfw.KEY_SPACE: self.state.single_step_mode = not self.state.single_step_mode; print(f"\n--- SIMULATION {'PAUSED' if self.state.single_step_mode else 'PLAYING'} ---"); return
            if self.state.single_step_mode and key == glfw.KEY_N: self.state.execute_one_step = True; return
            if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
            if key == glfw.KEY_R: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard"); return
            if key == glfw.KEY_X: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft"); return
            if key == glfw.KEY_Y: event_bus.publish(EVENT_TERRAIN_REGENERATE_REQUESTED); return
            if key == glfw.KEY_P: event_bus.publish(EVENT_TERRAIN_SNAPSHOT_REQUESTED); return
            if key == glfw.KEY_TAB: event_bus.publish(EVENT_UI_PAGE_CHANGE_REQUESTED, direction=1); return
            if key == glfw.KEY_M: event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"); return
            if key == glfw.KEY_U: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial"); return
            if key == glfw.KEY_J: event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad"); return
            if key == glfw.KEY_K: event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED); return
            if key == glfw.KEY_V: event_bus.publish(EVENT_TERRAIN_MODE_CHANGE_REQUESTED, direction=1); return

        # --- 長按事件 (重複觸發) ---
        if action in [glfw.PRESS, glfw.REPEAT]:
            # --- 運動指令 ---
            step = self.config.keyboard_velocity_adjust_step
            # [修改] 使用 command_buffer 避免每次按鍵都 publish，只在有變化時 publish
            cmd_changed = False
            current_command = self.state.command.copy()
            if key == glfw.KEY_W: current_command[1] += step; cmd_changed = True
            elif key == glfw.KEY_S: current_command[1] -= step; cmd_changed = True
            elif key == glfw.KEY_A: current_command[0] += step; cmd_changed = True
            elif key == glfw.KEY_D: current_command[0] -= step; cmd_changed = True
            elif key == glfw.KEY_Q: current_command[2] += step; cmd_changed = True
            elif key == glfw.KEY_E: current_command[2] -= step; cmd_changed = True
            elif key == glfw.KEY_C and action == glfw.PRESS: # 清除指令只在按下時觸發
                current_command.fill(0.0); cmd_changed = True
            
            if cmd_changed:
                event_bus.publish(EVENT_COMMAND_UPDATED, command=current_command)
                return

            # --- 參數/關節調整 ---
            # 選擇索引
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, direction=-1)
                else:
                    with self.state.lock: # 讀取 UI 狀態仍需加鎖
                        self.state.tuning_param_index = (self.state.tuning_param_index - 1) % self.num_params
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, direction=1)
                else:
                    with self.state.lock:
                        self.state.tuning_param_index = (self.state.tuning_param_index + 1) % self.num_params
            # 調整值
            elif key == glfw.KEY_UP or key == glfw.KEY_DOWN:
                direction = 1 if key == glfw.KEY_UP else -1
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=direction, step=0.1)
                else: # 參數調整模式
                    param_to_adjust = self.param_keys[self.state.tuning_param_index]
                    event_bus.publish(EVENT_TUNING_PARAM_ADJUST_REQUESTED, param_name=param_to_adjust, direction=direction)
            # 清除關節偏移
            elif key == glfw.KEY_C and action == glfw.PRESS and self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, clear=True)