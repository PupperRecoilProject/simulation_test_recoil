# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """
    處理所有鍵盤輸入事件，包括對序列埠和關節測試模式的特殊處理。
    """
    def __init__(self, state: SimulationState):
        self.state = state
        self.config = state.config

    def register_callbacks(self, window):
        """註冊所有需要的 GLFW 回調。"""
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

    def char_callback(self, window, codepoint):
        """處理字元輸入，用於在 SERIAL_MODE 下建立指令。"""
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        """處理按鍵事件，根據不同模式分派邏輯。"""
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.state.single_step_mode = not self.state.single_step_mode
                status = "PAUSED (Press N for next step)" if self.state.single_step_mode else "PLAYING"
                print(f"\n--- SIMULATION {status} ---")
                return

            if self.state.single_step_mode and key == glfw.KEY_N:
                self.state.execute_one_step = True
                return

        if self.state.control_mode == "SERIAL_MODE":
            if action == glfw.PRESS or action == glfw.REPEAT:
                if key == glfw.KEY_ENTER:
                    self.state.serial_command_to_send = self.state.serial_command_buffer
                    self.state.serial_command_buffer = ""
                elif key == glfw.KEY_BACKSPACE:
                    self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]
                elif key == glfw.KEY_T and action == glfw.PRESS:
                    self.state.set_control_mode("WALKING")
            return

        if self.state.control_mode == "JOINT_TEST":
            if action == glfw.PRESS:
                if key == glfw.KEY_1: self.state.joint_test_index = (self.state.joint_test_index - 1) % 12
                elif key == glfw.KEY_2: self.state.joint_test_index = (self.state.joint_test_index + 1) % 12
                elif key == glfw.KEY_UP: self.state.joint_test_offsets[self.state.joint_test_index] += 0.1
                elif key == glfw.KEY_DOWN: self.state.joint_test_offsets[self.state.joint_test_index] -= 0.1
                elif key == glfw.KEY_C: self.state.joint_test_offsets.fill(0.0)
                elif key == glfw.KEY_G: self.state.set_control_mode("WALKING")
            return

        if action != glfw.PRESS: return

        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
        if key == glfw.KEY_R: self.state.hard_reset_requested = True; return
        if key == glfw.KEY_TAB: self.state.display_page = (self.state.display_page + 1) % self.state.num_display_pages; return
        if key == glfw.KEY_M: self.state.toggle_input_mode("GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"); return
        
        # --- 新增：X 鍵觸發軟重置 ---
        if key == glfw.KEY_X:
            self.state.soft_reset_requested = True
            return
        
        if key == glfw.KEY_F:
            new_mode = "FLOATING" if self.state.control_mode == "WALKING" else "WALKING"
            self.state.set_control_mode(new_mode)
            return
        if key == glfw.KEY_T: self.state.set_control_mode("SERIAL_MODE"); return
        if key == glfw.KEY_G: self.state.set_control_mode("JOINT_TEST"); return

        if self.state.input_mode != "KEYBOARD": return
        
        step = self.config.keyboard_velocity_adjust_step
        if key == glfw.KEY_C: self.state.clear_command()
        elif key == glfw.KEY_W: self.state.command[1] += step
        elif key == glfw.KEY_S: self.state.command[1] -= step
        elif key == glfw.KEY_A: self.state.command[0] += step
        elif key == glfw.KEY_D: self.state.command[0] -= step
        elif key == glfw.KEY_Q: self.state.command[2] += step
        elif key == glfw.KEY_E: self.state.command[2] -= step

        params = self.state.tuning_params
        p_step = self.config.param_adjust_steps
        if key == glfw.KEY_I: params.kp += p_step['kp']
        elif key == glfw.KEY_K: params.kp -= p_step['kp']
        elif key == glfw.KEY_L: params.kd += p_step['kd']
        elif key == glfw.KEY_J: params.kd -= p_step['kd']
        elif key == glfw.KEY_Y: params.action_scale += p_step['action_scale']
        elif key == glfw.KEY_H: params.action_scale -= p_step['action_scale']
        elif key == glfw.KEY_P: params.bias += p_step['bias']
        elif key == glfw.KEY_SEMICOLON: params.bias -= p_step['bias']
        
        params.kp = max(0, params.kp)
        params.kd = max(0, params.kd)
        params.action_scale = max(0, params.action_scale)