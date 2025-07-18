# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """
    處理所有鍵盤輸入事件，並根據當前模式進行分派。
    """
    # 【修改】接收 terrain_manager
    def __init__(self, state: SimulationState, serial_comm, xbox_handler, terrain_manager):
        self.state = state
        self.config = state.config
        self.serial_comm = serial_comm
        self.xbox_handler = xbox_handler
        self.terrain_manager = terrain_manager # <-- 【新增】
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']
        self.num_params = len(self.param_keys)

    def register_callbacks(self, window):
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

    def char_callback(self, window, codepoint):
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        # --- 1. 只在按鍵按下時觸發的通用功能 (模式切換、重置等) ---
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE: self.state.single_step_mode = not self.state.single_step_mode; print(f"\n--- SIMULATION {'PAUSED' if self.state.single_step_mode else 'PLAYING'} ---"); return
            if self.state.single_step_mode and key == glfw.KEY_N: self.state.execute_one_step = True; return
            if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
            if key == glfw.KEY_R: self.state.hard_reset_requested = True; return
            if key == glfw.KEY_X: self.state.soft_reset_requested = True; return
            if key == glfw.KEY_TAB: self.state.display_page = (self.state.display_page + 1) % self.state.num_display_pages; return
            if key == glfw.KEY_M: self.state.toggle_input_mode("GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"); return
            # 【新增】V 鍵切換地形
            if key == glfw.KEY_V: 
                self.terrain_manager.cycle_terrain()
                return
            
            # 設備掃描
            if key == glfw.KEY_U: self.state.serial_is_connected = self.serial_comm.scan_and_connect(); return
            if key == glfw.KEY_J: self.state.gamepad_is_connected = self.xbox_handler.scan_and_connect(); return

        # --- 2. 可重複觸發的模式特定功能 ---
        # 這是最關鍵的修改：優先處理特定模式的按鍵，如果處理了就 return
        if action in [glfw.PRESS, glfw.REPEAT]:
            if self.state.control_mode == "SERIAL_MODE":
                if key == glfw.KEY_ENTER: self.state.serial_command_to_send = self.state.serial_command_buffer; self.state.serial_command_buffer = ""
                elif key == glfw.KEY_BACKSPACE: self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]
                elif key == glfw.KEY_T and action == glfw.PRESS: self.state.set_control_mode("WALKING")
                return

            if self.state.control_mode == "JOINT_TEST":
                if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index - 1) % 12
                elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index + 1) % 12
                elif key == glfw.KEY_UP: self.state.joint_test_offsets[self.state.joint_test_index] += 0.1
                elif key == glfw.KEY_DOWN: self.state.joint_test_offsets[self.state.joint_test_index] -= 0.1
                elif key == glfw.KEY_C and action == glfw.PRESS: self.state.joint_test_offsets.fill(0.0)
                elif key == glfw.KEY_G and action == glfw.PRESS: self.state.set_control_mode("WALKING")
                return

            if self.state.control_mode == "MANUAL_CTRL":
                if key == glfw.KEY_F and action == glfw.PRESS:
                    self.state.manual_mode_is_floating = not self.state.manual_mode_is_floating
                    is_floating = self.state.manual_mode_is_floating
                    if is_floating:
                        if self.state.floating_controller_ref: self.state.floating_controller_ref.enable(self.state.latest_pos)
                    else:
                        if self.state.floating_controller_ref: self.state.floating_controller_ref.disable()
                elif key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS: self.state.manual_ctrl_index = (self.state.manual_ctrl_index - 1) % 12
                elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS: self.state.manual_ctrl_index = (self.state.manual_ctrl_index + 1) % 12
                elif key == glfw.KEY_UP: self.state.manual_final_ctrl[self.state.manual_ctrl_index] += 0.1
                elif key == glfw.KEY_DOWN: self.state.manual_final_ctrl[self.state.manual_ctrl_index] -= 0.1
                elif key == glfw.KEY_C and action == glfw.PRESS: self.state.manual_final_ctrl.fill(0.0)
                elif key == glfw.KEY_G and action == glfw.PRESS: self.state.set_control_mode("WALKING")
                return

        # --- 3. 如果以上模式都不是，則執行 WALKING/FLOATING 模式的預設按鍵邏輯 ---
        if action == glfw.PRESS:
            if key == glfw.KEY_F: self.state.set_control_mode("FLOATING" if self.state.control_mode == "WALKING" else "WALKING"); return
            if key == glfw.KEY_T: self.state.set_control_mode("SERIAL_MODE"); return
            if key == glfw.KEY_G: self.state.set_control_mode("JOINT_TEST"); return
            if key == glfw.KEY_B: self.state.set_control_mode("MANUAL_CTRL"); return
        
        if self.state.input_mode != "KEYBOARD": return
            
        if action in [glfw.PRESS, glfw.REPEAT]:
            # 參數調整
            if key == glfw.KEY_LEFT_BRACKET: self.state.tuning_param_index = (self.state.tuning_param_index - 1) % self.num_params
            elif key == glfw.KEY_RIGHT_BRACKET: self.state.tuning_param_index = (self.state.tuning_param_index + 1) % self.num_params
            elif key == glfw.KEY_UP or key == glfw.KEY_DOWN:
                param_to_adjust = self.param_keys[self.state.tuning_param_index]
                step = self.config.param_adjust_steps.get(param_to_adjust, 0.1)
                current_value = getattr(self.state.tuning_params, param_to_adjust)
                direction = 1 if key == glfw.KEY_UP else -1
                setattr(self.state.tuning_params, param_to_adjust, current_value + step * direction)
                
                self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
                self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
                self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)

            # 移動控制
            step = self.config.keyboard_velocity_adjust_step
            if key == glfw.KEY_C: self.state.clear_command()
            elif key == glfw.KEY_W: self.state.command[1] += step
            elif key == glfw.KEY_S: self.state.command[1] -= step
            elif key == glfw.KEY_A: self.state.command[0] += step
            elif key == glfw.KEY_D: self.state.command[0] -= step
            elif key == glfw.KEY_Q: self.state.command[2] += step
            elif key == glfw.KEY_E: self.state.command[2] -= step