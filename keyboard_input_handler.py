# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """
    處理所有鍵盤輸入事件，並根據當前模式進行分派。
    """
    def __init__(self, state: SimulationState, serial_comm, xbox_handler, terrain_manager):
        self.state = state
        self.config = state.config
        self.serial_comm = serial_comm
        self.xbox_handler = xbox_handler
        self.terrain_manager = terrain_manager
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']
        self.num_params = len(self.param_keys)

    def register_callbacks(self, window):
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

    def char_callback(self, window, codepoint):
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE: self.state.single_step_mode = not self.state.single_step_mode; print(f"\n--- SIMULATION {'PAUSED' if self.state.single_step_mode else 'PLAYING'} ---"); return
            if self.state.single_step_mode and key == glfw.KEY_N: self.state.execute_one_step = True; return
            if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
            
            if key == glfw.KEY_R: self.state.hard_reset_requested = True; return
            if key == glfw.KEY_X: self.state.soft_reset_requested = True; return
            
            # 【修改】'Y'鍵只在無限模式下觸發地形重置
            if key == glfw.KEY_Y:
                if self.state.terrain_mode == "INFINITE":
                    self.terrain_manager.regenerate_terrain_and_adjust_robot(self.state.latest_pos)
                else:
                    print("⚠️ 'Y'鍵 (重生地形) 只在無限地形模式下有效。")
                return
            
            # 【新增】'V'鍵觸發地形模式循環
            if key == glfw.KEY_V:
                self.terrain_manager.cycle_terrain_mode(self.state)
                return

            if key == glfw.KEY_P: self.terrain_manager.save_hfield_to_png(); return

            if key == glfw.KEY_TAB: self.state.display_page = (self.state.display_page + 1) % self.state.num_display_pages; return
            if key == glfw.KEY_M: self.state.toggle_input_mode("GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"); return
            
            if key == glfw.KEY_U: self.state.serial_is_connected = self.serial_comm.scan_and_connect(); return
            if key == glfw.KEY_J: self.state.gamepad_is_connected = self.xbox_handler.scan_and_connect(); return
            
            if key == glfw.KEY_H:
                new_mode = "HARDWARE_MODE" if self.state.control_mode != "HARDWARE_MODE" else "WALKING"
                self.state.set_control_mode(new_mode)
                return
            if key == glfw.KEY_K:
                if self.state.control_mode == "HARDWARE_MODE" and self.state.hardware_controller_ref:
                    if self.state.hardware_ai_is_active:
                        self.state.hardware_controller_ref.disable_ai()
                    else:
                        self.state.hardware_controller_ref.enable_ai()
                else: print("請先按 'H' 進入硬體模式。")
                return

            policy_keys = {
                glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3,
                glfw.KEY_5: 4, glfw.KEY_6: 5,
            }
            if key in policy_keys:
                target_index = policy_keys[key]
                if self.state.policy_manager_ref and self.state.available_policies:
                    if target_index < len(self.state.available_policies):
                        target_policy_name = self.state.available_policies[target_index]
                        self.state.policy_manager_ref.select_target_policy(target_policy_name)
                    else:
                        print(f"⚠️ 警告: 策略索引 {target_index+1} 超出範圍。")
                return

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

        if action == glfw.PRESS:
            if key == glfw.KEY_F: self.state.set_control_mode("FLOATING" if self.state.control_mode == "WALKING" else "WALKING"); return
            if key == glfw.KEY_T: self.state.set_control_mode("SERIAL_MODE"); return
            if key == glfw.KEY_G: self.state.set_control_mode("JOINT_TEST"); return
            if key == glfw.KEY_B: self.state.set_control_mode("MANUAL_CTRL"); return
        
        if self.state.input_mode != "KEYBOARD": return
            
        if action in [glfw.PRESS, glfw.REPEAT]:
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

            step = self.config.keyboard_velocity_adjust_step
            if key == glfw.KEY_C: self.state.clear_command()
            elif key == glfw.KEY_W: self.state.command[1] += step
            elif key == glfw.KEY_S: self.state.command[1] -= step
            elif key == glfw.KEY_A: self.state.command[0] += step
            elif key == glfw.KEY_D: self.state.command[0] -= step
            elif key == glfw.KEY_Q: self.state.command[2] += step
            elif key == glfw.KEY_E: self.state.command[2] -= step