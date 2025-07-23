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
        glfw.set_key_callback(window, self.key_callback) # 註冊按鍵回呼函式
        glfw.set_char_callback(window, self.char_callback) # 註冊字元回呼函式

    def char_callback(self, window, codepoint):
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint) # 如果在序列埠模式，將輸入的字元加入緩衝區

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
            if key == glfw.KEY_V: 
                self.terrain_manager.cycle_terrain() # 循環切換地形
                return
            
            # 設備掃描
            if key == glfw.KEY_U: self.state.serial_is_connected = self.serial_comm.scan_and_connect(); return
            if key == glfw.KEY_J: self.state.gamepad_is_connected = self.xbox_handler.scan_and_connect(); return
            
            # 硬體模式相關按鍵
            if key == glfw.KEY_H: # 'H' for Hardware
                new_mode = "HARDWARE_MODE" if self.state.control_mode != "HARDWARE_MODE" else "WALKING"
                self.state.set_control_mode(new_mode)
                return
            if key == glfw.KEY_K: # 'K' for Kill-switch/Activate
                if self.state.control_mode == "HARDWARE_MODE" and self.state.hardware_controller_ref:
                    if self.state.hardware_ai_is_active:
                        self.state.hardware_controller_ref.disable_ai()
                    else:
                        self.state.hardware_controller_ref.enable_ai()
                else: print("請先按 'H' 進入硬體模式。")
                return

            # --- 【新】使用數字鍵選擇性切換策略模型 ---
            # 建立一個從 GLFW 按鍵碼到策略索引的映射
            policy_keys = {
                glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3,
                glfw.KEY_5: 4, glfw.KEY_6: 5, # 可依需求擴充
            }
            if key in policy_keys:
                target_index = policy_keys[key] # 獲取按鍵對應的索引
                if self.state.policy_manager_ref and self.state.available_policies:
                    # 檢查索引是否在可用策略的範圍內
                    if target_index < len(self.state.available_policies):
                        target_policy_name = self.state.available_policies[target_index] # 獲取目標策略的名稱
                        # 呼叫新的選擇性切換函式
                        self.state.policy_manager_ref.select_target_policy(target_policy_name)
                    else:
                        print(f"⚠️ 警告: 策略索引 {target_index+1} 超出範圍。")
                return

        # --- 2. 可重複觸發的模式特定功能 ---
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