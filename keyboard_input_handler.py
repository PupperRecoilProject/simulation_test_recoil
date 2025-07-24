# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """處理所有鍵盤輸入事件，並根據當前模式進行分派。"""
    def __init__(self, state: SimulationState, xbox_handler, terrain_manager):
        """初始化函式，儲存必要的物件參考。"""
        self.state = state
        self.config = state.config
        self.serial_comm_ref = state.serial_communicator_ref 
        self.xbox_handler = xbox_handler
        self.terrain_manager = terrain_manager
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']
        self.num_params = len(self.param_keys)

    def register_callbacks(self, window):
        """向 GLFW 註冊鍵盤事件的回呼函式。"""
        glfw.set_key_callback(window, self.key_callback) # 註冊按鍵事件
        glfw.set_char_callback(window, self.char_callback) # 註冊字元輸入事件

    def char_callback(self, window, codepoint):
        """處理可列印字元的輸入，專門用於序列埠模式。"""
        # 只有在序列埠模式下，才將輸入的字元附加到指令緩衝區
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        """【邏輯重構】處理所有按鍵事件，優先處理模式壁壘。"""

        # --- 只處理按下的事件，以簡化邏輯 ---
        if action == glfw.PRESS:
            # --- 【快捷鍵變更】使用 `~` 鍵作為序列埠控制台的唯一開關 ---
            # glfw.KEY_GRAVE_ACCENT 對應的就是 `~` 鍵
            if key == glfw.KEY_GRAVE_ACCENT:
                # 判斷當前模式並進行切換
                if self.state.control_mode == "SERIAL_MODE":
                    self.state.set_control_mode("WALKING") # 如果在序列埠模式，則退出到走路模式
                else:
                    self.state.set_control_mode("SERIAL_MODE") # 否則，進入序列埠模式
                return # 處理完畢，直接返回

            # --- 【模式壁壘】如果處於序列埠模式，則忽略所有其他功能鍵 ---
            if self.state.control_mode == "SERIAL_MODE":
                # 在此模式下，只有 Enter 和 Backspace 有特殊功能
                # （`~` 鍵已在上面處理，其他字元由 char_callback 處理）
                if key == glfw.KEY_ENTER:
                    self.state.serial_command_to_send = self.state.serial_command_buffer
                    self.state.serial_command_buffer = ""
                elif key == glfw.KEY_BACKSPACE:
                    self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]
                return # 立即返回，阻止後續快捷鍵

            # --- 全域快捷鍵 (任何非 SERIAL 模式下都有效) ---
            if key == glfw.KEY_SPACE: self.state.single_step_mode = not self.state.single_step_mode; print(f"\n--- SIMULATION {'PAUSED' if self.state.single_step_mode else 'PLAYING'} ---"); return
            if self.state.single_step_mode and key == glfw.KEY_N: self.state.execute_one_step = True; return
            if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
            if key == glfw.KEY_R: self.state.hard_reset_requested = True; return
            if key == glfw.KEY_X: self.state.soft_reset_requested = True; return
            if key == glfw.KEY_Y:
                if self.state.terrain_mode == "INFINITE": self.terrain_manager.regenerate_terrain_and_adjust_robot(self.state.latest_pos)
                else: print("⚠️ 'Y'鍵 (重生地形) 只在無限地形模式下有效。")
                return
            if key == glfw.KEY_V: self.terrain_manager.cycle_terrain_mode(self.state); return
            if key == glfw.KEY_P: self.terrain_manager.save_hfield_to_png(); return
            if key == glfw.KEY_TAB: self.state.display_page = (self.state.display_page + 1) % self.state.num_display_pages; return
            if key == glfw.KEY_M: self.state.toggle_input_mode("GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"); return
            if key == glfw.KEY_U: self.state.serial_is_connected = self.serial_comm_ref.scan_and_connect(); return
            if key == glfw.KEY_J: self.state.gamepad_is_connected = self.xbox_handler.scan_and_connect(); return
            
            # --- 模式切換快捷鍵 ---
            if key == glfw.KEY_F: self.state.set_control_mode("FLOATING" if self.state.control_mode == "WALKING" else "WALKING"); return
            if key == glfw.KEY_G: self.state.set_control_mode("JOINT_TEST" if self.state.control_mode != "JOINT_TEST" else "WALKING"); return
            if key == glfw.KEY_B: self.state.set_control_mode("MANUAL_CTRL" if self.state.control_mode != "MANUAL_CTRL" else "WALKING"); return
            if key == glfw.KEY_H: self.state.set_control_mode("HARDWARE_MODE" if self.state.control_mode != "HARDWARE_MODE" else "WALKING"); return
            # 'T' 鍵不再用於切換模式，它現在是一個普通字元

            # --- 硬體模式專用快捷鍵 ---
            if key == glfw.KEY_K:
                if self.state.control_mode == "HARDWARE_MODE" and self.state.hardware_controller_ref:
                    if self.state.hardware_ai_is_active: self.state.hardware_controller_ref.disable_ai()
                    else: self.state.hardware_controller_ref.enable_ai()
                else: print("Please enter Hardware Mode by pressing 'H' first.")
                return

            # --- 策略模型選擇快捷鍵 ---
            policy_keys = { glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3, glfw.KEY_5: 4, glfw.KEY_6: 5 }
            if key in policy_keys:
                target_index = policy_keys[key]
                if self.state.policy_manager_ref and self.state.available_policies:
                    if target_index < len(self.state.available_policies):
                        target_policy_name = self.state.available_policies[target_index]
                        self.state.policy_manager_ref.select_target_policy(target_policy_name)
                    else: print(f"⚠️ 警告: 策略索引 {target_index+1} 超出範圍。")
                return

        # --- 分模式處理 PRESS 和 REPEAT 事件 (例如長按) ---
        if action in [glfw.PRESS, glfw.REPEAT]:
            # 【修正】此處不再需要 SERIAL_MODE 的處理邏輯，已移到最前面
            
            if self.state.control_mode == "JOINT_TEST":
                if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index - 1) % 12
                elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index + 1) % 12
                elif key == glfw.KEY_UP: self.state.joint_test_offsets[self.state.joint_test_index] += 0.1
                elif key == glfw.KEY_DOWN: self.state.joint_test_offsets[self.state.joint_test_index] -= 0.1
                elif key == glfw.KEY_C and action == glfw.PRESS: self.state.joint_test_offsets.fill(0.0)
                
                if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                    final_command = self.state.sim.default_pose + self.state.joint_test_offsets
                    action_str = ' '.join(f"{a:.4f}" for a in final_command)
                    command_to_send = f"move all {action_str}\n"
                    hw_ser = self.state.hardware_controller_ref.ser
                    if hw_ser and hw_ser.is_open:
                        try: hw_ser.write(command_to_send.encode('utf-8'))
                        except Exception as e: print(f"❌ 關節測試模式發送指令失敗: {e}")
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
                return
        
            # --- 預設模式（WALKING, FLOATING）下的鍵盤控制 ---
            if self.state.input_mode != "KEYBOARD": return
            
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