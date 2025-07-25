# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """處理所有鍵盤輸入事件，並根據當前模式進行分派。"""
    def __init__(self, state: SimulationState, xbox_handler, terrain_manager):
        """初始化函式，儲存必要的物件參考。"""
        self.state = state # 全域狀態的參考
        self.config = state.config # 設定檔的參考
        self.serial_comm_ref = state.serial_communicator_ref # 序列埠通訊器的參考
        self.xbox_handler = xbox_handler # Xbox 搖桿處理器的參考
        self.terrain_manager = terrain_manager # 地形管理器的參考
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias'] # 可調參數的鍵名列表
        self.num_params = len(self.param_keys) # 可調參數的數量

    def register_callbacks(self, window):
        """向 GLFW 註冊鍵盤事件的回呼函式。"""
        glfw.set_key_callback(window, self.key_callback) # 註冊按鍵事件
        glfw.set_char_callback(window, self.char_callback) # 註冊字元輸入事件

    def char_callback(self, window, codepoint):
        """處理可列印字元的輸入，專門用於序列埠模式。"""
        if self.state.control_mode == "SERIAL_MODE": # 檢查是否處於序列埠模式
            self.state.serial_command_buffer += chr(codepoint) # 將輸入的字元附加到指令緩衝區

    def key_callback(self, window, key, scancode, action, mods):
        """【最終重構】處理所有按鍵事件，為所有專用模式建立壁壘。"""
        # --- 模式壁壘邏輯：根據當前模式，分派給不同的處理函式 ---
        if self.state.control_mode == "SERIAL_MODE": # 如果是序列埠模式
            self.handle_serial_mode_keys(key, action) # 交給序列埠模式處理器
            return # 處理完畢，立即返回

        if self.state.control_mode == "JOINT_TEST": # 如果是關節測試模式
            self.handle_joint_test_mode_keys(key, action) # 交給關節測試模式處理器
            return # 處理完畢，立即返回

        if self.state.control_mode == "MANUAL_CTRL": # 如果是手動控制模式
            self.handle_manual_ctrl_mode_keys(key, action) # 交給手動模式處理器
            return # 處理完畢，立即返回
        
        # --- 如果不在任何專用模式中，則執行通用和預設模式的按鍵處理 ---
        self.handle_global_and_default_keys(window, key, action)

    def handle_serial_mode_keys(self, key, action):
        """專門處理序列埠模式下的按鍵。"""
        if key == glfw.KEY_GRAVE_ACCENT and action == glfw.PRESS: # 如果按下 `~` 鍵
            # 【智慧退出】退出時，返回到進入此模式前的上一個模式
            self.state.set_control_mode(self.state.previous_control_mode)
            return
            
        if action in [glfw.PRESS, glfw.REPEAT]: # 處理按下或長按
            if key == glfw.KEY_ENTER: # 如果是 Enter 鍵
                self.state.serial_command_to_send = self.state.serial_command_buffer # 將緩衝區內容設為待發送
                self.state.serial_command_buffer = "" # 清空緩衝區
            elif key == glfw.KEY_BACKSPACE: # 如果是 Backspace 鍵
                self.state.serial_command_buffer = self.state.serial_command_buffer[:-1] # 刪除最後一個字元

    def handle_joint_test_mode_keys(self, key, action):
        """專門處理關節測試模式下的按鍵，只更新狀態，不發送指令。"""
        if action == glfw.PRESS and key == glfw.KEY_G: # 如果按下 'G' 鍵
            # 【模式切換修正】如果硬體控制器正在運行，則返回 HARDWARE_MODE，否則返回 WALKING
            if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                self.state.set_control_mode("HARDWARE_MODE")
            else:
                self.state.set_control_mode("WALKING")
            return
            
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index - 1) % 12
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS: self.state.joint_test_index = (self.state.joint_test_index + 1) % 12
            elif key == glfw.KEY_UP: self.state.joint_test_offsets[self.state.joint_test_index] += 0.1
            elif key == glfw.KEY_DOWN: self.state.joint_test_offsets[self.state.joint_test_index] -= 0.1
            elif key == glfw.KEY_C and action == glfw.PRESS: self.state.joint_test_offsets.fill(0.0)
            
            # 【核心修正】此處不再需要發送指令的邏輯，已統一由 HardwareController 處理

    def handle_manual_ctrl_mode_keys(self, key, action):
        """專門處理手動控制模式下的按鍵。"""
        if action == glfw.PRESS and key == glfw.KEY_G: # 如果按下 'G' 鍵
            self.state.set_control_mode("WALKING") # 退出到走路模式
            return
            
        if action in [glfw.PRESS, glfw.REPEAT]:
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
            
    def handle_global_and_default_keys(self, window, key, action):
        """處理所有非專用模式下的全域快捷鍵和預設控制。"""
        if action == glfw.PRESS:
            # 【智慧進入】按下 `~` 鍵進入序列埠模式。進入前的模式會被 state.set_control_mode 自動記錄
            if key == glfw.KEY_GRAVE_ACCENT:
                self.state.set_control_mode("SERIAL_MODE")
                return

            # --- 全域快捷鍵 ---
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
            
            # 模式切換等功能已移至 GUI，保留鍵盤僅做基礎操作

        # --- 長按事件 (重複觸發) ---
        if action in [glfw.PRESS, glfw.REPEAT]:
            if self.state.input_mode != "KEYBOARD": return
            
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
            
            # 移動指令
            step = self.config.keyboard_velocity_adjust_step
            if key == glfw.KEY_C: self.state.clear_command()
            elif key == glfw.KEY_W: self.state.command[1] += step
            elif key == glfw.KEY_S: self.state.command[1] -= step
            elif key == glfw.KEY_A: self.state.command[0] += step
            elif key == glfw.KEY_D: self.state.command[0] -= step
            elif key == glfw.KEY_Q: self.state.command[2] += step
            elif key == glfw.KEY_E: self.state.command[2] -= step
