try:
    import glfw
except ImportError:
    glfw = None
from state import SimulationState
from logger import log
from event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_TUNING_PARAM_ADJUSTED,
    EVENT_TUNING_PARAM_SELECT_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_UI_PAGE_CHANGE_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_COMMAND_UPDATED,
)


class KeyboardInputHandler:
    """處理所有鍵盤輸入事件，並根據當前模式進行分派。"""
    def __init__(self, state: SimulationState, xbox_handler, terrain_manager):
        """初始化函式，儲存必要的物件參考。"""
        self.state = state # 全域狀態的參考
        self.config = state.config # 設定檔的參考
        # [修改] 移除不再直接使用的 serial_comm_ref 和 xbox_handler
        self.terrain_manager = terrain_manager # 地形管理器的參考
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias'] # 可調參數的鍵名列表
        self.num_params = len(self.param_keys) # 可調參數的數量

    def register_callbacks(self, window):
        """向 GLFW 註冊鍵盤事件的回呼函式 (若可用)。"""
        if glfw is None:
            log.warning("glfw 模組不存在，無法註冊鍵盤事件")
            return
        glfw.set_key_callback(window, self.key_callback)  # 註冊按鍵事件
        glfw.set_char_callback(window, self.char_callback)  # 註冊字元輸入事件

    def char_callback(self, window, codepoint):
        """處理可列印字元的輸入，專門用於序列埠模式。"""
        if self.state.control_mode == "SERIAL_MODE":
            # 在序列埠模式下，將輸入的字元加入指令緩衝區
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        """處理所有按鍵事件，為所有專用模式建立壁壘。"""
        # 根據當前模式，分派給不同的處理函式
        if self.state.control_mode == "SERIAL_MODE":
            self.handle_serial_mode_keys(key, action)
            return

        if self.state.control_mode == "JOINT_TEST":
            self.handle_joint_test_mode_keys(key, action)
            return

        if self.state.control_mode == "MANUAL_CTRL":
            self.handle_manual_ctrl_mode_keys(key, action)
            return
        
        # 如果不在任何專用模式中，則執行通用和預設模式的按鍵處理
        # [修改] 將 window 參數傳遞下去
        self.handle_global_and_default_keys(window, key, action)

    def handle_serial_mode_keys(self, key, action):
        """專門處理序列埠模式下的按鍵。"""
        if action == glfw.PRESS:
            if key == glfw.KEY_GRAVE_ACCENT:
                # [修改] 發布模式切換請求，目標是返回上一個模式
                event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode=self.state.previous_control_mode)
                return
        
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_ENTER:
                # [修改] 發布序列埠指令發送請求
                # 注意：這裡依然需要直接讀取 serial_command_buffer，因為它是純UI狀態
                # 更徹底的方案是也將 buffer 內容通過事件傳遞
                event_bus.publish("request.serial_command_send", command=self.state.serial_command_buffer)
                self.state.serial_command_buffer = ""
            elif key == glfw.KEY_BACKSPACE:
                self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]

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
                # 只切換狀態旗標，由模擬執行緒處理實際啟用或停用懸浮
                self.state.manual_mode_is_floating = not self.state.manual_mode_is_floating
            elif key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS: self.state.manual_ctrl_index = (self.state.manual_ctrl_index - 1) % 12
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS: self.state.manual_ctrl_index = (self.state.manual_ctrl_index + 1) % 12
            elif key == glfw.KEY_UP: self.state.manual_final_ctrl[self.state.manual_ctrl_index] += 0.1
            elif key == glfw.KEY_DOWN: self.state.manual_final_ctrl[self.state.manual_ctrl_index] -= 0.1
            elif key == glfw.KEY_C and action == glfw.PRESS: self.state.manual_final_ctrl.fill(0.0)
            
    def handle_global_and_default_keys(self, window, key, action):
        """處理所有非專用模式下的全域快捷鍵和預設控制。"""
        # --- 只在按鍵按下的瞬間觸發一次的事件 ---
        if action == glfw.PRESS:
            # --- 模式切換 ---
            if key == glfw.KEY_GRAVE_ACCENT:
                event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="SERIAL_MODE")
                return

            # --- 全域快捷鍵 (發布事件) ---
            if key == glfw.KEY_ESCAPE:
                event_bus.publish(EVENT_SHUTDOWN_REQUESTED)
                return
            if key == glfw.KEY_R:
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")
                return
            if key == glfw.KEY_X:
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft")
                return
            if key == glfw.KEY_Y:
                # 這個邏輯與 terrain_manager 強耦合，暫時保留直接呼叫，但理想情況也應發布事件
                if self.state.terrain_mode == "INFINITE":
                    self.terrain_manager.regenerate_terrain_and_adjust_robot(self.state.latest_pos)
                else:
                    print("⚠️ 'Y'鍵 (重生地形) 只在無限地形模式下有效。")
                return
            if key == glfw.KEY_P:
                self.terrain_manager.save_hfield_to_png()
                return
            
            # --- UI/輸入 狀態切換 ---
            if key == glfw.KEY_TAB:
                event_bus.publish(EVENT_UI_PAGE_CHANGE_REQUESTED, direction=1)
                return
            if key == glfw.KEY_M:
                new_mode = "GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"
                event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode=new_mode)
                return
            
            # --- 設備連接請求 ---
            if key == glfw.KEY_U:
                event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial")
                return
            if key == glfw.KEY_J:
                event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad")
                return

        # --- 長按事件 (重複觸發) ---
        if action in [glfw.PRESS, glfw.REPEAT]:
            # --- 參數調整 ---
            # 選擇要調整的參數
            if key == glfw.KEY_LEFT_BRACKET:
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=-1)
            elif key == glfw.KEY_RIGHT_BRACKET:
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=1)
            # 調整選中參數的值
            elif key == glfw.KEY_UP:
                event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=1)
            elif key == glfw.KEY_DOWN:
                event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=-1)
            
            # --- 移動指令 (直接修改 state.command 是可接受的，因為有專門的 COMMAND_UPDATED 事件) ---
            # 為了與搖桿的事件驅動方式統一，我們也將其改為發布事件
            step = self.config.keyboard_velocity_adjust_step
            cmd = self.state.command.copy() # 取得當前指令作為基礎
            
            if key == glfw.KEY_W: cmd[1] += step
            elif key == glfw.KEY_S: cmd[1] -= step
            elif key == glfw.KEY_A: cmd[0] += step
            elif key == glfw.KEY_D: cmd[0] -= step
            elif key == glfw.KEY_Q: cmd[2] += step
            elif key == glfw.KEY_E: cmd[2] -= step
            elif key == glfw.KEY_C: cmd.fill(0.0)
            
            # 發布指令更新事件
            event_bus.publish(EVENT_COMMAND_UPDATED, command=cmd)
            
