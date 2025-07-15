# keyboard_input_handler.py
import glfw
from state import SimulationState

class KeyboardInputHandler:
    """
    處理所有鍵盤輸入事件，並更新 SimulationState。
    它只依賴於 SimulationState，不直接與 Simulation 物件交互。
    """
    def __init__(self, state: SimulationState):
        """
        初始化 KeyboardInputHandler。
        這個 handler 只依賴 state 物件，不依賴 Simulation。
        """
        self.state = state
        self.config = state.config

    def key_callback(self, window, key, scancode, action, mods):
        """GLFW 的鍵盤回調函式。"""
        if action != glfw.PRESS:
            return

        # --- 全局控制按鍵 (在任何模式下都有效) ---
        if key == glfw.KEY_ESCAPE: 
            glfw.set_window_should_close(window, 1)
            return
        if key == glfw.KEY_R: 
            self.state.reset_requested = True
            return
        if key == glfw.KEY_TAB: 
            self.state.display_page = (self.state.display_page + 1) % self.state.num_display_pages
            return
        if key == glfw.KEY_M: 
            self.state.toggle_input_mode("GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD")
            return
        
        # F 鍵需要傳入當前的位置和姿態來啟用平滑過渡或固定模式
        if key == glfw.KEY_F:
            # 從 state 中獲取由 main 迴圈在每一幀更新的最新資訊
            current_pos = self.state.latest_pos
            current_quat = self.state.latest_quat
            self.state.toggle_control_mode(current_pos, current_quat)
            return

        # 如果當前不是鍵盤模式，則忽略後續的專屬按鍵
        if self.state.input_mode != "KEYBOARD":
            return
        
        # --- 鍵盤模式專屬控制 ---
        step = self.config.keyboard_velocity_adjust_step
        if key == glfw.KEY_C: self.state.clear_command()
        elif key == glfw.KEY_W: self.state.command[1] += step
        elif key == glfw.KEY_S: self.state.command[1] -= step
        elif key == glfw.KEY_A: self.state.command[0] += step
        elif key == glfw.KEY_D: self.state.command[0] -= step
        elif key == glfw.KEY_Q: self.state.command[2] += step
        elif key == glfw.KEY_E: self.state.command[2] -= step

        # 參數調校
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

        # 確保參數不為負
        params.kp = max(0, params.kp)
        params.kd = max(0, params.kd)
        params.action_scale = max(0, params.action_scale)
        
        # 打印狀態以便即時除錯
        self._print_status()

    def _print_status(self):
        """打印目前的指令和調校參數。"""
        cmd, p = self.state.command, self.state.tuning_params
        print(f"[KB] Cmd(vy,vx,wz): [{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}] | "
              f"Kp:{p.kp:.1f}, Kd:{p.kd:.2f}, ActScl:{p.action_scale:.3f}, Bias:{p.bias:.1f}")