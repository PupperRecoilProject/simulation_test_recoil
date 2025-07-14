import glfw
from state import SimulationState

class InputHandler:
    """
    處理所有鍵盤輸入事件，並更新 SimulationState。
    """
    def __init__(self, state: SimulationState):
        """
        初始化 InputHandler。

        Args:
            state (SimulationState): 應用程式的中央狀態物件。
        """
        self.state = state
        self.config = state.config # 方便存取設定

    def key_callback(self, window, key, scancode, action, mods):
        """
        GLFW 的鍵盤回調函式。
        這個函式會被註冊到 GLFW，並在鍵盤事件發生時被呼叫。
        """
        if action != glfw.PRESS:
            return

        # --- 程式控制 ---
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, 1)
            return
        if key == glfw.KEY_R:
            self.state.reset_requested = True
            print("重置請求已發送。")
            return
        if key == glfw.KEY_C:
            self.state.clear_command()
            return

        # --- 運動指令調整 (vx, vy, wz) ---
        step = self.config.velocity_adjust_step
        # 順序: [vy (側向), vx (前向), wz (偏航)]
        if key == glfw.KEY_W: self.state.command[1] += step
        elif key == glfw.KEY_S: self.state.command[1] -= step
        elif key == glfw.KEY_A: self.state.command[0] += step
        elif key == glfw.KEY_D: self.state.command[0] -= step
        elif key == glfw.KEY_Q: self.state.command[2] += step
        elif key == glfw.KEY_E: self.state.command[2] -= step

        # --- 參數即時調校 ---
        params = self.state.tuning_params
        if key == glfw.KEY_I: params.kp += 5.0
        elif key == glfw.KEY_K: params.kp -= 5.0
        elif key == glfw.KEY_L: params.kd += 0.1
        elif key == glfw.KEY_J: params.kd -= 0.1
        elif key == glfw.KEY_Y: params.action_scale += 0.05
        elif key == glfw.KEY_H: params.action_scale -= 0.05
        elif key == glfw.KEY_P: params.bias += 5.0
        elif key == glfw.KEY_SEMICOLON: params.bias -= 5.0

        # 確保參數值在合理範圍
        params.kp = max(0, params.kp)
        params.kd = max(0, params.kd)
        params.action_scale = max(0, params.action_scale)

        # 打印當前狀態以便除錯
        self._print_status()

    def _print_status(self):
        """打印目前的指令和調校參數。"""
        cmd = self.state.command
        p = self.state.tuning_params
        print(f"Cmd (vy, vx, wz): [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}] | "
              f"Kp: {p.kp:.1f}, Kd: {p.kd:.2f}, ActScl: {p.action_scale:.3f}, Bias: {p.bias:.1f}")