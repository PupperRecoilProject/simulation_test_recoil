# xbox_input_handler.py
from state import SimulationState
from xbox_controller import XboxController # 現在會導入 pygame 版本的控制器

class XboxInputHandler:
    """
    處理 Xbox 搖桿的輸入，並將其轉換為對 SimulationState 的更新。
    """
    def __init__(self, state: SimulationState):
        """初始化 XboxInputHandler。"""
        self.state = state
        self.config = state.config
        self.controller = XboxController()
        self.last_input_state = self.controller.get_input() if self.is_available() else {}
    
    def is_available(self) -> bool:
        """檢查搖桿是否已成功初始化並連接。"""
        return self.controller.is_connected()

    def update_state(self):
        """從搖桿讀取輸入並更新 SimulationState。"""
        if not self.is_available():
            if self.state.input_mode == "GAMEPAD":
                print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                self.state.toggle_input_mode("KEYBOARD")
            return

        # *** 關鍵修改：先處理事件，再讀取狀態 ***
        self.controller.update() 
        current_input = self.controller.get_input()
        
        # 後續的邏輯完全不變
        self.state.command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy']
        self.state.command[1] = current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx'] * -1 # Y軸反向
        self.state.command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz']

        p_step, params = self.config.param_adjust_steps, self.state.tuning_params

        if current_input['button_select'] and not self.last_input_state.get('button_select', 0):
            self.state.reset_requested = True
            
        dpad_y = current_input['dpad'][1]
        last_dpad_y = self.last_input_state.get('dpad', (0,0))[1]
        if dpad_y == 1 and last_dpad_y != 1: params.kp += p_step['kp']
        if dpad_y == -1 and last_dpad_y != -1: params.kp -= p_step['kp']
        
        dpad_x = current_input['dpad'][0]
        last_dpad_x = self.last_input_state.get('dpad', (0,0))[0]
        if dpad_x == 1 and last_dpad_x != 1: params.kd += p_step['kd']
        if dpad_x == -1 and last_dpad_x != -1: params.kd -= p_step['kd']

        if current_input['button_r1'] and not self.last_input_state.get('button_r1', 0): params.action_scale += p_step['action_scale']
        if current_input['button_l1'] and not self.last_input_state.get('button_l1', 0): params.action_scale -= p_step['action_scale']
        if current_input['button_y'] and not self.last_input_state.get('button_y', 0): params.bias += p_step['bias']
        if current_input['button_a'] and not self.last_input_state.get('button_a', 0): params.bias -= p_step['bias']
        
        self.last_input_state = current_input
        
        params.kp = max(0, params.kp)
        params.kd = max(0, params.kd)
        params.action_scale = max(0, params.action_scale)

    def close(self):
        """關閉搖桿連接。"""
        if self.controller:
            self.controller.close()