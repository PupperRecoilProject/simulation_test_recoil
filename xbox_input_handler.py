# xbox_input_handler.py
from state import SimulationState
try:
    from xbox_controller import XboxController
except ImportError:
    XboxController = None

class XboxInputHandler:
    """
    處理 Xbox 搖桿的輸入，並將其轉換為對 SimulationState 的更新。
    這個類別在主迴圈中被輪詢 (polled)。
    """
    def __init__(self, state: SimulationState):
        """初始化 XboxInputHandler。"""
        self.state = state
        self.config = state.config
        self.controller = None
        self.last_input_state = {}

        if XboxController:
            try:
                self.controller = XboxController()
                if not self.controller.is_connected():
                    self.controller = None
            except Exception as e:
                print(f"⚠️ 警告: 搖桿控制器初始化失敗: {e}")
                self.controller = None
        
        if self.controller:
            self.last_input_state = self.controller.get_input()
    
    def is_available(self) -> bool:
        """檢查搖桿是否已成功初始化並連接。"""
        return self.controller is not None and self.controller.is_connected()

    def update_state(self):
        """從搖桿讀取輸入並更新 SimulationState，應在主迴圈中被呼叫。"""
        if not self.is_available():
            if self.state.input_mode == "GAMEPAD":
                print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                self.state.toggle_input_mode("KEYBOARD")
            return

        current_input = self.controller.get_input()
        
        self.state.command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy']
        self.state.command[1] = -current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx']
        self.state.command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz']

        p_step, params = self.config.param_adjust_steps, self.state.tuning_params

        if current_input['button_select'] and not self.last_input_state['button_select']:
            self.state.reset_requested = True
            
        dpad_y, last_dpad_y = current_input['dpad'][1], self.last_input_state['dpad'][1]
        if dpad_y == 1 and last_dpad_y != 1: params.kp += p_step['kp']
        if dpad_y == -1 and last_dpad_y != 1: params.kp -= p_step['kp']
        
        dpad_x, last_dpad_x = current_input['dpad'][0], self.last_input_state['dpad'][0]
        if dpad_x == 1 and last_dpad_x != 1: params.kd += p_step['kd']
        if dpad_x == -1 and last_dpad_x != 1: params.kd -= p_step['kd']

        if current_input['button_r1'] and not self.last_input_state['button_r1']: params.action_scale += p_step['action_scale']
        if current_input['button_l1'] and not self.last_input_state['button_l1']: params.action_scale -= p_step['action_scale']
        if current_input['button_y'] and not self.last_input_state['button_y']: params.bias += p_step['bias']
        if current_input['button_a'] and not self.last_input_state['button_a']: params.bias -= p_step['bias']
        
        self.last_input_state = current_input
        
        params.kp = max(0, params.kp)
        params.kd = max(0, params.kd)
        params.action_scale = max(0, params.action_scale)

    def close(self):
        """關閉搖桿連接。"""
        if self.controller:
            self.controller.close()