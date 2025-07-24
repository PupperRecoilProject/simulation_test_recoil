# xbox_input_handler.py
from state import SimulationState
from xbox_controller import XboxController

class XboxInputHandler:
    """
    處理 Xbox 搖桿的輸入，並將其轉換為對 SimulationState 的更新。
    """
    def __init__(self, state: SimulationState):
        """初始化 XboxInputHandler。"""
        self.state = state
        self.config = state.config
        self.controller = XboxController()
        self.last_input_state = {}
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']
        self.num_params = len(self.param_keys)
    
    def scan_and_connect(self) -> bool:
        """呼叫底層控制器進行掃描和連接。"""
        is_success = self.controller.scan_and_connect()
        if is_success:
            self.state.toggle_input_mode("GAMEPAD")
        return is_success

    def is_available(self) -> bool:
        """檢查搖桿是否已成功初始化並連接。"""
        return self.controller.is_connected()

    def update_state(self):
        """從搖桿讀取輸入並更新 SimulationState。"""
        if not self.is_available():
            if self.state.input_mode == "GAMEPAD":
                print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                self.state.toggle_input_mode("KEYBOARD")
                self.state.gamepad_is_connected = False
            return

        self.controller.update() 
        current_input = self.controller.get_input()
        
        self.state.command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy']
        self.state.command[1] = current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx'] * -1
        self.state.command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz']

        if current_input['button_select'] and not self.last_input_state.get('button_select', 0):
            self.state.hard_reset_requested = True
            
        if current_input['button_l1'] and not self.last_input_state.get('button_l1', 0):
            self.state.tuning_param_index = (self.state.tuning_param_index - 1) % self.num_params
        
        if current_input['button_r1'] and not self.last_input_state.get('button_r1', 0):
            self.state.tuning_param_index = (self.state.tuning_param_index + 1) % self.num_params

        dpad_y = current_input['dpad'][1]
        last_dpad_y = self.last_input_state.get('dpad', (0,0))[1]

        if dpad_y != last_dpad_y:
            param_to_adjust = self.param_keys[self.state.tuning_param_index]
            step = self.config.param_adjust_steps.get(param_to_adjust, 0.1)
            current_value = getattr(self.state.tuning_params, param_to_adjust)

            if dpad_y == 1:
                setattr(self.state.tuning_params, param_to_adjust, current_value + step)
            elif dpad_y == -1:
                setattr(self.state.tuning_params, param_to_adjust, current_value - step)
        
        self.last_input_state = current_input
        
        self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
        self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
        self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)

    def close(self):
        """關閉搖桿連接。"""
        if self.controller:
            self.controller.close()
