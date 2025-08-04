# xbox_input_handler.py
from state import SimulationState
from xbox_controller import XboxController
import threading
import time
# [新增] 導入事件系統
from event_system import event_bus, EVENT_COMMAND_UPDATED

class XboxInputHandler:
    """在背景執行緒中同步搖桿狀態至 SimulationState。"""

    def __init__(self, state: SimulationState):
        self.state = state
        self.config = state.config
        self.controller = XboxController()

        self._running = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """啟動處理器的背景執行緒。"""
        if self.thread and self.thread.is_alive():
            return
        self._running.set()
        # 先啟動底層 Pygame 輪詢執行緒
        self.controller.start_polling()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("✅ Xbox 輸入處理執行緒已啟動。")

    # 【重構】_update_loop 函式
    def _update_loop(self) -> None:
        """【修改後】持續從搖桿讀取數據，並將其作為事件發布，而不是直接修改全域狀態。"""
        param_keys = ['kp', 'kd', 'action_scale', 'bias']
        num_params = len(param_keys)
        last_input_state = {}

        while self._running.is_set():
            if not self.controller.is_connected():
                with self.state.lock:
                    if self.state.input_mode == "GAMEPAD":
                        self.state.input_mode = "KEYBOARD"
                        self.state.gamepad_is_connected = False
                        print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                time.sleep(1)
                continue

            current_input = self.controller.get_input()
            
            # --- 核心修改：從直接寫入 state 改為計算並發布事件 ---
            # 1. 計算新的指令向量
            new_command = np.zeros(3)
            # 只有在 GAMEPAD 模式下才計算指令，避免干擾鍵盤模式
            if self.state.input_mode == "GAMEPAD":
                new_command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy']
                new_command[1] = current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx'] * -1
                new_command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz']

            # 2. 發布指令更新事件
            # 即使指令是[0,0,0]，我們也發布，讓State Manager知道最新的用戶意圖。
            event_bus.publish(EVENT_COMMAND_UPDATED, command=new_command)

            # 3. 處理按鍵事件 (例如重置、參數調整)
            # 這些目前仍然是通過寫入請求旗標來實現，我們可以後續將它們也改為事件。
            # (為了分步進行，我們暫時保留這部分邏輯)
            with self.state.lock:
                if current_input['button_select'] and not last_input_state.get('button_select', 0):
                    self.state.hard_reset_requested = True

                if current_input['button_l1'] and not last_input_state.get('button_l1', 0):
                    self.state.tuning_param_index = (self.state.tuning_param_index - 1) % num_params
                if current_input['button_r1'] and not last_input_state.get('button_r1', 0):
                    self.state.tuning_param_index = (self.state.tuning_param_index + 1) % num_params

                dpad_y = current_input['dpad'][1]
                last_dpad_y = last_input_state.get('dpad', (0, 0))[1]
                if dpad_y != last_dpad_y:
                    param_to_adjust = param_keys[self.state.tuning_param_index]
                    step = self.config.param_adjust_steps.get(param_to_adjust, 0.1)
                    current_value = getattr(self.state.tuning_params, param_to_adjust)
                    if dpad_y == 1:
                        setattr(self.state.tuning_params, param_to_adjust, current_value + step)
                    elif dpad_y == -1:
                        setattr(self.state.tuning_params, param_to_adjust, current_value - step)

                self.state.tuning_params.kp = max(0, self.state.tuning_params.kp)
                self.state.tuning_params.kd = max(0, self.state.tuning_params.kd)
                self.state.tuning_params.action_scale = max(0, self.state.tuning_params.action_scale)

            last_input_state = current_input
            time.sleep(0.01)

    def scan_and_connect(self) -> bool:
        """呼叫底層控制器進行掃描並連接。"""
        is_success = self.controller.scan_and_connect()
        with self.state.lock:
            self.state.gamepad_is_connected = is_success
            if is_success:
                self.state.input_mode = "GAMEPAD"
        return is_success

    def close(self) -> None:
        """停止所有相關執行緒。"""
        self._running.clear()
        self.controller.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

