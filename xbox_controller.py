# xbox_controller.py
import math
import threading
from inputs import get_gamepad

# 嘗試匯入我們自訂的 XboxController 類別
class XboxController(object):
    """
    一個用於讀取 Xbox 搖桿輸入的類別。
    它在背景執行緒中持續監聽搖桿事件，以避免阻塞主程式。
    """
    # 類比搖桿的最大絕對值
    MAX_TRIG_VAL = math.pow(2, 8)  # 扳機鍵 (Trigger) 的最大值 (8-bit)
    MAX_JOY_VAL = math.pow(2, 15) # 類比搖桿 (Analog Stick) 的最大值 (15-bit)

    def __init__(self):
        """初始化搖桿狀態字典和背景監聽執行緒。"""
        self._monitor_thread = None
        self.deadzone = 0.15 # 類比搖桿的死區，避免飄移

        # 儲存所有搖桿輸入的狀態
        self.state = {
            'left_analog_x': 0.0, 'left_analog_y': 0.0,
            'right_analog_x': 0.0, 'right_analog_y': 0.0,
            'left_trigger': 0.0, 'right_trigger': 0.0,
            'dpad': (0, 0), # (x, y)
            'button_a': 0, 'button_b': 0, 'button_x': 0, 'button_y': 0,
            'button_l1': 0, 'button_r1': 0, # Shoulder buttons (LB, RB)
            'button_select': 0, # "View" button on modern controllers
            'button_start': 0, # "Menu" button on modern controllers
        }

        self._is_connected = False
        self._start_monitoring()

    def is_connected(self):
        """檢查搖桿是否已連接。"""
        return self._is_connected

    def _start_monitoring(self):
        """啟動背景執行緒來監聽搖桿事件。"""
        # 檢查是否有搖桿連接
        try:
            get_gamepad()
            self._is_connected = True
            print("✅ Xbox 搖桿已連接。")
        except Exception as e:
            self._is_connected = False
            print(f"⚠️ 警告: 未找到 Xbox 搖桿。 {e}")
            return

        # 啟動執行緒
        self._monitor_thread = threading.Thread(target=self._monitor_controller, daemon=True)
        self._monitor_thread.start()

    def _monitor_controller(self):
        """
        在背景執行緒中執行的主函式，持續讀取和解析搖桿事件。
        """
        while True:
            try:
                events = get_gamepad()
                for event in events:
                    self._parse_event(event)
            except Exception as e:
                print(f"❌ 搖桿讀取錯誤，可能已斷開連接: {e}")
                self._is_connected = False
                # 清除所有狀態
                for key in self.state:
                    if isinstance(self.state[key], tuple):
                        self.state[key] = (0, 0)
                    else:
                        self.state[key] = 0.0 if isinstance(self.state[key], float) else 0
                break # 結束監聽

    def _parse_event(self, event):
        """解析單個搖桿事件並更新狀態字典。"""
        # 類比搖桿
        if event.code == 'ABS_Y':
            self.state['left_analog_y'] = event.state / XboxController.MAX_JOY_VAL
        elif event.code == 'ABS_X':
            self.state['left_analog_x'] = event.state / XboxController.MAX_JOY_VAL
        elif event.code == 'ABS_RY':
            self.state['right_analog_y'] = event.state / XboxController.MAX_JOY_VAL
        elif event.code == 'ABS_RX':
            self.state['right_analog_x'] = event.state / XboxController.MAX_JOY_VAL
        # 扳機
        elif event.code == 'ABS_Z':
            self.state['left_trigger'] = event.state / XboxController.MAX_TRIG_VAL
        elif event.code == 'ABS_RZ':
            self.state['right_trigger'] = event.state / XboxController.MAX_TRIG_VAL
        # D-Pad
        elif event.code == 'ABS_HAT0Y':
            self.state['dpad'] = (self.state['dpad'][0], -event.state)
        elif event.code == 'ABS_HAT0X':
            self.state['dpad'] = (-event.state, self.state['dpad'][1])
        # 按鈕
        elif event.code in ['BTN_SOUTH', 'BTN_WEST', 'BTN_NORTH', 'BTN_EAST', 'BTN_TL', 'BTN_TR', 'BTN_SELECT', 'BTN_START']:
            key_map = {
                'BTN_SOUTH': 'button_a', 'BTN_EAST': 'button_b',
                'BTN_NORTH': 'button_y', 'BTN_WEST': 'button_x',
                'BTN_TL': 'button_l1', 'BTN_TR': 'button_r1',
                'BTN_SELECT': 'button_select', 'BTN_START': 'button_start'
            }
            self.state[key_map[event.code]] = event.state

    def get_input(self):
        """
        獲取當前搖桿狀態的淺拷貝，並應用死區。
        
        Returns:
            dict: 包含所有搖桿輸入狀態的字典。
        """
        # 創建一個副本以避免在讀取時發生執行緒衝突
        current_state = self.state.copy()

        # 應用死區
        for axis in ['left_analog_x', 'left_analog_y', 'right_analog_x', 'right_analog_y']:
            if abs(current_state[axis]) < self.deadzone:
                current_state[axis] = 0.0
        
        return current_state

    def close(self):
        """
        雖然 daemon thread 會自動結束，但提供一個 close 方法是好的實踐。
        實際上，因為 inputs 庫的設計，我們無法輕易停止 get_gamepad()。
        """
        print("搖桿監聽執行緒將隨主程式結束。")

if __name__ == '__main__':
    # 測試程式碼
    import time
    joy = XboxController()
    if joy.is_connected():
        print("搖桿已連接，開始監聽輸入。按 Ctrl+C 結束。")
        while True:
            try:
                state = joy.get_input()
                # 只在狀態改變時打印
                # 為了簡潔，這裡每次都打印
                print(state)
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n測試結束。")
                break
    else:
        print("未能連接到搖桿，測試結束。")
    