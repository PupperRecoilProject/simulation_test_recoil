from src.core.state import SimulationState
from src.hardware.xbox_controller import XboxController
import threading
import time
import numpy as np  # 確保 numpy 已導入
from src.core.event_system import (
    event_bus, 
    EVENT_COMMAND_UPDATED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_TUNING_PARAM_ADJUSTED,
    EVENT_TUNING_PARAM_SELECT_REQUESTED # 導入新事件
)


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
        """
        【v3.0.1】持續從搖桿讀取數據，並將所有用戶意圖作為事件發布。
        此函式不再包含任何業務邏輯，只負責翻譯輸入。
        """
        last_input_state = {}

        while self._running.is_set():
            if not self.controller.is_connected():
                # [保留] 斷線後自動切換模式的邏輯
                with self.state.lock:
                    if self.state.input_mode == "GAMEPAD":
                        # 這裡暫時還需要直接修改 state，後續可以改為發布事件
                        self.state.toggle_input_mode("KEYBOARD")
                        self.state.gamepad_is_connected = False
                        print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                time.sleep(1)
                continue

            current_input = self.controller.get_input()
            
            # --- 1. 翻譯移動指令 ---
            # 只有在 GAMEPAD 模式下才計算並發布指令
            if self.state.input_mode == "GAMEPAD":
                new_command = np.zeros(3)
                new_command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy']
                new_command[1] = current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx'] * -1
                new_command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz']
                # [保留] 發布指令更新事件
                event_bus.publish(EVENT_COMMAND_UPDATED, command=new_command)

            # --- 2. 翻譯按鍵事件 (只在按鍵按下的瞬間觸發一次) ---
            # 硬重置請求
            if current_input['button_select'] and not last_input_state.get('button_select', 0):
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")

            # 參數選擇請求 (L1/R1)
            if current_input['button_l1'] and not last_input_state.get('button_l1', 0):
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=-1)
            if current_input['button_r1'] and not last_input_state.get('button_r1', 0):
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=1)

            # 參數調整請求 (D-Pad Up/Down)
            dpad_y = current_input['dpad'][1]
            last_dpad_y = last_input_state.get('dpad', (0, 0))[1]
            if dpad_y != last_dpad_y:  # 檢測狀態變化
                if dpad_y == 1:
                    # 發布一個「增加」當前參數的請求
                    event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=1)
                elif dpad_y == -1:
                    # 發布一個「減少」當前參數的請求
                    event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=-1)
            
            # --- 3. 更新上一幀的狀態以進行邊緣檢測 ---
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

