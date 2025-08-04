# xbox_input_handler.py
import threading
import time
import numpy as np
from typing import TYPE_CHECKING

# 從 typing 模組導入 TYPE_CHECKING 以用於類型提示。
# 這避免了在運行時引入循環依賴。
if TYPE_CHECKING:
    from state import SimulationState

# 導入事件系統模組和所需的事件類型。
from event_system import (
    event_bus,
    EVENT_COMMAND_UPDATED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_TUNING_PARAM_ADJUST_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED
)

# 導入底層 Xbox 搖桿控制器介面。
from xbox_controller import XboxController

class XboxInputHandler:
    """
    [輸入層] Xbox 輸入處理器。
    在背景執行緒中持續輪詢 Xbox 搖桿的狀態，
    並將搖桿輸入（如類比搖桿、按鈕、D-Pad）翻譯成標準化的事件，
    發布到事件匯流排中，供系統中的其他模組響應。
    """

    def __init__(self, state: 'SimulationState'):
        """
        [初始化] 初始化 XboxInputHandler。
        Args:
            state (SimulationState): 全域模擬狀態的參考，用於讀取配置和部分內部狀態。
        """
        self.state = state
        self.config = state.config
        self.controller = XboxController() # 底層搖桿驅動

        self._running = threading.Event()  # 控制輪詢執行緒的運行狀態
        self.thread: threading.Thread | None = None # 搖桿輪詢執行緒

    def start(self) -> None:
        """
        [核心迴圈控制] 啟動 Xbox 輸入處理器的背景執行緒。
        如果執行緒已經在運行，則不重複啟動。
        """
        if self.thread and self.thread.is_alive():
            return
        self._running.set() # 設定運行旗標
        self.controller.start_polling() # 啟動底層 Pygame 搖桿輪詢
        self.thread = threading.Thread(target=self._update_loop, daemon=True) # 創建處理迴圈執行緒
        self.thread.start() # 啟動執行緒
        print("✅ Xbox 輸入處理執行緒已啟動。")

    def _update_loop(self) -> None:
        """
        [核心迴圈] Xbox 搖桿數據輪詢與事件發布迴圈。
        此函式在一個獨立的背景執行緒中運行，持續讀取搖桿狀態，
        並根據輸入發布相應的事件，實現與核心邏輯的解耦。
        """
        param_keys = ['kp', 'kd', 'action_scale', 'bias'] # 可調整參數的名稱列表
        num_params = len(param_keys) # 參數數量
        last_input_state = {} # 儲存上一次的搖桿輸入狀態，用於檢測按鍵的邊緣觸發

        while self._running.is_set(): # 只要運行旗標被設定，就持續循環
            # 檢查搖桿連接狀態。如果斷開，嘗試自動切換到鍵盤模式。
            if not self.controller.is_connected():
                with self.state.lock: # 讀取共享狀態需要加鎖
                    if self.state.input_mode == "GAMEPAD":
                        self.state.input_mode = "KEYBOARD" # 自動切換輸入模式
                        self.state.gamepad_is_connected = False # 更新連接狀態
                        print("🎮 搖桿已斷開，自動切換回鍵盤模式。")
                time.sleep(1) # 短暫休眠，避免頻繁檢查
                continue # 繼續下一輪迴圈

            current_input = self.controller.get_input() # 獲取當前搖桿輸入狀態

            # --- 1. 移動指令處理 ---
            # 根據左/右類比搖桿輸入，計算新的運動指令向量。
            new_command = np.zeros(3, dtype=np.float32)
            if self.state.input_mode == "GAMEPAD": # 只有在 GAMEPAD 模式下才處理搖桿的運動指令
                new_command[0] = current_input['left_analog_x'] * self.config.gamepad_sensitivity['vy'] # 左右平移 (vy)
                new_command[1] = current_input['left_analog_y'] * self.config.gamepad_sensitivity['vx'] * -1 # 前後移動 (vx)，注意搖桿 Y 軸可能方向相反
                new_command[2] = current_input['right_analog_x'] * self.config.gamepad_sensitivity['wz'] # 轉向 (wz)

            # 發布指令更新事件。即使指令是零向量，也發布以確保狀態同步。
            event_bus.publish(EVENT_COMMAND_UPDATED, command=new_command)

            # --- 2. 按鍵事件處理 (重構成事件發布) ---
            # 這些操作會發布請求事件，由中央調度器 (SimulationController) 處理。

            # 硬重置請求 (Select 按鈕)
            if current_input['button_select'] and not last_input_state.get('button_select', 0):
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")

            # 參數索引切換 (L1/R1 按鈕)
            # 這是 UI 相關的狀態，目前仍由本模組直接修改 state.tuning_param_index。
            # 更徹底的重構可能會為此定義一個事件 (EVENT_TUNING_PARAM_INDEX_CHANGE_REQUESTED)。
            with self.state.lock: # 修改共享狀態需要加鎖
                if current_input['button_l1'] and not last_input_state.get('button_l1', 0):
                    self.state.tuning_param_index = (self.state.tuning_param_index - 1) % num_params
                if current_input['button_r1'] and not last_input_state.get('button_r1', 0):
                    self.state.tuning_param_index = (self.state.tuning_param_index + 1) % num_params

            # 參數值調整 (D-Pad 上/下)
            dpad_y = current_input['dpad'][1]
            last_dpad_y = last_input_state.get('dpad', (0, 0))[1]
            if dpad_y != last_dpad_y: # 檢測 D-Pad 狀態變化
                if dpad_y in [1, -1]: # 如果 D-Pad 上或下被按下
                    # 獲取當前選中的參數名稱
                    param_to_adjust = param_keys[self.state.tuning_param_index]
                    # 判斷調整方向
                    direction = 1 if dpad_y == 1 else -1
                    # 發布參數調整請求事件
                    event_bus.publish(EVENT_TUNING_PARAM_ADJUST_REQUESTED, param_name=param_to_adjust, direction=direction)

            last_input_state = current_input.copy() # 更新上一次的輸入狀態
            time.sleep(0.01) # 短暫休眠，避免 CPU 佔用過高

    def scan_and_connect(self) -> bool:
        """
        [設備連接] 呼叫底層 XboxController 進行掃描並連接搖桿。
        並發布輸入模式切換事件。
        Returns:
            bool: 連接是否成功。
        """
        is_success = self.controller.scan_and_connect()
        with self.state.lock: # 讀寫共享狀態需要加鎖
            self.state.gamepad_is_connected = is_success # 更新搖桿連接狀態
            if is_success:
                # 如果連接成功，發布輸入模式切換請求事件
                event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode="GAMEPAD")
        return is_success

    def close(self) -> None:
        """
        [核心迴圈控制] 安全停止 Xbox 輸入處理器的背景執行緒和底層搖桿輪詢。
        """
        self._running.clear() # 清除運行旗標，通知執行緒停止
        self.controller.close() # 關閉底層搖桿輪詢
        if self.thread and self.thread.is_alive(): # 等待執行緒結束
            self.thread.join(timeout=1)