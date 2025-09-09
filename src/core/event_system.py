# event_system.py
import threading
from collections import defaultdict
from src.core.logger import log

# ===================================================================
# 1. 系統事件字典 (System Event Dictionary)
#    - 這是我們系統的"標準語言"。所有模組間的通信都必須使用這裡定義的事件名稱。
#    - 使用大寫字母來表示這是一個全域的、不變的契約。
# ===================================================================

# --- 數據流事件 (Data Flow Events) ---
# 這些事件攜帶運行時的高頻數據。
EVENT_SIMULATION_TICK = "tick.simulation" # 來自模擬器的主迴圈滴答
EVENT_HARDWARE_TICK = "tick.hardware"     # 來自硬體控制器的主迴圈滴答

# --- 請求事件 (Request Events) ---
# 通常由UI或輸入處理器發起，請求核心邏輯執行某個動作。
EVENT_MODE_CHANGE_REQUESTED = "request.mode_change"
EVENT_HARDWARE_CONNECT_REQUESTED = "request.hardware_connect"
EVENT_HARDWARE_DISCONNECT_REQUESTED = "request.hardware_disconnect"
EVENT_HARDWARE_AI_TOGGLE_REQUESTED = "request.hardware_ai_toggle"
EVENT_SIMULATION_RESET_REQUESTED = "request.simulation_reset" # (包含 hard 和 soft)
EVENT_SHUTDOWN_REQUESTED = "request.shutdown"
EVENT_COMMAND_UPDATED = "request.command_updated"

# 擴充的 UI/輸入 請求事件，用於取代直接修改 state 的舊有邏輯
EVENT_TUNING_PARAM_ADJUSTED = "request.tuning_param_adjusted"  # 參數調校請求 {param_name: str, value?: float, direction?: int}
EVENT_INPUT_MODE_CHANGE_REQUESTED = "request.input_mode_change" # 輸入模式切換請求 {mode: str}
EVENT_DEVICE_CONNECT_REQUESTED = "request.device_connect"       # 設備連接請求 {device: str} (e.g., 'serial', 'gamepad')
EVENT_UI_PAGE_CHANGE_REQUESTED = "request.ui_page_change"         # UI 顯示頁面切換請求 {direction: int}
EVENT_TUNING_PARAM_SELECT_REQUESTED = "request.tuning_param_select" # 選擇下一個要調整的參數 {direction: int}
EVENT_POLICY_CHANGE_REQUESTED = "request.policy_change"         # AI 策略模型切換請求 {policy_name: str}
EVENT_TERRAIN_CHANGE_REQUESTED = "request.terrain_change"        # 地形模式/類型切換請求 {mode?: str, name?: str, index?: int}
EVENT_MANUAL_FLOAT_TOGGLED = "request.manual_float_toggled"    # 手動模式下的懸浮開關請求 {is_floating: bool}
EVENT_JOINT_SELECT_REQUESTED = "request.joint_select"           # 關節測試/手動模式下的關節選擇請求 {index: int}
EVENT_JOINT_VALUE_ADJUSTED = "request.joint_value_adjusted"     # 關節測試/手動模式下的關節值調整請求 {value?: float, direction?: float, clear?: bool}
EVENT_SERIAL_COMMAND_SEND = "request.serial_command_send"       # 序列埠命令發送請求 {command: str}


# --- 通知事件 (Notification Events) ---
# 由核心邏輯在完成某個動作後發出，用來通知UI或其他模組狀態已變更。
EVENT_STATE_UPDATED = "notification.state_updated" # 全域狀態已更新，通知UI刷新
EVENT_MODE_CHANGED = "notification.mode_changed"   # 控制模式已成功切換

EVENT_FRW_AUTO_INHIBIT_SET   = "request.frw_auto_inhibit_set"
EVENT_FRW_AUTO_INHIBIT_CLEAR = "request.frw_auto_inhibit_clear"
EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED = "request.firearm_recoil_warning.trigger"
EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED   = "request.firearm_recoil_warning.reset"

# ===================================================================
# 2. 事件匯流排實現 (Event Bus Implementation)
# ===================================================================
class EventSystem:
    """
    一個執行緒安全的事件匯流排。
    它作為系統中所有模組通信的中介，實現了完全的解耦。
    """
    def __init__(self):
        # _subscribers: 字典，鍵是事件名稱(str)，值是訂閱該事件的回呼函式列表。
        # defaultdict(list) 讓我們在訂閱一個新事件時無需檢查鍵是否存在。
        self._subscribers = defaultdict(list)
        # _lock: 一個執行緒鎖，用來保護 _subscribers 字典在多執行緒環境下的讀寫安全。
        self._lock = threading.Lock()
        log.info("✅ 事件匯流排已初始化。")

    def subscribe(self, event_name: str, callback):
        """
        將一個回呼函式註冊到指定的事件上。
        
        Args:
            event_name (str): 要訂閱的事件名稱 (必須是上面定義的常數之一)。
            callback (Callable): 事件觸發時要執行的函式。
        """
        with self._lock:
            # 確保同一個回呼函式不會被重複註冊
            if callback not in self._subscribers[event_name]:
                self._subscribers[event_name].append(callback)
                log.info(f"'{callback.__name__}' 已訂閱事件 '{event_name}'")

    def publish(self, event_name: str, *args, **kwargs):
        """
        發布一個事件，觸發所有已訂閱的回呼函式。
        
        Args:
            event_name (str): 要發布的事件名稱。
            *args, **kwargs: 傳遞給回呼函式的參數。
        """
        callbacks_to_run = []
        with self._lock:
            # 我們先複製一份回呼函式列表。
            # 這是一個重要的安全措施：如果在某個回呼函式內部又嘗試去訂閱或取消訂閱，
            # 這樣做可以避免在迭代過程中修改列表而導致的錯誤。
            callbacks_to_run = self._subscribers.get(event_name, [])[:]
        
        log.debug(f"發布事件 '{event_name}' (有 {len(callbacks_to_run)} 個訂閱者)")
        for callback in callbacks_to_run:
            try:
                # 執行回呼函式，並將所有參數傳遞給它
                callback(*args, **kwargs)
            except Exception as e:
                # 捕捉並記錄回呼函式中的異常，確保一個訂閱者的失敗不會影響其他訂閱者。
                log.error(f"執行事件 '{event_name}' 的回呼 '{callback.__name__}' 時出錯: {e}", exc_info=True)

# 創建一個全域唯一的事件匯流排實例，供整個應用程式使用。
# 這種單例模式確保了所有模組都與同一個神經中樞通信。
event_bus = EventSystem()