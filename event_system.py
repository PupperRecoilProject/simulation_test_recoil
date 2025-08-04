# event_system.py
import threading
from collections import defaultdict
from logger import log
from typing import Callable, Any

# ===================================================================
# 系統事件字典 (System Event Dictionary)
# 這是應用程式中所有模組間通信的標準語言，確保解耦與一致性。
# ===================================================================

# --- 數據流事件 (Data Flow Events) ---
# 這些事件攜帶運行時的高頻數據，通常由模擬或硬體核心產生。
EVENT_SIMULATION_TICK = "tick.simulation" # 來自模擬器的主迴圈滴答
EVENT_HARDWARE_TICK = "tick.hardware"     # 來自硬體控制器的主迴圈滴答

# --- 請求事件 (Request Events) ---
# 通常由 UI 或輸入處理器發起，請求核心邏輯執行某個動作。
# 這些事件將被中央調度器 (SimulationController) 監聽和處理。
EVENT_MODE_CHANGE_REQUESTED = "request.mode_change"                 # 請求切換控制模式 (WALKING, FLOATING, HARDWARE_MODE 等)
EVENT_HARDWARE_AI_TOGGLE_REQUESTED = "request.hardware_ai_toggle"   # 請求啟用/停用硬體 AI 控制
EVENT_SHUTDOWN_REQUESTED = "request.shutdown"                       # 請求關閉整個應用程式
EVENT_COMMAND_UPDATED = "request.command_updated"                   # 請求更新機器人運動指令 (線速度、角速度)
EVENT_SIMULATION_RESET_REQUESTED = "request.simulation_reset"       # 請求重置模擬狀態 (硬重置/軟重置)

EVENT_INPUT_MODE_CHANGE_REQUESTED = "request.input_mode_change"     # 請求切換輸入模式 (KEYBOARD, GAMEPAD, VJOY)
EVENT_TUNING_PARAM_ADJUST_REQUESTED = "request.tuning_param_adjust" # 請求調整機器人控制參數 (Kp, Kd, Action Scale, Bias)
EVENT_POLICY_CHANGE_REQUESTED = "request.policy_change"             # 請求切換當前使用的 AI 策略模型
EVENT_TERRAIN_MODE_CHANGE_REQUESTED = "request.terrain_mode_change" # 請求切換地形生成模式 (INFINITE, SINGLE)
EVENT_TERRAIN_REGENERATE_REQUESTED = "request.terrain_regenerate"    # 請求重新生成無限地形 (在 INFINITE 模式下)
EVENT_TERRAIN_SNAPSHOT_REQUESTED = "request.terrain_snapshot"       # 請求保存當前地形的高度場快照為 PNG 圖片
EVENT_DEVICE_CONNECT_REQUESTED = "request.device_connect"           # 請求連接外部設備 (序列埠或搖桿)

# --- 純 UI 狀態變更請求事件 ---
# 這些事件主要影響 UI 的顯示狀態，但為了保持事件驅動架構的一致性而定義。
EVENT_UI_PAGE_CHANGE_REQUESTED = "request.ui_page_change"           # 請求切換 UI 顯示介面上的除錯頁面
EVENT_JOINT_SELECT_REQUESTED = "request.joint_select"               # 請求在關節測試/手動控制模式下選擇目標關節
EVENT_JOINT_VALUE_ADJUST_REQUESTED = "request.joint_value_adjust"   # 請求在關節測試/手動控制模式下調整目標關節角度
EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED = "request.manual_float_toggle" # 請求在手動控制模式下切換機器人懸浮狀態

# --- 通知事件 (Notification Events) ---
# 由核心邏輯在完成某個動作後發出，用來通知 UI 或其他模組狀態已變更。
EVENT_STATE_UPDATED = "notification.state_updated" # 全域狀態已更新，通知 UI 刷新 (目前未被使用，但作為設計預留)
EVENT_MODE_CHANGED = "notification.mode_changed"   # 控制模式已成功切換

# ===================================================================
# 事件匯流排實現 (Event Bus Implementation)
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

    def subscribe(self, event_name: str, callback: Callable):
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

    def publish(self, event_name: str, *args: Any, **kwargs: Any):
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
        
        # 調整日誌級別為 DEBUG，避免在正式運行時產生過多日誌，只在除錯時顯示。
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