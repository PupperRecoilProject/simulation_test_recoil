# event_system.py
import threading
from collections import defaultdict
from logger import log

# ===================================================================
# 1. 系統事件字典 (System Event Dictionary)
#    - [修改] 擴充請求事件，使其覆蓋所有輸入層操作。
# ===================================================================

# --- 數據流事件 (Data Flow Events) ---
# [保留] 這些事件攜帶運行時的高頻數據。
EVENT_SIMULATION_TICK = "tick.simulation" # 來自模擬器的主迴圈滴答
EVENT_HARDWARE_TICK = "tick.hardware"     # 來自硬體控制器的主迴圈滴答

# --- 請求事件 (Request Events) ---
# [保留] 這些是系統中已有的核心請求。
EVENT_MODE_CHANGE_REQUESTED = "request.mode_change"
EVENT_HARDWARE_CONNECT_REQUESTED = "request.hardware_connect" # 將被通用的 DEVICE_CONNECT 取代，但暫時保留以兼容
EVENT_HARDWARE_DISCONNECT_REQUESTED = "request.hardware_disconnect"
EVENT_HARDWARE_AI_TOGGLE_REQUESTED = "request.hardware_ai_toggle"
EVENT_SHUTDOWN_REQUESTED = "request.shutdown"
EVENT_COMMAND_UPDATED = "request.command_updated"

# [修改] 將原有的 SIMULATION_RESET_REQUESTED 從旗標改為正式事件。
#       舊的旗標方式: state.hard_reset_requested = True
#       新的事件方式: event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")
EVENT_SIMULATION_RESET_REQUESTED = "request.simulation_reset"

# [新增] 為所有輕量級狀態變更和操作定義新的請求事件，以取代直接修改 state。
EVENT_INPUT_MODE_CHANGE_REQUESTED = "request.input_mode_change"   # 請求切換輸入模式 (鍵盤/搖桿)
EVENT_TUNING_PARAM_ADJUST_REQUESTED = "request.tuning_param_adjust" # 請求調整參數 (Kp, Kd, etc.)
EVENT_POLICY_CHANGE_REQUESTED = "request.policy_change"           # 請求切換 AI 策略模型
EVENT_TERRAIN_MODE_CHANGE_REQUESTED = "request.terrain_mode_change" # 請求切換地形模式 (無限/單一)
EVENT_TERRAIN_REGENERATE_REQUESTED = "request.terrain_regenerate"    # 請求重新生成無限地形
EVENT_TERRAIN_SNAPSHOT_REQUESTED = "request.terrain_snapshot"       # 請求保存地形快照
EVENT_DEVICE_CONNECT_REQUESTED = "request.device_connect"           # [通用化] 請求連接設備 (序列埠/搖桿)

# [新增] 為純 UI 狀態變更定義請求事件，雖然它們不影響核心物理，但保持架構一致性。
EVENT_UI_PAGE_CHANGE_REQUESTED = "request.ui_page_change"           # 請求切換 UI 顯示頁面
EVENT_JOINT_SELECT_REQUESTED = "request.joint_select"               # 請求在測試模式下選擇關節
EVENT_JOINT_VALUE_ADJUST_REQUESTED = "request.joint_value_adjust"   # 請求在測試模式下調整關節值
EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED = "request.manual_float_toggle" # 請求在手動模式下切換懸浮

# --- 通知事件 (Notification Events) ---
# [保留] 由核心邏輯在完成某個動作後發出，用來通知UI或其他模組狀態已變更。
EVENT_STATE_UPDATED = "notification.state_updated" # 全域狀態已更新，通知UI刷新
EVENT_MODE_CHANGED = "notification.mode_changed"   # 控制模式已成功切換

# ===================================================================
# 2. 事件匯流排實現 (Event Bus Implementation)
#    - [保留] 這部分的實現非常健壯，無需修改。
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
        
        # [修改] 調整日誌級別為 DEBUG，避免在正式運行時產生過多日誌，只在除錯時顯示
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