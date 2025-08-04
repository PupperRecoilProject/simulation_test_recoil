# keyboard_input_handler.py
import numpy as np # 用於處理命令向量和清除命令
from typing import TYPE_CHECKING, Any # 用於類型提示，避免循環依賴

# 嘗試導入 glfw 庫，如果沒有安裝，則將 glfw 設定為 None。
# glfw 主要用於 MuJoCo 的 3D 渲染視窗交互。
try:
    import glfw
except ImportError:
    glfw = None

# 從 typing 模組導入 TYPE_CHECKING 以用於類型提示。
if TYPE_CHECKING:
    from state import SimulationState
    from terrain_manager import TerrainManager
    from xbox_input_handler import XboxInputHandler

# 導入日誌模組。
from logger import log

# 導入事件系統模組和所有需要使用的事件類型。
from event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_TERRAIN_REGENERATE_REQUESTED,
    EVENT_TERRAIN_SNAPSHOT_REQUESTED,
    EVENT_TERRAIN_MODE_CHANGE_REQUESTED,
    EVENT_UI_PAGE_CHANGE_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
    EVENT_TUNING_PARAM_ADJUST_REQUESTED,
    EVENT_COMMAND_UPDATED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUST_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED,
)

class KeyboardInputHandler:
    """
    [輸入層] 鍵盤輸入處理器。
    負責捕捉和處理所有來自 GLFW 視窗的鍵盤輸入事件，
    並將這些事件翻譯成系統可以理解的標準化請求事件，
    發布到事件匯流排 (`event_bus`)，實現與核心邏輯的解耦。
    """
    def __init__(self, state: 'SimulationState', xbox_handler: 'XboxInputHandler', terrain_manager: 'TerrainManager'):
        """
        [初始化] 初始化鍵盤輸入處理器。
        Args:
            state (SimulationState): 全域模擬狀態的參考。
            xbox_handler (XboxInputHandler): Xbox 搖桿處理器的參考。
            terrain_manager (TerrainManager): 地形管理器的參考。
        """
        self.state = state
        self.config = state.config
        self.serial_comm_ref = state.serial_communicator_ref
        self.xbox_handler = xbox_handler
        self.terrain_manager = terrain_manager # 用於查詢地形名稱，而非直接操作
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias'] # 可調參數的鍵名列表
        self.num_params = len(self.param_keys) # 可調參數的數量

    def register_callbacks(self, window: Any):
        """
        [回呼註冊] 向 GLFW 註冊鍵盤事件的回呼函式。
        Args:
            window (Any): GLFW 視窗物件。
        """
        if glfw is None:
            log.warning("glfw 模組不存在，無法註冊鍵盤事件，鍵盤輸入將無效。")
            return
        glfw.set_key_callback(window, self.key_callback)    # 註冊按鍵按下/釋放事件
        glfw.set_char_callback(window, self.char_callback)  # 註冊字元輸入事件 (用於文字輸入)

    def char_callback(self, window: Any, codepoint: int):
        """
        [事件回呼] 處理可列印字元輸入的回呼函式。
        主要用於序列埠控制台模式下的文字輸入。
        Args:
            window (Any): 觸發事件的 GLFW 視窗。
            codepoint (int): 輸入字元的 Unicode 編碼。
        """
        # 只有在序列埠模式下才處理字元輸入，將字元加入指令緩衝區。
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window: Any, key: int, scancode: int, action: int, mods: int):
        """
        [事件回呼] 處理所有按鍵事件的主回呼函式。
        根據當前控制模式和按鍵類型，發布相應的請求事件。
        Args:
            window (Any): 觸發事件的 GLFW 視窗。
            key (int): 按鍵的 GLFW 識別碼。
            scancode (int): 按鍵的平台特定掃描碼。
            action (int): 按鍵動作 (glfw.PRESS, glfw.REPEAT, glfw.RELEASE)。
            mods (int): 修飾鍵 (Ctrl, Alt, Shift) 的狀態。
        """
        # --- 模式壁壘邏輯：根據當前模式，分派給不同的處理函式 ---
        # 序列埠模式有特殊的輸入處理邏輯，不發布通用事件。
        if self.state.control_mode == "SERIAL_MODE":
            self.handle_serial_mode_keys(key, action)
            return # 處理完畢，立即返回

        # 對於非序列埠模式（包括 WALKING, FLOATING, JOINT_TEST, MANUAL_CTRL, HARDWARE_MODE），
        # 都統一由 handle_global_and_default_keys 處理，並發布事件。
        self.handle_global_and_default_keys(window, key, action)

    def handle_serial_mode_keys(self, key: int, action: int):
        """
        [輸入處理] 專門處理序列埠模式下的按鍵。
        這部分邏輯與核心模擬或硬體狀態無關，屬於 UI 自身的操作，故不發布事件。
        Args:
            key (int): 按鍵的 GLFW 識別碼。
            action (int): 按鍵動作。
        """
        # `~` 鍵：切換回上一個控制模式（退出序列埠模式）。
        if key == glfw.KEY_GRAVE_ACCENT and action == glfw.PRESS:
            # 這裡直接呼叫 set_control_mode 是可接受的，因為它屬於模式切換的狀態邏輯，
            # 最終的副作用處理會由 SimulationController 在其 on_mode_change_requested 中完成。
            self.state.set_control_mode(self.state.previous_control_mode)
            return

        # 處理 Enter 和 Backspace 鍵。
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_ENTER:
                # 發送指令到序列埠通訊器。
                log.info(f"[UI > Serial]: {self.state.serial_command_buffer}")
                self.serial_comm_ref.send_command(self.state.serial_command_buffer)
                self.state.serial_command_buffer = "" # 清空輸入緩衝區
            elif key == glfw.KEY_BACKSPACE:
                # 刪除緩衝區中的最後一個字元。
                self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]

    def handle_global_and_default_keys(self, window: Any, key: int, action: int):
        """
        [事件發布] 處理所有非序列埠模式下的鍵盤事件，並將其翻譯為事件發布。
        此函式是鍵盤輸入轉化為系統請求事件的核心。
        Args:
            window (Any): 觸發事件的 GLFW 視窗。
            key (int): 按鍵的 GLFW 識別碼。
            action (int): 按鍵動作 (glfw.PRESS, glfw.REPEAT, glfw.RELEASE)。
        """
        # --- 只在按下瞬間觸發的事件 (glfw.PRESS) ---
        if action == glfw.PRESS:
            # 1. 模式切換請求 (例如：F 鍵切換懸浮模式，G 鍵切換關節測試模式等)
            key_to_mode = {
                glfw.KEY_GRAVE_ACCENT: "SERIAL_MODE", # `~` 鍵，進入序列埠模式
                glfw.KEY_F: "FLOATING",               # F 鍵，切換到懸浮模式
                glfw.KEY_G: "JOINT_TEST",             # G 鍵，切換到關節測試模式
                glfw.KEY_B: "MANUAL_CTRL",            # B 鍵，切換到手動控制模式
                glfw.KEY_H: "HARDWARE_MODE",          # H 鍵，切換到硬體模式
            }
            if key in key_to_mode:
                # 特殊處理 'G' 鍵的退出邏輯：如果當前在 JOINT_TEST 或 MANUAL_CTRL 模式，
                # 'G' 鍵會根據硬體連接狀態返回 WALKING 或 HARDWARE_MODE。
                if key == glfw.KEY_G and self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                        event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="HARDWARE_MODE")
                    else:
                        event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="WALKING")
                else:
                    # 對於其他模式切換，直接發布請求事件。
                    event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode=key_to_mode[key])
                return

            # 2. AI 策略模型切換請求 (1-4 鍵)
            # 允許使用者通過數字鍵快速切換不同的 AI 策略模型。
            key_to_policy_index = {
                glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3
            }
            if key in key_to_policy_index:
                target_index = key_to_policy_index[key]
                # 確保目標索引在可用策略模型範圍內。
                if target_index < len(self.state.available_policies):
                    policy_name = self.state.available_policies[target_index]
                    event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=policy_name)
                else:
                    log.warning(f"策略索引 {target_index} 超出範圍，可用策略數：{len(self.state.available_policies)}")
                return

            # 3. 全域快捷鍵請求 (不與模式綁定，直接觸發系統級操作)
            if key == glfw.KEY_SPACE:
                # 暫停/播放模擬。此操作直接影響 SimulationController 的內部狀態，不通過事件。
                # 這裡是一個例外，因為它是直接控制底層時間流逝的開關，且只影響 SimulationController 自身。
                self.state.single_step_mode = not self.state.single_step_mode
                print(f"\n--- SIMULATION {'PAUSED' if self.state.single_step_mode else 'PLAYING'} ---")
                return
            if self.state.single_step_mode and key == glfw.KEY_N:
                # 在單步模式下，請求執行一次模擬步驟。
                self.state.execute_one_step = True
                return
            if key == glfw.KEY_ESCAPE:
                # 請求關閉 GLFW 視窗，這會觸發應用程式關閉流程。
                glfw.set_window_should_close(window, 1)
                return
            if key == glfw.KEY_R:
                # 發布硬重置模擬請求。
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")
                return
            if key == glfw.KEY_X:
                # 發布軟重置模擬請求。
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft")
                return
            if key == glfw.KEY_Y:
                # 發布重新生成地形請求 (僅在無限地形模式下有效)。
                # 這裡直接調用 terrain_manager 的判斷邏輯是為了提供更精確的 UI 反饋/日誌。
                if self.state.terrain_mode == "INFINITE":
                    event_bus.publish(EVENT_TERRAIN_REGENERATE_REQUESTED)
                else:
                    log.warning("⚠️ 'Y' 鍵 (重生地形) 只在無限地形模式下有效。")
                return
            if key == glfw.KEY_P:
                # 發布保存地形快照請求。
                event_bus.publish(EVENT_TERRAIN_SNAPSHOT_REQUESTED)
                return
            if key == glfw.KEY_TAB:
                # 發布 UI 頁面切換請求。
                event_bus.publish(EVENT_UI_PAGE_CHANGE_REQUESTED, direction=1)
                return
            if key == glfw.KEY_M:
                # 發布輸入模式切換請求 (鍵盤/搖桿)。
                target_mode = "GAMEPAD" if self.state.input_mode == "KEYBOARD" else "KEYBOARD"
                event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode=target_mode)
                return
            if key == glfw.KEY_U:
                # 發布連接序列埠設備請求。
                event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="serial")
                return
            if key == glfw.KEY_J:
                # 發布連接搖桿設備請求。
                event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="gamepad")
                return
            if key == glfw.KEY_K:
                # 發布硬體 AI 控制切換請求。
                event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED)
                return
            if key == glfw.KEY_V:
                # 發布地形模式切換請求 (循環切換無限/單一地形)。
                event_bus.publish(EVENT_TERRAIN_MODE_CHANGE_REQUESTED, direction=1)
                return
            
            # 在手動控制模式下，處理懸浮切換 (F 鍵，在模式切換部分已經處理，這裡避免重複)
            # if key == glfw.KEY_F and self.state.control_mode == "MANUAL_CTRL":
            #     event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLE_REQUESTED, value=not self.state.manual_mode_is_floating)
            #     return

        # --- 長按事件 (glfw.PRESS 或 glfw.REPEAT) ---
        if action in [glfw.PRESS, glfw.REPEAT]:
            # 1. 運動指令調整 (WASDQE)
            # 構建一個臨時的命令向量，然後發布更新事件。
            cmd_changed = False
            current_command = self.state.command.copy() # 獲取當前命令的副本以進行修改
            
            step = self.config.keyboard_velocity_adjust_step # 步進值
            
            if key == glfw.KEY_W: current_command[1] += step; cmd_changed = True # 前進 (vx)
            elif key == glfw.KEY_S: current_command[1] -= step; cmd_changed = True # 後退 (vx)
            elif key == glfw.KEY_A: current_command[0] += step; cmd_changed = True # 左移 (vy)
            elif key == glfw.KEY_D: current_command[0] -= step; cmd_changed = True # 右移 (vy)
            elif key == glfw.KEY_Q: current_command[2] += step; cmd_changed = True # 左轉 (wz)
            elif key == glfw.KEY_E: current_command[2] -= step; cmd_changed = True # 右轉 (wz)
            
            elif key == glfw.KEY_C and action == glfw.PRESS: # 清除指令 (只在按下瞬間觸發)
                current_command.fill(0.0); cmd_changed = True

            if cmd_changed:
                event_bus.publish(EVENT_COMMAND_UPDATED, command=current_command)
                return # 處理完畢，不繼續檢查其他按鍵

            # 2. 參數/關節調整 (方括號鍵選擇參數/關節，上下箭頭調整數值)
            
            # 選擇參數/關節索引 ([ 或 ])
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                # 根據當前模式發布不同的選擇請求事件
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, direction=-1)
                else: # 預設為調整調校參數
                    with self.state.lock: # 讀取 UI 狀態仍需加鎖，但修改索引仍然由 InputHandler 完成，因其僅影響 UI 選項。
                        self.state.tuning_param_index = (self.state.tuning_param_index - 1) % self.num_params
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    event_bus.publish(EVENT_JOINT_SELECT_REQUESTED, direction=1)
                else:
                    with self.state.lock:
                        self.state.tuning_param_index = (self.state.tuning_param_index + 1) % self.num_params
            
            # 調整參數/關節數值 (UP 或 DOWN)
            elif key == glfw.KEY_UP or key == glfw.KEY_DOWN:
                direction = 1 if key == glfw.KEY_UP else -1 # 1: 增加, -1: 減少
                if self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                    # 在關節測試/手動控制模式下，發布關節數值調整請求。
                    event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, direction=direction, step=0.1)
                else:
                    # 在其他模式下，發布調校參數調整請求。
                    param_to_adjust = self.param_keys[self.state.tuning_param_index]
                    event_bus.publish(EVENT_TUNING_PARAM_ADJUST_REQUESTED, param_name=param_to_adjust, direction=direction)
            
            # 清除關節偏移/目標 (C 鍵，僅在測試模式下)
            elif key == glfw.KEY_C and action == glfw.PRESS and self.state.control_mode in ["JOINT_TEST", "MANUAL_CTRL"]:
                event_bus.publish(EVENT_JOINT_VALUE_ADJUST_REQUESTED, clear=True)