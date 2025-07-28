# xbox_controller.py
import pygame

class XboxController:
    """
    一個使用 Pygame 函式庫來讀取 Xbox 搖桿輸入的類別。
    這個版本是非阻塞的，可以安全地在主迴圈中更新。
    由於 Pygame 的事件系統必須在初始化它的執行緒中運作，
    因此真正的 `pygame.init()` 將由模擬執行緒呼叫 :func:`initialize` 完成。
    """

    def __init__(self):
        """僅設定初始狀態，不立即初始化 Pygame。"""
        self.joystick = None
        self.deadzone = 0.15
        self.state = {
            'left_analog_x': 0.0, 'left_analog_y': 0.0,
            'right_analog_x': 0.0, 'right_analog_y': 0.0,
            'dpad': (0, 0),
            'button_a': 0, 'button_b': 0, 'button_x': 0, 'button_y': 0,
            'button_l1': 0, 'button_r1': 0,
            'button_select': 0, 'button_start': 0,
        }
        print("✅ XBox Controller 物件已建立 (等待執行緒初始化)。")

    def initialize(self) -> bool:
        """在當前執行緒中初始化 Pygame 與搖桿模組。"""
        try:
            pygame.init()
            # 【關鍵修正】建立一個 1x1 的不可見視窗，
            # 以確保 pygame 的事件系統穩定運作。
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
            pygame.joystick.init()
            print("✅ Pygame 在模擬執行緒中初始化完成 (含虛擬視窗)。")
            return True
        except pygame.error as e:
            print(f"❌ Pygame 初始化失敗: {e}")
            return False

    def scan_and_connect(self) -> bool:
        """掃描並連接到第一個可用的搖桿。"""
        if self.is_connected():
            print("搖桿已連接，無需重新掃描。")
            return True

        print("\n" + "="*20 + " 正在掃描搖桿 " + "="*20)
        # 重新初始化搖桿子系統，確保能偵測到新插入的設備
        pygame.joystick.quit()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"✅ 成功連接到搖桿: {self.joystick.get_name()}")
            return True
        else:
            print("--- 未偵測到任何搖桿 ---")
            self.joystick = None
            return False

    def is_connected(self) -> bool:
        """檢查搖桿是否已成功初始化。"""
        return self.joystick is not None

    def update(self):
        """處理 Pygame 事件佇列，更新搖桿狀態。"""
        if not self.is_connected():
            return

        try:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # 只處理我們認識的軸索引，其餘忽略
                    if event.axis == 0:
                        self.state['left_analog_x'] = event.value
                    elif event.axis == 1:
                        self.state['left_analog_y'] = event.value
                    elif event.axis == 2:
                        self.state['right_analog_x'] = event.value
                    elif event.axis == 3:
                        self.state['right_analog_y'] = event.value
                elif event.type == pygame.JOYBUTTONDOWN:
                    # 使用映射表處理，避免未知索引導致 KeyError
                    button_map = {
                        0: 'button_a', 1: 'button_b', 2: 'button_x', 3: 'button_y',
                        4: 'button_l1', 5: 'button_r1', 6: 'button_select', 7: 'button_start'
                    }
                    if event.button in button_map:
                        self.state[button_map[event.button]] = 1
                elif event.type == pygame.JOYBUTTONUP:
                    button_map = {
                        0: 'button_a', 1: 'button_b', 2: 'button_x', 3: 'button_y',
                        4: 'button_l1', 5: 'button_r1', 6: 'button_select', 7: 'button_start'
                    }
                    if event.button in button_map:
                        self.state[button_map[event.button]] = 0
                elif event.type == pygame.JOYHATMOTION:
                    # 部分控制器可能有多個 hat，這裡僅處理第一個
                    if hasattr(event, 'hat') and event.hat == 0:
                        self.state['dpad'] = event.value
        except pygame.error as e:
            # 在某些情況下（例如視窗失去焦點），事件處理可能會失敗，我們在此捕獲錯誤以避免程式崩潰
            print(f"⚠️ Pygame 事件處理時發生錯誤: {e}")

    def get_input(self) -> dict:
        """獲取當前搖桿狀態的淺拷貝，並應用死區。"""
        for axis in ['left_analog_x', 'left_analog_y', 'right_analog_x', 'right_analog_y']:
            if abs(self.state[axis]) < self.deadzone:
                self.state[axis] = 0.0
        return self.state.copy()

    def close(self):
        """關閉 Pygame。"""
        pygame.quit()
