# xbox_controller.py
import pygame

class XboxController:
    """
    一個使用 Pygame 函式庫來讀取 Xbox 搖桿輸入的類別。
    這個版本是非阻塞的，可以安全地在主迴圈中更新。
    """
    def __init__(self):
        """初始化 Pygame 並偵測搖桿。"""
        pygame.init() # 初始化所有 pygame 模組
        pygame.joystick.init() # 初始化搖桿模組

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0) # 獲取第一個搖桿
            self.joystick.init() # 初始化該搖桿
            print(f"✅ Pygame 偵測到搖桿: {self.joystick.get_name()}")
        else:
            print("⚠️ Pygame 未偵測到任何搖桿。")
            
        self.deadzone = 0.15
        # 狀態字典的結構保持不變，以相容 xbox_input_handler.py
        self.state = {
            'left_analog_x': 0.0, 'left_analog_y': 0.0,
            'right_analog_x': 0.0, 'right_analog_y': 0.0,
            'dpad': (0, 0),
            'button_a': 0, 'button_b': 0, 'button_x': 0, 'button_y': 0,
            'button_l1': 0, 'button_r1': 0,
            'button_select': 0, 'button_start': 0,
        }

    def is_connected(self) -> bool:
        """檢查搖桿是否已成功初始化。"""
        return self.joystick is not None

    def update(self):
        """
        處理 Pygame 事件佇列，更新搖桿狀態。
        這個方法應該在主模擬迴圈中被頻繁呼叫。
        """
        if not self.is_connected():
            return
            
        for event in pygame.event.get(): # 處理所有待辦事件
            if event.type == pygame.JOYAXISMOTION:
                # 類比搖桿 (Axis)
                # Axis 0: 左搖桿 X (-1 to 1)
                # Axis 1: 左搖桿 Y (-1 to 1, 上為負)
                # Axis 4: 右搖桿 X (-1 to 1)
                # Axis 3: 右搖桿 Y (-1 to 1, 上為負)
                if event.axis == 0: self.state['left_analog_x'] = event.value
                elif event.axis == 1: self.state['left_analog_y'] = event.value
                elif event.axis == 2: self.state['right_analog_x'] = event.value
                elif event.axis == 3: self.state['right_analog_y'] = event.value

            elif event.type == pygame.JOYBUTTONDOWN:
                # 按鈕按下
                if event.button == 0: self.state['button_a'] = 1
                elif event.button == 1: self.state['button_b'] = 1
                elif event.button == 2: self.state['button_x'] = 1
                elif event.button == 3: self.state['button_y'] = 1
                elif event.button == 4: self.state['button_l1'] = 1
                elif event.button == 5: self.state['button_r1'] = 1
                elif event.button == 6: self.state['button_select'] = 1
                elif event.button == 7: self.state['button_start'] = 1

            elif event.type == pygame.JOYBUTTONUP:
                # 按鈕釋放
                if event.button == 0: self.state['button_a'] = 0
                elif event.button == 1: self.state['button_b'] = 0
                elif event.button == 2: self.state['button_x'] = 0
                elif event.button == 3: self.state['button_y'] = 0
                elif event.button == 4: self.state['button_l1'] = 0
                elif event.button == 5: self.state['button_r1'] = 0
                elif event.button == 6: self.state['button_select'] = 0
                elif event.button == 7: self.state['button_start'] = 0

            elif event.type == pygame.JOYHATMOTION:
                # D-Pad
                self.state['dpad'] = event.value

    def get_input(self) -> dict:
        """獲取當前搖桿狀態的淺拷貝，並應用死區。"""
        # 應用死區
        for axis in ['left_analog_x', 'left_analog_y', 'right_analog_x', 'right_analog_y']:
            if abs(self.state[axis]) < self.deadzone:
                self.state[axis] = 0.0
        return self.state.copy()

    def close(self):
        """關閉 Pygame。"""
        pygame.quit()