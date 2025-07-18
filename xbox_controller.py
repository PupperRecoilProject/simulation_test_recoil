# xbox_controller.py
import pygame

class XboxController:
    """
    一個使用 Pygame 函式庫來讀取 Xbox 搖桿輸入的類別。
    這個版本是非阻塞的，可以安全地在主迴圈中更新。
    """
    def __init__(self):
        """初始化 Pygame 但不立即掃描搖桿。"""
        pygame.init()
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
        print("✅ XBox 控制器已初始化 (等待連接指令)。")

    def scan_and_connect(self) -> bool:
        """掃描並連接到第一個可用的搖桿。"""
        if self.is_connected():
            print("搖桿已連接，無需重新掃描。")
            return True

        print("\n" + "="*20 + " 正在掃描搖桿 " + "="*20)
        pygame.joystick.init() # 每次掃描時重新初始化
        
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
            
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 0: self.state['left_analog_x'] = event.value
                elif event.axis == 1: self.state['left_analog_y'] = event.value
                elif event.axis == 2: self.state['right_analog_x'] = event.value
                elif event.axis == 3: self.state['right_analog_y'] = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0: self.state['button_a'] = 1
                elif event.button == 1: self.state['button_b'] = 1
                elif event.button == 2: self.state['button_x'] = 1
                elif event.button == 3: self.state['button_y'] = 1
                elif event.button == 4: self.state['button_l1'] = 1
                elif event.button == 5: self.state['button_r1'] = 1
                elif event.button == 6: self.state['button_select'] = 1
                elif event.button == 7: self.state['button_start'] = 1
            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0: self.state['button_a'] = 0
                elif event.button == 1: self.state['button_b'] = 0
                elif event.button == 2: self.state['button_x'] = 0
                elif event.button == 3: self.state['button_y'] = 0
                elif event.button == 4: self.state['button_l1'] = 0
                elif event.button == 5: self.state['button_r1'] = 0
                elif event.button == 6: self.state['button_select'] = 0
                elif event.button == 7: self.state['button_start'] = 0
            elif event.type == pygame.JOYHATMOTION:
                self.state['dpad'] = event.value

    def get_input(self) -> dict:
        """獲取當前搖桿狀態的淺拷貝，並應用死區。"""
        for axis in ['left_analog_x', 'left_analog_y', 'right_analog_x', 'right_analog_y']:
            if abs(self.state[axis]) < self.deadzone:
                self.state[axis] = 0.0
        return self.state.copy()

    def close(self):
        """關閉 Pygame。"""
        pygame.quit()