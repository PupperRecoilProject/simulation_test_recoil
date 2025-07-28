# xbox_controller.py
import pygame
import threading
import time

class XboxController:
    """管理 Xbox 搖桿輸入的類別。"""

    def __init__(self):
        """僅設定初始狀態，不立即初始化 Pygame。"""
        self.joystick = None
        self.deadzone = 0.15
        self.state = {
            'left_analog_x': 0.0,
            'left_analog_y': 0.0,
            'right_analog_x': 0.0,
            'right_analog_y': 0.0,
            'dpad': (0, 0),
            'button_a': 0,
            'button_b': 0,
            'button_x': 0,
            'button_y': 0,
            'button_l1': 0,
            'button_r1': 0,
            'button_select': 0,
            'button_start': 0,
        }

        self._running = threading.Event()  # 控制輪詢執行緒停止
        self.thread: threading.Thread | None = None  # 背景輪詢執行緒
        self.lock = threading.Lock()  # 保護 self.state 的鎖

        print("✅ XBox Controller 物件已建立 (等待執行緒初始化)。")

    # ------------------------------------------------------------------
    def start_polling(self) -> None:
        """啟動背景執行緒來輪詢搖桿事件。"""
        if self.thread and self.thread.is_alive():
            print("搖桿輪詢執行緒已在運行中。")
            return

        self._running.set()
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        """在獨立執行緒中初始化 Pygame 並持續處理事件。"""
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)  # 建立 1x1 假視窗
        pygame.joystick.init()
        print("✅ Pygame 在專屬的搖桿執行緒中初始化完成。")

        while self._running.is_set():
            for event in pygame.event.get():
                with self.lock:
                    if event.type == pygame.JOYAXISMOTION:
                        if event.axis == 0:
                            self.state['left_analog_x'] = event.value
                        elif event.axis == 1:
                            self.state['left_analog_y'] = event.value
                        elif event.axis == 2:
                            self.state['right_analog_x'] = event.value
                        elif event.axis == 3:
                            self.state['right_analog_y'] = event.value
                    elif event.type == pygame.JOYBUTTONDOWN:
                        button_map = {
                            0: 'button_a',
                            1: 'button_b',
                            2: 'button_x',
                            3: 'button_y',
                            4: 'button_l1',
                            5: 'button_r1',
                            6: 'button_select',
                            7: 'button_start',
                        }
                        if event.button in button_map:
                            self.state[button_map[event.button]] = 1
                    elif event.type == pygame.JOYBUTTONUP:
                        button_map = {
                            0: 'button_a',
                            1: 'button_b',
                            2: 'button_x',
                            3: 'button_y',
                            4: 'button_l1',
                            5: 'button_r1',
                            6: 'button_select',
                            7: 'button_start',
                        }
                        if event.button in button_map:
                            self.state[button_map[event.button]] = 0
                    elif event.type == pygame.JOYHATMOTION:
                        if hasattr(event, 'hat') and event.hat == 0:
                            self.state['dpad'] = event.value
            time.sleep(0.01)  # 避免 CPU 佔用過高

        pygame.quit()
        print("Pygame 已安全退出。")

    def scan_and_connect(self) -> bool:
        """掃描並連接搖桿。"""
        pygame.joystick.quit()
        pygame.joystick.init()

        print("\n" + "=" * 20 + " 正在掃描搖桿 " + "=" * 20)
        if pygame.joystick.get_count() > 0:
            with self.lock:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
            print(f"✅ 成功連接到搖桿: {self.joystick.get_name()}")
            return True
        else:
            with self.lock:
                self.joystick = None
            print("--- 未偵測到任何搖桿 ---")
            return False

    def is_connected(self) -> bool:
        """檢查搖桿是否已成功初始化。"""
        with self.lock:
            return self.joystick is not None

    def get_input(self) -> dict:
        """獲取當前搖桿狀態的淺拷貝，並應用死區。"""
        with self.lock:
            state_copy = self.state.copy()

        for axis in ['left_analog_x', 'left_analog_y', 'right_analog_x', 'right_analog_y']:
            if abs(state_copy[axis]) < self.deadzone:
                state_copy[axis] = 0.0
        return state_copy

    def close(self):
        """通知輪詢執行緒停止。"""
        self._running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

