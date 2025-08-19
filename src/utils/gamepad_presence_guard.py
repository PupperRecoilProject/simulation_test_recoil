import threading
import time


def start_gamepad_presence_guard(state, interval_sec: float = 0.5):
    """啟動搖桿存在偵測守門員，定期更新 state.ui_gamepad_connected。"""
    try:
        import pygame
    except Exception:
        return None  # 沒有 pygame 模組就直接略過

    state.ui_gamepad_connected = False

    def _loop():
        pygame.joystick.init()
        while True:
            try:
                n = pygame.joystick.get_count()
                state.ui_gamepad_connected = n > 0
            except Exception:
                state.ui_gamepad_connected = False
            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t
