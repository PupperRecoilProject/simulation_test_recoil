import numpy as np
import time

class MockMjData:
    """模擬 MuJoCo 的 MjData 物件，提供必需的屬性"""
    def __init__(self, num_motors: int):
        self.time = 0.0
        self.qpos = np.zeros(7 + num_motors)
        self.qpos[2] = 0.3
        self.qpos[3] = 1.0
        self.xpos = np.zeros((100, 3))
        self.xquat = np.zeros((100, 4))
        self.xquat[:, 0] = 1.0
        self.ctrl = np.zeros(num_motors)

    def body(self, name: str):
        class MockBody:
            def __init__(self):
                self.xpos = np.array([0.0, 0.0, 0.3])
                self.xquat = np.array([1.0, 0.0, 0.0, 0.0])
        return MockBody()

class MockSimulation:
    """無需 MuJoCo 的模擬器，用於 headless 測試"""
    def __init__(self, config):
        print("--- [MOCK] 使用 MockSimulation 進行無頭運行 ---")
        self.config = config
        self.data = MockMjData(config.num_motors)
        self.default_pose = np.zeros(config.num_motors)
        self.torso_id = 1
        self._last_render_time = time.perf_counter()

    def initialize_window_and_context(self):
        print("[MOCK] initialize_window_and_context() called. Doing nothing.")

    def register_callbacks(self, keyboard_handler):
        print("[MOCK] register_callbacks() called. Doing nothing.")

    def should_close(self):
        return False

    def apply_position_control(self, target_pos, params):
        pass

    def render_from_thread(self, state):
        current = time.perf_counter()
        dt = current - self._last_render_time
        self._last_render_time = current
        self.data.time += dt
        state.latest_pos[0] += state.command[1] * dt * 0.1
        state.latest_pos[1] += state.command[0] * dt * 0.1
        time.sleep(1.0/60.0)

    def close(self):
        print("[MOCK] close() called. Shutting down.")
