# src/mock/mock_simulation.py

import numpy as np
import time

# 【保留】 MockMjData 類
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

# 【保留】 MockSimulation 類
class MockSimulation:
    """無需 MuJoCo 的模擬器，用於 headless 測試"""
    def __init__(self, config):
        print("--- [MOCK] 使用 MockSimulation 進行無頭運行 ---")
        self.config = config
        self.data = MockMjData(config.num_motors)
        self.default_pose = np.zeros(config.num_motors)
        self.torso_id = 1
        # 【v4.3.2 新增】 為 MockSimulation 也添加 accelerometer_id 屬性
        self.accelerometer_id = -1
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
        
    def poll_window_events(self):
        # 【v4.3.2 新增】 為 MockSimulation 添加 poll_window_events 方法
        time.sleep(0.01)

# 【保留】 MockFloatingController 類
class MockFloatingController:
    """假浮空控制器，僅提供介面"""
    def __init__(self, *args, **kwargs):
        print("FloatingController disabled in mock mode")
    def enable(self, *args, **kwargs):
        pass
    def disable(self, *args, **kwargs):
        pass

# 【保留】 MockTerrainManager 類
class MockTerrainManager:
    """假地形管理器"""
    def __init__(self, *args, **kwargs):
        print("TerrainManager disabled in mock mode")
        self.is_functional = False
        # 在無頭模式下提供空的地形列表以供 UI 使用
        self.single_terrain_names = []
    def get_current_terrain_name_simple(self, state):
        return "N/A"
    def update(self, *args, **kwargs):
        pass

# 【v4.3.2 刪除】 移除 MockObservationBuilder
# class MockObservationBuilder:
#    ...

# 【v4.3.2 新增】 新增 MockObservationManager 
class MockObservationManager:
    """ 
    【v4.3.2 新增】
    ObservationManager 的 Mock 版本，用於無頭模式。 
    它提供與真實版本相同的 API 接口，但內部不做任何計算。 
    【v4.3.2 之後的修正】補上缺失的 ALL_OBS_DIMS 屬性以修復 ONNX 模型載入錯誤。 
    """ 
    # 從真實的 ObservationManager 複製過來的屬性 
     # 這是解決 'MockObservationManager' object has no attribute 'ALL_OBS_DIMS' 錯誤的關鍵 
    ALL_OBS_DIMS: dict[str, int] = { 
        'gravity_vector': 3, 
        'commands_3d': 3,
        'commands_4d': 4, 
        'joint_positions': 12,
        'joint_velocities': 12, 
        'last_action': 12, 
        'angular_velocity': 3, 
        'linear_velocity': 3, 
        'accelerometer': 3, 
        'current_pitch': 1, 
        'firearm_recoil_warming': 1, 
        'z_angular_velocity': 1, 
        'foot_contact_states': 4, 
        'phase_signal': 1,
    } 

    def __init__(self, *args, **kwargs): 
        self.recipe = [] 
        self.component_dims = {} 
        print("ObservationManager disabled in mock mode") 

    def set_recipe(self, recipe): 
        """Mock set_recipe 方法""" 
        # 即使是 Mock，也最好模擬一下真實行為 
        self.recipe = recipe 
        print(f"[MOCK] Observation recipe set to: {recipe}") 

    def get_observation(self, *args, **kwargs) -> np.ndarray: 
        """Mock get_observation 方法，永遠返回一個符合維度的零向量""" 
        # 雖然目前用不到，但一個更健壯的 Mock 應該返回正確形狀的陣列 
        total_dims = sum(self.ALL_OBS_DIMS[key] for key in self.recipe) 
        return np.zeros(total_dims, dtype=np.float32)

    def update_all_observations(self):
        """【v4.10.x 錯誤修正】補上缺失的方法以匹配真實 ObservationManager 的 API。"""
        # 在無頭/模擬模式下，這個方法不需要執行任何操作，
        # 只需要存在即可避免 AttributeError。
        pass # 提供一個空的實現，防止程式在 SimulationController 中崩潰