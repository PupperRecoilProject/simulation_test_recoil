# terrain_manager.py
import mujoco
import numpy as np

class TerrainManager:
    """
    管理和動態切換 MuJoCo 高度場 (hfield) 地形。
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
        
        self.needs_scene_update = False
        
        if self.hfield_id == -1:
            print("❌ 錯誤: 在 XML 中找不到名為 'terrain' 的 hfield。地形切換功能將被禁用。")
            self.is_functional = False
            return

        self.nrow = model.hfield_nrow[self.hfield_id]
        self.ncol = model.hfield_ncol[self.hfield_id]
        self.size = model.hfield_size[self.hfield_id]
        self.adr = model.hfield_adr[self.hfield_id]
        
        self.terrains = {
            "Flat": self.generate_flat,
            "Sine Waves": self.generate_sine_waves,
            "Steps": self.generate_steps,
            "Random Noise": self.generate_random_noise
        }
        self.terrain_names = list(self.terrains.keys())
        self.current_terrain_index = 0
        self.is_functional = True
        
        self.switch_terrain(0)
        print("✅ 地形管理器初始化完成 (使用高度場)。")

    def generate_flat(self):
        return np.zeros((self.nrow, self.ncol))

    def generate_sine_waves(self):
        x = np.linspace(0, 2 * np.pi, self.ncol)
        y = np.linspace(0, 2 * np.pi, self.nrow)
        X, Y = np.meshgrid(x, y)
        return 0.05 * (np.sin(X * 3) + np.sin(Y * 2))

    def generate_steps(self):
        hfield = np.zeros((self.nrow, self.ncol))
        step_height = 0.03
        step_width = self.nrow // 10
        for i in range(10):
            hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield

    def generate_random_noise(self):
        return np.random.rand(self.nrow, self.ncol) * 0.05

    def cycle_terrain(self):
        if not self.is_functional: return
        self.current_terrain_index = (self.current_terrain_index + 1) % len(self.terrain_names)
        self.switch_terrain(self.current_terrain_index)

    def get_current_terrain_name(self):
        if not self.is_functional: return "N/A (hfield missing)"
        return self.terrain_names[self.current_terrain_index]

    def switch_terrain(self, index):
        if not self.is_functional: return
        
        terrain_name = self.terrain_names[index]
        print(f"🏞️ 切換地形至: {terrain_name}")
        
        generator = self.terrains[terrain_name]
        hfield_data = generator()
        
        self.model.hfield_data[self.adr:self.adr + self.nrow*self.ncol] = hfield_data.flatten()
        
        self.needs_scene_update = True