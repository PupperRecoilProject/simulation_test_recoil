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
        
        # 初始化為平地
        self.switch_terrain(0)
        print("✅ 地形管理器初始化完成 (使用高度場)。")

    def generate_flat(self):
        """生成平坦地形。"""
        return np.zeros((self.nrow, self.ncol))

    def generate_sine_waves(self):
        """生成正弦波地形。"""
        x = np.linspace(0, 2 * np.pi, self.ncol)
        y = np.linspace(0, 2 * np.pi, self.nrow)
        X, Y = np.meshgrid(x, y)
        # 振幅 0.05，頻率調整
        return 0.05 * (np.sin(X * 3) + np.sin(Y * 2))

    def generate_steps(self):
        """生成階梯地形。"""
        hfield = np.zeros((self.nrow, self.ncol))
        step_height = 0.03
        step_width = self.nrow // 10
        for i in range(10):
            hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield

    def generate_random_noise(self):
        """生成隨機噪音地形。"""
        return np.random.rand(self.nrow, self.ncol) * 0.05

    def cycle_terrain(self):
        """循環切換到下一個地形。"""
        if not self.is_functional: return
        self.current_terrain_index = (self.current_terrain_index + 1) % len(self.terrain_names)
        self.switch_terrain(self.current_terrain_index)

    def get_current_terrain_name(self):
        if not self.is_functional: return "N/A (hfield missing)"
        return self.terrain_names[self.current_terrain_index]

    def switch_terrain(self, index):
        """切換到指定索引的地形。"""
        if not self.is_functional: return
        
        terrain_name = self.terrain_names[index]
        print(f"🏞️ 切換地形至: {terrain_name}")
        
        generator = self.terrains[terrain_name]
        hfield_data = generator()
        
        # 將地形數據複製到 MuJoCo 的 hfield_data 中
        # 注意：MuJoCo 的 hfield_data 是一個一維數組
        self.model.hfield_data[self.adr:self.adr + self.nrow*self.ncol] = hfield_data.flatten()
        
        # 重要：如果模擬正在運行，我們需要通知 MuJoCo 更新碰撞幾何體
        # 在 mj_step 或 mj_forward 之前呼叫這個函式
        if self.data.time > 0:
             mujoco.mj_forward(self.model, self.data)