# terrain_manager.py
import mujoco
import numpy as np

class TerrainManager:
    """
    管理和動態切換 MuJoCo 高度場 (hfield) 地形。
    """
    def __init__(self, model, data):
        self.model = model # 儲存 MuJoCo 模型物件
        self.data = data # 儲存 MuJoCo 資料物件
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain') # 根據名稱 'terrain' 獲取高度場的 ID
        
        self.needs_scene_update = False # 標記是否需要更新渲染場景
        
        if self.hfield_id == -1: # 檢查是否成功找到高度場
            print("❌ 錯誤: 在 XML 中找不到名為 'terrain' 的 hfield。地形切換功能將被禁用。")
            self.is_functional = False # 若找不到，則禁用此功能
            return

        self.nrow = model.hfield_nrow[self.hfield_id] # 從模型中獲取高度場的行數
        self.ncol = model.hfield_ncol[self.hfield_id] # 從模型中獲取高度場的列數
        self.size = model.hfield_size[self.hfield_id] # 從模型中獲取高度場的物理尺寸
        self.adr = model.hfield_adr[self.hfield_id] # 從模型中獲取高度場資料在 mjModel.hfield_data 中的起始位址
        
        # 【修改】在地形字典中加入新的金字塔地形
        self.terrains = {
            "Flat": self.generate_flat,
            "Sine Waves": self.generate_sine_waves,
            "Steps": self.generate_steps,
            "Random Noise": self.generate_random_noise,
            "Pyramid": self.generate_pyramid, # <-- 新增金字塔地形
        }
        self.terrain_names = list(self.terrains.keys()) # 獲取所有地形的名稱列表
        self.current_terrain_index = 0 # 將目前地形索引初始化為 0
        self.is_functional = True # 標記功能為可用
        
        self.switch_terrain(0) # 初始化時，切換到第一個地形（平地）
        print("✅ 地形管理器初始化完成 (使用高度場)。")

    def generate_flat(self):
        """生成一個完全平坦的地形。"""
        return np.zeros((self.nrow, self.ncol)) # 回傳一個全為 0 的二維陣列

    def generate_sine_waves(self):
        """生成由正弦波組成的波浪狀地形。"""
        x = np.linspace(0, 2 * np.pi, self.ncol) # 建立 x 軸座標
        y = np.linspace(0, 2 * np.pi, self.nrow) # 建立 y 軸座標
        X, Y = np.meshgrid(x, y) # 建立二維網格
        return 0.05 * (np.sin(X * 3) + np.sin(Y * 2)) # 計算每個點的高度

    def generate_steps(self):
        """生成階梯狀地形。"""
        hfield = np.zeros((self.nrow, self.ncol)) # 初始化為平地
        step_height = 0.03 # 設定每個階梯的高度
        step_width = self.nrow // 10 # 設定每個階梯的寬度
        for i in range(10): # 循環生成 10 個階梯
            hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield # 回傳階梯地形資料

    def generate_random_noise(self):
        """生成隨機的崎嶇地形。"""
        return np.random.rand(self.nrow, self.ncol) * 0.05 # 生成 0 到 0.05 之間的隨機高度

    # 【新增】生成金字塔地形的函式
    def generate_pyramid(self):
        """生成一個中央高、四周低的正金字塔地形。"""
        max_height = 1.0  # 設定金字塔的最高點高度（單位：米）
        
        # 建立從 -1 到 1 的標準化座標系，這樣中心點就是 (0,0)
        x = np.linspace(-1, 1, self.ncol)
        y = np.linspace(-1, 1, self.nrow)
        # 使用 meshgrid 產生二維座標網格
        X, Y = np.meshgrid(x, y)
        
        # 計算每個點到中心的切比雪夫距離 (Chebyshev distance)，即 max(|dx|, |dy|)
        # 這會形成一個方形的等高線，正好是金字塔的形狀
        dist = np.maximum(np.abs(X), np.abs(Y))
        
        # 高度 = 最高高度 * (1 - 距離)
        # 在中心點 (dist=0)，高度為 max_height
        # 在邊界 (dist=1)，高度為 0
        hfield_data = max_height * (1 - dist)
        
        return hfield_data # 回傳金字塔高度場資料

    def cycle_terrain(self):
        """循環切換到下一個地形。"""
        if not self.is_functional: return # 如果功能未啟用，直接返回
        self.current_terrain_index = (self.current_terrain_index + 1) % len(self.terrain_names) # 計算下一個地形的索引
        self.switch_terrain(self.current_terrain_index) # 切換到該地形

    def get_current_terrain_name(self):
        """獲取目前地形的名稱。"""
        if not self.is_functional: return "N/A (hfield missing)" # 如果功能未啟用，返回提示訊息
        return self.terrain_names[self.current_terrain_index] # 返回目前地形的名稱

    def switch_terrain(self, index):
        """切換到指定索引的地形。"""
        if not self.is_functional: return # 如果功能未啟用，直接返回
        
        terrain_name = self.terrain_names[index] # 獲取地形名稱
        print(f"🏞️ 切換地形至: {terrain_name}") # 在控制台輸出提示
        
        generator = self.terrains[terrain_name] # 根據名稱獲取對應的生成函式
        hfield_data = generator() # 呼叫生成函式，產生高度資料
        
        # 將新生成的高度資料寫入 MuJoCo 模型的高度場資料緩衝區
        self.model.hfield_data[self.adr:self.adr + self.nrow*self.ncol] = hfield_data.flatten()
        
        self.needs_scene_update = True # 設定標記，通知渲染迴圈需要更新場景