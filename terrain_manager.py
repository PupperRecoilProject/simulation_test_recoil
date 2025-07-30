# terrain_manager.py
import mujoco
import numpy as np
import random
from typing import Dict, Optional, Callable, Tuple
from datetime import datetime
from PIL import Image
from logger import log

# 為了型別提示，避免循環匯入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state import SimulationState

class TerrainTile:
    """代表地形網格中的一個地塊(Tile)的資料類別。"""
    def __init__(self, grid_x: int, grid_y: int, terrain_type: str = "Flat"):
        self.grid_x = grid_x # 在世界網格中的 x 索引
        self.grid_y = grid_y # 在世界網格中的 y 索引
        self.terrain_type = terrain_type # 地形類型名稱，例如 "Flat", "Steps"

class TerrainManager:
    """
    【雙模式版】管理無限地形和單一固定地形。
    - INFINITE 模式: 根據機器人位置動態生成無盡的隨機地形。
    - SINGLE 模式: 顯示一個由單一類型地塊組成的巨大、固定的地形。
    """
    def __init__(self, model, data):
        self.model = model # 儲存 MuJoCo 模型物件
        self.data = data # 儲存 MuJoCo 資料物件
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain') # 根據名稱 'terrain' 獲取高度場的 ID
        # 當高度場資料被改動後，需同步物理與渲染，此旗標將在 SimulationController 中被檢查
        self.needs_physics_and_scene_update = False

        if self.hfield_id == -1:  # 檢查是否成功找到高度場
            log.error("在 XML 中找不到名為 'terrain' 的 hfield。地形功能將被禁用。")
            self.is_functional = False  # 若找不到，則禁用此功能
            return

        # --- 地塊和網格設定 ---
        self.tile_resolution = 101  # 每個地塊的解析度 (e.g., 101x101 points)，奇數方便中心對稱
        self.grid_size = 5          # 可見網格的大小 (5x5)
        self.tile_world_size = 5.0  # 每個地塊在世界中的物理尺寸 (e.g., 5x5 meters)
        
        # --- 從 MuJoCo 模型讀取並驗證設定 ---
        self.hfield_nrow = model.hfield_nrow[self.hfield_id] # 從模型中獲取高度場的行數
        self.hfield_ncol = model.hfield_ncol[self.hfield_id] # 從模型中獲取高度場的列數
        self.hfield_size = model.hfield_size[self.hfield_id] # 從模型中獲取高度場的物理尺寸
        self.hfield_adr = model.hfield_adr[self.hfield_id] # 從模型中獲取高度場資料在 mjModel.hfield_data 中的起始位址

        # 驗證XML中的hfield尺寸是否符合Python腳本的預期，這是一個重要的健全性檢查
        expected_hfield_dim = (self.tile_resolution - 1) * self.grid_size + 1
        if self.hfield_nrow != expected_hfield_dim or self.hfield_ncol != expected_hfield_dim:
            log.error(
                f"XML hfield 解析度 ({self.hfield_nrow}x{self.hfield_ncol}) 與 TerrainManager 設定不符。"
            )
            log.error(
                f"     預期解析度應為: {expected_hfield_dim}x{expected_hfield_dim} (基於 {self.grid_size}x{self.grid_size} 網格)"
            )
            self.is_functional = False
            return
            
        # --- 內部狀態 ---
        self.world_center_x = 0 # 無限地形模式下，可見網格的中心地塊索引 X
        self.world_center_y = 0 # 無限地形模式下，可見網格的中心地塊索引 Y
        self.terrain_cache: Dict[Tuple[int, int], TerrainTile] = {} # 【核心】使用快取儲存所有生成過的地塊，確保地形的持久性
        self.full_hfield_data = np.zeros((self.hfield_nrow, self.hfield_ncol)) # 完整的 hfield 數據，相當於一個大畫布

        # --- 地形生成器註冊 ---
        self.terrain_generators: Dict[str, Callable] = {
            "Flat": self.generate_flat,
            "Sine Waves": self.generate_sine_waves,
            "Steps": self.generate_steps,
            "Random Noise": self.generate_random_noise,
            "Pyramid": self.generate_pyramid,
            "Stepped Pyramid": self.generate_stepped_pyramid,
        }
        
        self.terrain_types = list(self.terrain_generators.keys()) # 用於無限模式下的隨機選擇
        self.single_terrain_names = list(self.terrain_generators.keys()) # 用於單一地形模式下的循環
        
        self.is_functional = True # 標記功能為可用
        
        self.initial_generate() # 執行初始地形生成
        log.info(f"✅ 雙模式地形管理器初始化完成 (使用 {self.grid_size}x{self.grid_size} 網格)。")

    def reset(self):
        """重置地形管理器的世界中心並重新生成地形。"""
        if not self.is_functional:
            return
        log.info("正在重置地形管理器狀態...")
        self.world_center_x = 0
        self.world_center_y = 0
        self.terrain_cache.clear()
        self.initial_generate()

    def update(self, robot_pos: np.ndarray, current_mode: str):
        """
        根據地形模式決定是否更新。只在無限模式下執行滑動窗口邏輯。
        """
        if not self.is_functional or current_mode != "INFINITE":
            return # 如果功能禁用或不在無限模式，則不執行任何操作

        # 計算機器人目前所在的地塊索引
        robot_grid_x = int(round(robot_pos[0] / self.tile_world_size))
        robot_grid_y = int(round(robot_pos[1] / self.tile_world_size))
        
        # 計算機器人相對於網格中心的位置差距
        dx = robot_grid_x - self.world_center_x
        dy = robot_grid_y - self.world_center_y
        
        # 定義觸發更新的緩衝區半徑。
        # 以 5x5 網格為例，半徑為 2，減去 1 代表 1 格的安全緩衝區。
        # 當機器人走到最外圍一圈地塊時才會真正觸發滑動，避免頻繁更新。
        trigger_radius = max(0, (self.grid_size // 2) - 1)
        
        # 【除錯日誌】觀察觸發計算的詳細參數
        log.debug(
            f"TerrainCheck: RobotGrid({robot_grid_x},{robot_grid_y}), "
            f"Center({self.world_center_x},{self.world_center_y}), "
            f"Delta({dx},{dy}), Radius({trigger_radius})"
        )

        # 當機器人與中心的距離大於緩衝區半徑時，觸發更新
        if abs(dx) > trigger_radius or abs(dy) > trigger_radius:
            # 根據差距的正負決定滑動方向（+1、-1 或 0）
            shift_x = np.sign(dx).astype(int) if abs(dx) > trigger_radius else 0
            shift_y = np.sign(dy).astype(int) if abs(dy) > trigger_radius else 0

            log.info(
                f"🔄 機器人觸發地形邊界 (Delta: {dx},{dy})，" 
                f"向 ({shift_x}, {shift_y}) 方向滑動地形..."
            )
            self.shift_grid_center(shift_x, shift_y)


    def set_single_terrain(self, terrain_name: str):
        """用指定的單一地形類型填滿整個網格，創造一個巨大的、均一的固定地形。"""
        if terrain_name not in self.terrain_generators:
            log.warning(f"找不到名為 '{terrain_name}' 的地形生成器。")
            return
            
        generator = self.terrain_generators[terrain_name]
        single_tile_data = generator()  # 生成一個地塊的數據

        # 【核心修正】在繪製新的單一地形前，先把整個 hfield 畫布清零，
        # 以避免殘留上一個地形造成混合或錯亂。
        self.full_hfield_data.fill(0)

        # 將這個地塊的數據平鋪 (tile) 到整個 hfield 畫布上
        tile_res_m1 = self.tile_resolution - 1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_row = i * tile_res_m1
                start_col = j * tile_res_m1
                end_row = start_row + self.tile_resolution
                end_col = start_col + self.tile_resolution
                # 使用 maximum 確保邊界重疊部分平滑
                self.full_hfield_data[start_row:end_row, start_col:end_col] = np.maximum(
                    self.full_hfield_data[start_row:end_row, start_col:end_col],
                    single_tile_data
                )
        
        # 將數據寫入模型並標記需要同步
        self._apply_hfield_data()
        log.info(f"✅ 已生成完整的 '{terrain_name}' 地形。")

    def regenerate_terrain_and_adjust_robot(self, robot_qpos, robot_height_offset=0.3):
        """(由Y鍵觸發) 只在無限模式下，清空快取並重新生成地形，然後調整機器人高度。"""
        if not self.is_functional: return
        
        log.info("🔄 (Y Key) 正在強制重新生成所有地形...")
        self.terrain_cache.clear() # 清空所有已生成的地塊記憶
        # 將世界中心重置為機器人當前所在的地塊，以獲得最佳體驗
        self.world_center_x = int(round(robot_qpos[0] / self.tile_world_size))
        self.world_center_y = int(round(robot_qpos[1] / self.tile_world_size))

        self.initial_generate() # 重新生成初始地形
        
        # 調整機器人高度以適應新的地面
        robot_x = robot_qpos[0]
        robot_y = robot_qpos[1]
        new_ground_z = self.get_height_at(robot_x, robot_y)
        self.data.qpos[2] = new_ground_z + robot_height_offset
        log.info(f"    機器人高度已調整以適應新地形：Z = {self.data.qpos[2]:.2f}m")
        mujoco.mj_forward(self.model, self.data) # 確保高度變更生效

    def get_current_terrain_name(self, state: 'SimulationState') -> str:
        """根據模式顯示不同的地形名稱，用於UI顯示。"""
        if not self.is_functional: return "N/A (hfield missing)"
        
        if state.terrain_mode == "INFINITE":
            center_tile = self.terrain_cache.get((self.world_center_x, self.world_center_y))
            if center_tile:
                return f"INFINITE (Center: {center_tile.terrain_type})"
            return "INFINITE (Unknown)"
        else: # SINGLE 模式
            terrain_name = self.single_terrain_names[state.single_terrain_index]
            return f"SINGLE ({terrain_name})"

    def get_current_terrain_name_simple(self, state: 'SimulationState') -> str:
        """只回傳地形名稱本身，供 UI 狀態比較用。"""
        if not self.is_functional:
            return "N/A"
        if state.terrain_mode == "INFINITE":
            return "INFINITE"
        if 0 <= state.single_terrain_index < len(self.single_terrain_names):
            return self.single_terrain_names[state.single_terrain_index]
        return "Unknown"

    def shift_grid_center(self, dx: int, dy: int):
        """平移世界的中心，並重新繪製整個 hfield。"""
        self.world_center_x += dx
        self.world_center_y += dy
        self.update_hfield()

    def _apply_hfield_data(self):
        """將 full_hfield_data 寫入模型並標記需要物理與渲染更新。"""
        # 將 NumPy 陣列內容複製到 mjModel 中
        self.model.hfield_data[
            self.hfield_adr : self.hfield_adr + self.hfield_nrow * self.hfield_ncol
        ] = self.full_hfield_data.flatten()
        # 在下一個模擬步中由 SimulationController 進行同步
        self.needs_physics_and_scene_update = True

    def get_or_generate_tile(self, grid_x: int, grid_y: int) -> TerrainTile:
        """如果地塊已在快取中，則返回它；否則，生成新地塊並存入快取。"""
        if (grid_x, grid_y) in self.terrain_cache:
            return self.terrain_cache[(grid_x, grid_y)]
        
        neighbor_types = set()
        for offset_x, offset_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_tile = self.terrain_cache.get((grid_x + offset_x, grid_y + offset_y))
            if neighbor_tile:
                neighbor_types.add(neighbor_tile.terrain_type)
            
        available_types = [t for t in self.terrain_types if t not in neighbor_types]
        if not available_types:
            available_types = self.terrain_types
            
        chosen_type = random.choice(available_types)
        new_tile = TerrainTile(grid_x, grid_y, chosen_type)
        self.terrain_cache[(grid_x, grid_y)] = new_tile
        return new_tile

    def update_hfield(self):
        """根據當前世界中心，從快取中讀取或生成地塊，並繪製到 hfield 畫布上。"""
        radius = self.grid_size // 2
        tile_res_m1 = self.tile_resolution - 1
        self.full_hfield_data.fill(0)

        for gx_offset in range(-radius, radius + 1):
            for gy_offset in range(-radius, radius + 1):
                world_gx = self.world_center_x + gx_offset
                world_gy = self.world_center_y + gy_offset
                tile = self.get_or_generate_tile(world_gx, world_gy)
                generator = self.terrain_generators[tile.terrain_type]
                tile_data = generator()
                start_row = (gy_offset + radius) * tile_res_m1
                start_col = (gx_offset + radius) * tile_res_m1
                end_row = start_row + self.tile_resolution
                end_col = start_col + self.tile_resolution
                self.full_hfield_data[start_row:end_row, start_col:end_col] = np.maximum(
                    self.full_hfield_data[start_row:end_row, start_col:end_col],
                    tile_data
                )
                
        # 將數據寫入模型並標記需要同步
        self._apply_hfield_data()
        log.info("✅ 完整高度場已更新。")

    def initial_generate(self):
        """首次生成時，只需更新一次 hfield。"""
        log.info("🏞️ 正在生成初始地形...")
        self.update_hfield()

    def get_height_at(self, world_x: float, world_y: float) -> float:
        """查詢世界座標 (x, y) 對應的地形高度。"""
        if not self.is_functional: return 0.0
        total_size_x = self.hfield_size[0] * 2
        total_size_y = self.hfield_size[1] * 2
        norm_x = (world_x / total_size_x) + 0.5
        norm_y = (world_y / total_size_y) + 0.5
        col = int(norm_x * (self.hfield_ncol - 1))
        row = int(norm_y * (self.hfield_nrow - 1))
        if not (0 <= row < self.hfield_nrow and 0 <= col < self.hfield_ncol): return 0.0
        return self.full_hfield_data[row, col]

    def save_hfield_to_png(self):
        """將當前完整的高度場數據儲存為一個灰階PNG檔案。"""
        if not self.is_functional:
            log.warning("地形功能未啟用，無法儲存PNG。")
            return
        log.info("💾 正在儲存當前地形為PNG檔案...")
        data = self.full_hfield_data
        h_min, h_max = data.min(), data.max()
        if h_max == h_min:
            normalized_data = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized_data = (data - h_min) / (h_max - h_min)
            normalized_data = (normalized_data * 255).astype(np.uint8)
        img = Image.fromarray(normalized_data, 'L')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"terrain_snapshot_{timestamp}.png"
        img.save(filename)
        log.info(f"✅ 地形快照已成功儲存至: {filename}")

    def _create_boundary_fade(self) -> np.ndarray:
        """創建一個邊界為0，中心為1的2D遮罩，用於確保地塊邊界高度為零。"""
        fade_margin_ratio = 0.2
        fade_size = int(self.tile_resolution * fade_margin_ratio)
        fade_curve = np.linspace(0.0, 1.0, fade_size)
        mask1d = np.ones(self.tile_resolution)
        mask1d[:fade_size] = fade_curve
        mask1d[-fade_size:] = np.flip(fade_curve)
        mask2d = np.minimum(np.ones((self.tile_resolution, self.tile_resolution)) * mask1d, (np.ones((self.tile_resolution, self.tile_resolution)) * mask1d).T)
        return mask2d

    def generate_flat(self):
        """生成一個完全平坦的地形。"""
        return np.zeros((self.tile_resolution, self.tile_resolution))

    def generate_sine_waves(self):
        """生成由正弦波組成的波浪狀地形，並應用邊界淡出。"""
        x = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        y = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        height_field = 0.08 * (np.sin(X) + np.sin(Y))
        return height_field * self._create_boundary_fade()

    def generate_steps(self):
        """生成階梯狀地形，並應用邊界淡出。"""
        hfield = np.zeros((self.tile_resolution, self.tile_resolution))
        step_height = 0.05
        num_steps = random.randint(5, 10)
        step_width = self.tile_resolution // num_steps
        for i in range(num_steps):
            hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield * self._create_boundary_fade()

    def generate_random_noise(self):
        """生成隨機的崎嶇地形，並應用邊界淡出。"""
        noise = np.random.rand(self.tile_resolution, self.tile_resolution) * 0.1
        return noise * self._create_boundary_fade()


    def generate_pyramid(self):
        """生成一個中央高、四周低的正金字塔地形。"""
        max_height = random.uniform(0.3, 0.6)
        x = np.linspace(-1, 1, self.tile_resolution)
        y = np.linspace(-1, 1, self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.maximum(np.abs(X), np.abs(Y))
        hfield_data = max_height * (1 - dist)
        # 直接回傳未平滑的金字塔高度
        return hfield_data

    def generate_stepped_pyramid(self):
        """
        【高度量化版】生成一個中央高、四周低的階梯狀金字塔地形。
        
        這個函式不使用迴圈，而是透過對一個連續的、平滑的金字塔高度場
        進行數學上的 "量化" 處理，來高效地生成階梯效果。
        """
        # --- 1. 初始化隨機參數 ---
        # 隨機決定金字塔的階梯數，增加地形的多樣性
        num_steps = random.randint(12, 16)
        # 隨機決定金字塔的總高度
        max_height = random.uniform(0.4, 0.8)
        # 根據總高度和階梯數，計算出每一階的固定高度
        step_height = max_height / num_steps

        # --- 2. 建立一個標準化的 2D 座標網格 ---
        # 建立一個從 -1 到 1 的線性序列，代表 x 軸座標
        x = np.linspace(-1, 1, self.tile_resolution)
        # 建立一個從 -1 到 1 的線性序列，代表 y 軸座標
        y = np.linspace(-1, 1, self.tile_resolution)
        # 使用 meshgrid 將 x 和 y 向量擴展成 2D 矩陣，代表網格上每個點的 (X, Y) 座標
        # 這個網格的中心是 (0, 0)
        X, Y = np.meshgrid(x, y)

        # --- 3. 計算到中心的方形距離 (Chebyshev Distance) ---
        # np.maximum(np.abs(X), np.abs(Y)) 計算的是每個點到中心 (0,0) 的切比雪夫距離。
        # 這種距離的等高線是正方形，這正是我們需要金字塔是方形底座而非圓形底座的關鍵。
        # dist 矩陣的值從中心點的 0 線性增加到網格邊緣的 1。
        dist = np.maximum(np.abs(X), np.abs(Y))

        # --- 4. 生成一個平滑、連續的金字塔斜坡 ---
        # 根據到中心的距離，計算出一個完美的、無階梯的金字塔。
        # 在中心 (dist=0)，高度為 max_height。
        # 在邊緣 (dist=1)，高度為 0。
        continuous_hfield = max_height * (1.0 - dist)
        
        # --- 5. 【核心步驟】高度量化，創造階梯效果 ---
        # (a) 計算每個點的連續高度相當於 "多少個階梯" (結果是浮點數)
        height_in_steps = continuous_hfield / step_height
        
        # (b) 使用 np.ceil() (向上取整) 將浮點階梯數轉換為整數階梯數。
        #     例如，高度為 2.1 階或 2.8 階的點，都會被歸為第 3 階。
        #     這一步驟創造了階梯的平坦頂部。
        quantized_steps = np.ceil(height_in_steps)
        
        # (c) 將取整後的階梯數乘以單階高度，得到最終的離散階梯高度。
        hfield_data = quantized_steps * step_height
        
        # --- 6. 確保邊界絕對為零 ---
        # 由於浮點數計算可能存在微小誤差，此步驟強制將網格最外圍的高度設為 0，
        # 以確保與相鄰地塊拼接時完美無縫。
        hfield_data[dist >= 1.0] = 0.0

        # 返回最終生成的階梯金字塔高度數據
        return hfield_data
