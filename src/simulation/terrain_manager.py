# terrain_manager.py
import mujoco
import numpy as np
import random
import os # 【新增】導入 os 模組
from typing import Dict, Optional, Callable, Tuple
from datetime import datetime
from PIL import Image
from src.core.logger import log

# 【v4.5.0 最終權威修正】 新增對 event_bus 和相關事件的導入
from src.core.event_system import event_bus, EVENT_SIMULATION_RELOAD_REQUESTED

# 為了型別提示，避免循環匯入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.state import SimulationState
    # 【v4.5.0 修正】 導入 MjData 類型提示
    from mujoco import MjData

class TerrainTile:
    """代表地形網格中的一個地塊(Tile)的資料類別。"""
    def __init__(self, grid_x: int, grid_y: int, terrain_type: str = "Flat"):
        self.grid_x = grid_x # 在世界網格中的 x 索引
        self.grid_y = grid_y # 在世界網格中的 y 索引
        self.terrain_type = terrain_type # 地形類型名稱，例如 "Flat", "Steps"

class TerrainManager:
    """
    【v4.5.0 最終修正版】
    管理地形生成。它將生成的地形數據寫入一個臨時 PNG 檔案，
    並發布一個事件來請求最高層協調者完整地重載模擬環境。
    """
    def __init__(self, model, data):
        self.model = model
        # 【v4.5.0 修正】 TerrainManager 不應再持有對 MjData 的直接引用，以遵守唯一所有權原則
        # self.data = data
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
        
        # 【v4.5.0 新增】定義 hfield PNG 的固定路徑，以便重載時讀取
        self.hfield_file_path = os.path.join("assets", "mesh", "current_hfield.png")

        if self.hfield_id == -1:
            log.warning("在 XML 中找不到名為 'terrain' 的 hfield。動態地形功能將被禁用。")
            self.is_functional = False
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
            log.error(f"XML hfield 解析度與 TerrainManager 設定不符。")
            self.is_functional = False
            return
            
        # --- 內部狀態 ---
        self.world_center_x, self.world_center_y = 0, 0
        self.terrain_cache: Dict[Tuple[int, int], TerrainTile] = {}
        self.full_hfield_data = np.zeros((self.hfield_nrow, self.hfield_ncol))
        
        # 【v4.5.1 修正】 補上 rendering_thread.py 所需的遺漏屬性
        self.needs_physics_and_scene_update = False

        # --- 地形生成器註冊 ---
        self.terrain_generators: Dict[str, Callable] = {
            "Flat": self.generate_flat, "Sine Waves": self.generate_sine_waves,
            "Steps": self.generate_steps, "Random Noise": self.generate_random_noise,
            "Pyramid": self.generate_pyramid, "Stepped Pyramid": self.generate_stepped_pyramid,
        }
        self.terrain_types = list(self.terrain_generators.keys())
        self.single_terrain_names = list(self.terrain_generators.keys())
        self.is_functional = True
        log.info(f"✅ 地形管理器初始化完成。")

    def _apply_hfield_data(self):
        """
        將生成的 hfield 數據寫入模型，並設定旗標以通知渲染執行緒更新 GPU 資源。
        """
        # 將計算好的地形數據寫入 MuJoCo 模型結構中
        self.model.hfield_data[
            self.hfield_adr : self.hfield_adr + self.hfield_nrow * self.hfield_ncol
        ] = self.full_hfield_data.flatten()
        
        # 設定旗標，告知 RenderingThread 需要將更新後的高度場數據上傳到 GPU
        self.needs_physics_and_scene_update = True
        
        # 【v4.5.0 移除】 不再發布 notification.model_recompiled 事件，
        # 因為我們不再需要完整的模擬重載，只需要渲染更新。
        # event_bus.publish("notification.model_recompiled")

    def reset(self):
        """重置地形管理器的狀態到初始狀態。"""
        if not self.is_functional: return
        log.info("正在重置地形管理器狀態...")
        self.world_center_x, self.world_center_y = 0, 0
        self.terrain_cache.clear() # 清空地形快取
        self.initial_generate() # 重新生成初始地形

    def update(self, robot_pos: np.ndarray, current_mode: str):
        """根據機器人位置更新無限地形網格。"""
        if not self.is_functional or current_mode != "INFINITE": return # 只在無限地形模式下作用
        # 計算機器人所在的網格座標
        robot_grid_x = int(round(robot_pos[0] / self.tile_world_size))
        robot_grid_y = int(round(robot_pos[1] / self.tile_world_size))
        # 計算與當前網格中心的偏移量
        dx, dy = robot_grid_x - self.world_center_x, robot_grid_y - self.world_center_y
        trigger_radius = max(0, (self.grid_size // 2) - 1) # 設定觸發半徑
        # 如果偏移量超過觸發半徑，則移動網格中心
        if abs(dx) > trigger_radius or abs(dy) > trigger_radius:
            # 根據差距的正負決定滑動方向（+1、-1 或 0）
            shift_x = np.sign(dx).astype(int) if abs(dx) > trigger_radius else 0
            shift_y = np.sign(dy).astype(int) if abs(dy) > trigger_radius else 0
            self.shift_grid_center(shift_x, shift_y)


    def set_single_terrain(self, terrain_name: str):
        """生成並應用單一類型的靜態地形。"""
        if terrain_name not in self.terrain_generators: return # 檢查地形名稱是否存在
        generator = self.terrain_generators[terrain_name] # 獲取對應的生成器函式
        single_tile_data = generator() # 生成單個地塊的數據
        self.full_hfield_data.fill(0) # 清空完整高度場
        tile_res_m1 = self.tile_resolution - 1
        # 將單個地塊的數據複製到整個網格
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_row, start_col = i * tile_res_m1, j * tile_res_m1
                end_row, end_col = start_row + self.tile_resolution, start_col + self.tile_resolution
                # 使用 maximum 確保邊界平滑過渡
                self.full_hfield_data[start_row:end_row, start_col:end_col] = np.maximum(
                    self.full_hfield_data[start_row:end_row, start_col:end_col], single_tile_data)
        self._apply_hfield_data() # 應用數據
        log.info(f"✅ 已請求生成 '{terrain_name}' 地形。")

    def regenerate_terrain_and_adjust_robot(self, sim_data: 'MjData', robot_qpos: np.ndarray, robot_height_offset=0.3):
        """【v4.5.0 修正】 接收 sim_data 作為參數，不再依賴 self.data"""
        if not self.is_functional: return
        log.info("🔄 (Y Key) 正在強制重新生成所有地形...")
        self.terrain_cache.clear()
        # 將世界中心移動到機器人當前位置
        self.world_center_x = int(round(robot_qpos[0] / self.tile_world_size))
        self.world_center_y = int(round(robot_qpos[1] / self.tile_world_size))
        self.initial_generate()
        # 重新計算機器人腳下的地面高度並調整其 Z 座標
        new_ground_z = self.get_height_at(robot_qpos[0], robot_qpos[1])
        sim_data.qpos[2] = new_ground_z + robot_height_offset
        mujoco.mj_forward(self.model, sim_data)

    def get_current_terrain_name(self, state: 'SimulationState') -> str:
        """獲取當前地形的詳細名稱，用於 UI 顯示。"""
        if not self.is_functional: return "N/A (hfield missing)"
        
        if state.terrain_mode == "INFINITE":
            center_tile = self.terrain_cache.get((self.world_center_x, self.world_center_y))
            return f"INFINITE (Center: {center_tile.terrain_type})" if center_tile else "INFINITE (Unknown)"
        else:
            return f"SINGLE ({self.single_terrain_names[state.single_terrain_index]})"

    def get_current_terrain_name_simple(self, state: 'SimulationState') -> str:
        """獲取當前地形的簡化名稱。"""
        if not self.is_functional: return "N/A"
        if state.terrain_mode == "INFINITE": return "INFINITE"
        if 0 <= state.single_terrain_index < len(self.single_terrain_names):
            return self.single_terrain_names[state.single_terrain_index]
        return "Unknown"

    def shift_grid_center(self, dx: int, dy: int):
        """移動地形網格的中心。"""
        self.world_center_x += dx
        self.world_center_y += dy
        self.update_hfield()

    def get_or_generate_tile(self, grid_x: int, grid_y: int) -> TerrainTile:
        """獲取或生成指定網格座標的地塊。"""
        if (grid_x, grid_y) in self.terrain_cache:
            return self.terrain_cache[(grid_x, grid_y)]
        # 避免生成與鄰居相同的地形類型以增加多樣性
        neighbor_types = {self.terrain_cache.get((grid_x + ox, grid_y + oy)).terrain_type for ox, oy in [(0,1), (0,-1), (1,0), (-1,0)] if (grid_x + ox, grid_y + oy) in self.terrain_cache}
        available_types = [t for t in self.terrain_types if t not in neighbor_types] or self.terrain_types
        chosen_type = random.choice(available_types)
        new_tile = TerrainTile(grid_x, grid_y, chosen_type)
        self.terrain_cache[(grid_x, grid_y)] = new_tile
        return new_tile

    def update_hfield(self):
        """根據當前網格中心和快取更新完整的高度場數據。"""
        radius, tile_res_m1 = self.grid_size // 2, self.tile_resolution - 1
        self.full_hfield_data.fill(0)
        # 遍歷整個網格，填充地形數據
        for gx_offset in range(-radius, radius + 1):
            for gy_offset in range(-radius, radius + 1):
                world_gx, world_gy = self.world_center_x + gx_offset, self.world_center_y + gy_offset
                tile = self.get_or_generate_tile(world_gx, world_gy)
                tile_data = self.terrain_generators[tile.terrain_type]()
                start_row, start_col = (gy_offset + radius) * tile_res_m1, (gx_offset + radius) * tile_res_m1
                end_row, end_col = start_row + self.tile_resolution, start_col + self.tile_resolution
                self.full_hfield_data[start_row:end_row, start_col:end_col] = np.maximum(
                    self.full_hfield_data[start_row:end_row, start_col:end_col], tile_data)
        self._apply_hfield_data() # 應用更新後的數據
        log.info("✅ 完整高度場已更新。")

    def initial_generate(self):
        """生成初始地形。"""
        log.info("🏞️ 正在生成初始地形...")
        self.update_hfield()

    def get_height_at(self, world_x: float, world_y: float) -> float:
        """查詢世界座標 (x, y) 對應的地形高度。"""
        if not self.is_functional: return 0.0
        total_size_x, total_size_y = self.hfield_size[0] * 2, self.hfield_size[1] * 2
        # 將世界座標轉換為高度場的陣列索引
        norm_x, norm_y = (world_x / total_size_x) + 0.5, (world_y / total_size_y) + 0.5
        col, row = int(norm_x * (self.hfield_ncol - 1)), int(norm_y * (self.hfield_nrow - 1))
        if not (0 <= row < self.hfield_nrow and 0 <= col < self.hfield_ncol): return 0.0
        return self.full_hfield_data[row, col]

    def save_hfield_to_png(self):
        """將當前的高度場數據儲存為 PNG 圖片。"""
        if not self.is_functional: return
        log.info("💾 正在儲存當前地形為PNG檔案...")
        data = self.full_hfield_data
        h_min, h_max = data.min(), data.max()
        # 將高度數據正規化到 0-255 的灰度範圍
        normalized_data = np.zeros_like(data, dtype=np.uint8) if h_max == h_min else ((data - h_min) / (h_max - h_min) * 255).astype(np.uint8)
        img = Image.fromarray(normalized_data, 'L')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/terrain_snapshot_{timestamp}.png"
        os.makedirs("output", exist_ok=True)
        img.save(filename)
        log.info(f"✅ 地形快照已成功儲存至: {filename}")

    def _create_boundary_fade(self) -> np.ndarray:
        """創建一個邊界淡出遮罩，用於平滑拼接不同類型的地塊。"""
        fade_size = int(self.tile_resolution * 0.2)
        fade_curve = np.linspace(0.0, 1.0, fade_size)
        mask1d = np.ones(self.tile_resolution)
        mask1d[:fade_size], mask1d[-fade_size:] = fade_curve, np.flip(fade_curve)
        return np.minimum(np.ones((self.tile_resolution, self.tile_resolution)) * mask1d, (np.ones((self.tile_resolution, self.tile_resolution)) * mask1d).T)

    # --- 各種地形的生成器函式 ---
    def generate_flat(self): return np.zeros((self.tile_resolution, self.tile_resolution))
    def generate_sine_waves(self):
        """生成由正弦波組成的波浪狀地形，並應用邊界淡出。"""
        x = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        y = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        return 0.08 * (np.sin(X) + np.sin(Y)) * self._create_boundary_fade()
    def generate_steps(self):
        """生成階梯狀地形，並應用邊界淡出。"""
        hfield = np.zeros((self.tile_resolution, self.tile_resolution))
        num_steps, step_height = random.randint(5, 10), 0.05
        step_width = self.tile_resolution // num_steps
        for i in range(num_steps): hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield * self._create_boundary_fade()
    def generate_random_noise(self): return np.random.rand(self.tile_resolution, self.tile_resolution) * 0.1 * self._create_boundary_fade()
    def generate_pyramid(self):
        """生成一個中央高、四周低的正金字塔地形。"""
        max_height = random.uniform(0.3, 0.6)
        x = np.linspace(-1, 1, self.tile_resolution)
        y = np.linspace(-1, 1, self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.maximum(np.abs(X), np.abs(Y))
        return max_height * (1 - dist)
    def generate_stepped_pyramid(self):
        num_steps, max_height = random.randint(12, 16), random.uniform(0.4, 0.8)
        step_height = max_height / num_steps
        x, y = np.linspace(-1, 1, self.tile_resolution), np.linspace(-1, 1, self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.maximum(np.abs(X), np.abs(Y))
        continuous_hfield = max_height * (1.0 - dist)
        hfield_data = np.ceil(continuous_hfield / step_height) * step_height
        hfield_data[dist >= 1.0] = 0.0

        # 返回最終生成的階梯金字塔高度數據
        return hfield_data
