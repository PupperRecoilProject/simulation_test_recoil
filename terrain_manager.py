# terrain_manager.py
import mujoco
import numpy as np
import random
from typing import Dict, Optional, Callable, Tuple
from datetime import datetime
from PIL import Image

# 為了型別提示，避免循環匯入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state import SimulationState

class TerrainTile:
    """代表地形網格中的一個地塊。"""
    def __init__(self, grid_x: int, grid_y: int, terrain_type: str = "Flat"):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.terrain_type = terrain_type

class TerrainManager:
    """
    【雙模式版】管理無限地形和單一固定地形。
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

        self.tile_resolution = 101
        self.grid_size = 9
        self.tile_world_size = 5.0
        
        self.hfield_nrow = model.hfield_nrow[self.hfield_id]
        self.hfield_ncol = model.hfield_ncol[self.hfield_id]
        self.hfield_size = model.hfield_size[self.hfield_id]
        self.hfield_adr = model.hfield_adr[self.hfield_id]

        expected_hfield_dim = (self.tile_resolution - 1) * self.grid_size + 1
        if self.hfield_nrow != expected_hfield_dim or self.hfield_ncol != expected_hfield_dim:
            print(f"❌ 錯誤: XML hfield 解析度 ({self.hfield_nrow}x{self.hfield_ncol}) 與 TerrainManager 設定不符。")
            print(f"     預期解析度應為: {expected_hfield_dim}x{expected_hfield_dim} (基於 {self.grid_size}x{self.grid_size} 網格)")
            self.is_functional = False
            return
            
        self.world_center_x = 0
        self.world_center_y = 0
        self.terrain_cache: Dict[Tuple[int, int], TerrainTile] = {}
        self.full_hfield_data = np.zeros((self.hfield_nrow, self.hfield_ncol))

        self.terrain_generators: Dict[str, Callable] = {
            "Flat": self.generate_flat,
            "Sine Waves": self.generate_sine_waves,
            "Steps": self.generate_steps,
            "Random Noise": self.generate_random_noise,
            "Pyramid": self.generate_pyramid,
            "Stepped Pyramid": self.generate_stepped_pyramid,
        }
        
        # 【核心修正】將變數名稱從 single_terrain_names 改回 terrain_types，以解決 AttributeError
        self.terrain_types = list(self.terrain_generators.keys()) 
        self.single_terrain_names = list(self.terrain_generators.keys()) # 同時保留 single_terrain_names 以供 SINGLE 模式使用
        
        self.is_functional = True
        
        self.initial_generate()
        print(f"✅ 雙模式地形管理器初始化完成 (使用 {self.grid_size}x{self.grid_size} 網格)。")

    def update(self, robot_pos: np.ndarray, current_mode: str):
        """根據地形模式決定是否更新。"""
        if not self.is_functional or current_mode != "INFINITE":
            return

        robot_grid_x = int(round(robot_pos[0] / self.tile_world_size))
        robot_grid_y = int(round(robot_pos[1] / self.tile_world_size))
        dx = robot_grid_x - self.world_center_x
        dy = robot_grid_y - self.world_center_y
        trigger_radius = (self.grid_size // 2) - 1

        if abs(dx) > trigger_radius or abs(dy) > trigger_radius:
            shift_x = np.sign(dx).astype(int) if abs(dx) > trigger_radius else 0
            shift_y = np.sign(dy).astype(int) if abs(dy) > trigger_radius else 0
            print(f"🔄 機器人接近網格邊緣，向 ({shift_x}, {shift_y}) 方向滑動地形...")
            self.shift_grid_center(shift_x, shift_y)

    def cycle_terrain_mode(self, state: 'SimulationState'):
        """V鍵的功能，在無限模式和各種單一地形之間循環。"""
        if not self.is_functional: return
        
        if state.terrain_mode == "INFINITE":
            state.terrain_mode = "SINGLE"
            state.single_terrain_index = 0
            terrain_name = self.single_terrain_names[state.single_terrain_index]
            print(f"🏞️ 切換到單一地形模式: {terrain_name}")
            self.set_single_terrain(terrain_name)
        else:
            state.single_terrain_index += 1
            if state.single_terrain_index < len(self.single_terrain_names):
                terrain_name = self.single_terrain_names[state.single_terrain_index]
                print(f"🏞️ 切換到下一個單一地形: {terrain_name}")
                self.set_single_terrain(terrain_name)
            else:
                state.terrain_mode = "INFINITE"
                state.single_terrain_index = 0
                print(f"🏞️ 返回無限地形模式")
                self.regenerate_terrain_and_adjust_robot(self.data.qpos)

    def set_single_terrain(self, terrain_name: str):
        """用指定的單一地形類型填滿整個網格。"""
        if terrain_name not in self.terrain_generators:
            print(f"⚠️ 警告: 找不到名為 '{terrain_name}' 的地形生成器。")
            return
            
        generator = self.terrain_generators[terrain_name]
        single_tile_data = generator()

        tile_res_m1 = self.tile_resolution - 1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_row = i * tile_res_m1
                start_col = j * tile_res_m1
                end_row = start_row + self.tile_resolution
                end_col = start_col + self.tile_resolution
                self.full_hfield_data[start_row:end_row, start_col:end_col] = np.maximum(
                    self.full_hfield_data[start_row:end_row, start_col:end_col],
                    single_tile_data
                )
        
        self.model.hfield_data[self.hfield_adr:self.hfield_adr + self.hfield_nrow * self.hfield_ncol] = self.full_hfield_data.flatten()
        self.needs_scene_update = True
        print(f"✅ 已生成完整的 '{terrain_name}' 地形。")

    def regenerate_terrain_and_adjust_robot(self, robot_qpos, robot_height_offset=0.3):
        """Y鍵功能：只在無限模式下重新生成地形。"""
        if not self.is_functional: return
        
        print("🔄 (Y Key) 正在強制重新生成所有地形...")
        self.terrain_cache.clear()
        self.world_center_x = int(round(robot_qpos[0] / self.tile_world_size))
        self.world_center_y = int(round(robot_qpos[1] / self.tile_world_size))

        self.initial_generate()
        
        robot_x = robot_qpos[0]
        robot_y = robot_qpos[1]
        new_ground_z = self.get_height_at(robot_x, robot_y)
        self.data.qpos[2] = new_ground_z + robot_height_offset
        print(f"    機器人高度已調整以適應新地形：Z = {self.data.qpos[2]:.2f}m")
        mujoco.mj_forward(self.model, self.data)

    def get_current_terrain_name(self, state: 'SimulationState') -> str:
        """根據模式顯示不同的地形名稱。"""
        if not self.is_functional: return "N/A (hfield missing)"
        
        if state.terrain_mode == "INFINITE":
            center_tile = self.terrain_cache.get((self.world_center_x, self.world_center_y))
            if center_tile:
                return f"INFINITE (Center: {center_tile.terrain_type})"
            return "INFINITE (Unknown)"
        else:
            terrain_name = self.single_terrain_names[state.single_terrain_index]
            return f"SINGLE ({terrain_name})"

    def shift_grid_center(self, dx: int, dy: int):
        self.world_center_x += dx
        self.world_center_y += dy
        self.update_hfield()

    def get_or_generate_tile(self, grid_x: int, grid_y: int) -> TerrainTile:
        if (grid_x, grid_y) in self.terrain_cache: return self.terrain_cache[(grid_x, grid_y)]
        
        neighbor_types = set()
        for offset_x, offset_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_tile = self.terrain_cache.get((grid_x + offset_x, grid_y + offset_y))
            if neighbor_tile: neighbor_types.add(neighbor_tile.terrain_type)
            
        # 這裡現在使用 self.terrain_types，就不會報錯了
        available_types = [t for t in self.terrain_types if t not in neighbor_types]
        if not available_types: available_types = self.terrain_types
            
        chosen_type = random.choice(available_types)
        new_tile = TerrainTile(grid_x, grid_y, chosen_type)
        self.terrain_cache[(grid_x, grid_y)] = new_tile
        return new_tile

    def update_hfield(self):
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
        self.model.hfield_data[self.hfield_adr:self.hfield_adr + self.hfield_nrow * self.hfield_ncol] = self.full_hfield_data.flatten()
        self.needs_scene_update = True
        print("✅ 完整高度場已更新。")

    def initial_generate(self):
        print("🏞️ 正在生成初始地形...")
        self.update_hfield()

    def get_height_at(self, world_x: float, world_y: float) -> float:
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
        if not self.is_functional:
            print("⚠️ 警告: 地形功能未啟用，無法儲存PNG。")
            return
        print("💾 正在儲存當前地形為PNG檔案...")
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
        print(f"✅ 地形快照已成功儲存至: {filename}")

    def _create_boundary_fade(self) -> np.ndarray:
        fade_margin_ratio = 0.2
        fade_size = int(self.tile_resolution * fade_margin_ratio)
        fade_curve = np.linspace(0.0, 1.0, fade_size)
        mask1d = np.ones(self.tile_resolution)
        mask1d[:fade_size] = fade_curve
        mask1d[-fade_size:] = np.flip(fade_curve)
        mask2d = np.minimum(np.ones((self.tile_resolution, self.tile_resolution)) * mask1d, (np.ones((self.tile_resolution, self.tile_resolution)) * mask1d).T)
        return mask2d

    def generate_flat(self): return np.zeros((self.tile_resolution, self.tile_resolution))
    def generate_sine_waves(self):
        x = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        y = np.linspace(0, 2 * np.pi * random.uniform(2, 4), self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        height_field = 0.08 * (np.sin(X) + np.sin(Y))
        return height_field * self._create_boundary_fade()
    def generate_steps(self):
        hfield = np.zeros((self.tile_resolution, self.tile_resolution))
        step_height = 0.05
        num_steps = random.randint(5, 10)
        step_width = self.tile_resolution // num_steps
        for i in range(num_steps): hfield[i*step_width:(i+1)*step_width, :] = i * step_height
        return hfield * self._create_boundary_fade()
    def generate_random_noise(self):
        noise = np.random.rand(self.tile_resolution, self.tile_resolution) * 0.1
        return noise * self._create_boundary_fade()
    def generate_pyramid(self):
        max_height = random.uniform(0.3, 0.6)
        x = np.linspace(-1, 1, self.tile_resolution)
        y = np.linspace(-1, 1, self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.maximum(np.abs(X), np.abs(Y))
        hfield_data = max_height * (1 - dist)
        return hfield_data
    def generate_stepped_pyramid(self):
        num_steps = random.randint(4, 8)
        max_height = random.uniform(0.4, 0.8)
        step_height = max_height / num_steps
        x = np.linspace(-1, 1, self.tile_resolution)
        y = np.linspace(-1, 1, self.tile_resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.maximum(np.abs(X), np.abs(Y))
        continuous_hfield = max_height * (1.0 - dist)
        height_in_steps = continuous_hfield / step_height
        quantized_steps = np.ceil(height_in_steps)
        hfield_data = quantized_steps * step_height
        hfield_data[dist >= 1.0] = 0.0
        return hfield_data