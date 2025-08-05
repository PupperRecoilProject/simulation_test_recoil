# floating_controller.py
import mujoco
import numpy as np
from utils.config import AppConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.terrain_manager import TerrainManager

class FloatingController:
    """
    透過啟用/禁用 weld 約束和設定 mocap body 的位置，
    來將機器人主幹固定在空中。
    【新版】現在會考慮地形高度。
    """
    def __init__(self, config: AppConfig, model, data, terrain_manager: 'TerrainManager'):
        """
        初始化懸浮控制器，並獲取必要的 MuJoCo ID 和索引。
        """
        self.config = config.floating_controller # 獲取懸浮控制器的專用設定
        self.model = model # 儲存MuJoCo模型
        self.data = data # 儲存MuJoCo數據
        self.terrain_manager = terrain_manager # 【新增】儲存地形管理器的參考
        self.is_functional = False # 標記控制器是否初始化成功
        
        try:
            # 獲取錨點body的ID
            anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'anchor')
            if anchor_body_id == -1: raise ValueError("在 XML 中找不到名為 'anchor' 的 body。")
            
            # 根據body ID獲取mocap索引
            self.mocap_index = model.body_mocapid[anchor_body_id]
            if self.mocap_index == -1: raise ValueError("'anchor' body 不是一個 mocap body。")

            # 獲取焊接約束的ID
            self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'torso_anchor_weld')
            if self.weld_id == -1: raise ValueError("在 XML 中找不到名為 'torso_anchor_weld' 的 weld 約束。")

            self.is_functional = True
            print(f"✅ 固定式懸浮控制器初始化完成。Mocap Index: {self.mocap_index}")
        except ValueError as e:
            print(f"❌ 懸浮控制器初始化錯誤: {e}")
            print("     請確保 scene_mjx.xml 檔案已正確定義 'anchor' body 和 'torso_anchor_weld' 約束。懸浮功能將被禁用。")

    def enable(self, current_pos: np.ndarray):
        """啟用懸浮模式。【修改】目標高度將是相對於地形的高度。"""
        if not self.is_functional: return # 如果控制器未成功初始化，則直接返回
        
        # 【核心修改】計算目標高度
        # 1. 查詢機器人當前 XY 位置下方的地形高度
        ground_z = self.terrain_manager.get_height_at(current_pos[0], current_pos[1])
        # 2. 計算最終的目標世界Z座標
        target_z = ground_z + self.config.target_height
        
        # 組合最終的目標世界座標
        target_pos = np.array([current_pos[0], current_pos[1], target_z])
        
        # 設定mocap body的位置和姿態
        self.data.mocap_pos[self.mocap_index] = target_pos
        self.data.mocap_quat[self.mocap_index] = np.array([1., 0, 0, 0]) # 保持水平姿態
        # 啟用焊接約束，將機器人"鎖"在mocap body上
        self.data.eq_active[self.weld_id] = 1
        print(f"🚀 已啟用相對高度懸浮模式 (地形高度: {ground_z:.2f}m, 目標世界Z: {target_z:.2f}m)。")

    def disable(self):
        """禁用懸浮模式。"""
        if not self.is_functional: return # 如果控制器未成功初始化，則直接返回
        self.data.eq_active[self.weld_id] = 0 # 禁用焊接約束
        print("🐾 已禁用固定懸浮模式。")
