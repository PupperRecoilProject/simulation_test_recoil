# floating_controller.py
import mujoco
import numpy as np
from config import AppConfig

class FloatingController:
    """
    透過啟用/禁用 weld 約束和設定 mocap body 的位置，
    來將機器人主幹固定在空中。
    """
    def __init__(self, config: AppConfig, model, data):
        """
        初始化懸浮控制器，並獲取必要的 MuJoCo ID 和索引。
        """
        self.config = config.floating_controller
        self.model = model
        self.data = data
        self.is_functional = False
        
        try:
            anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'anchor')
            if anchor_body_id == -1: raise ValueError("在 XML 中找不到名為 'anchor' 的 body。")
            
            self.mocap_index = model.body_mocapid[anchor_body_id]
            if self.mocap_index == -1: raise ValueError("'anchor' body 不是一個 mocap body。")

            self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'torso_anchor_weld')
            if self.weld_id == -1: raise ValueError("在 XML 中找不到名為 'torso_anchor_weld' 的 weld 約束。")

            self.is_functional = True
            print(f"✅ 固定式懸浮控制器初始化完成。Mocap Index: {self.mocap_index}")
        except ValueError as e:
            print(f"❌ 懸浮控制器初始化錯誤: {e}")
            print("     請確保 scene_mjx.xml 檔案已正確定義 'anchor' body 和 'torso_anchor_weld' 約束。懸浮功能將被禁用。")

    def enable(self, current_pos: np.ndarray):
        """啟用懸浮模式。"""
        if not self.is_functional: return
        
        target_pos = np.array([current_pos[0], current_pos[1], self.config.target_height])
        self.data.mocap_pos[self.mocap_index] = target_pos
        self.data.mocap_quat[self.mocap_index] = np.array([1., 0, 0, 0]) # 保持水平姿態
        self.data.eq_active[self.weld_id] = 1
        print("🚀 已啟用固定懸浮模式。")

    def disable(self):
        """禁用懸浮模式。"""
        if not self.is_functional: return
        self.data.eq_active[self.weld_id] = 0
        print("🐾 已禁用固定懸浮模式。")