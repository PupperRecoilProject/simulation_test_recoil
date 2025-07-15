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
            # 獲取 anchor body 的全局 ID
            anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'anchor')
            if anchor_body_id == -1:
                raise ValueError("在 XML 中找不到名為 'anchor' 的 body。")

            # === 關鍵修改：獲取 mocap 索引 ===
            # model.body_mocapid 是一個陣列，其索引是 body ID，值是 mocap 索引
            self.mocap_index = model.body_mocapid[anchor_body_id]
            if self.mocap_index == -1:
                raise ValueError("'anchor' body 不是一個 mocap body。")

            # 獲取 weld 約束的 ID (這個不變)
            self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'torso_anchor_weld')
            if self.weld_id == -1:
                raise ValueError("在 XML 中找不到名為 'torso_anchor_weld' 的 weld 約束。")

            self.is_functional = True
            print(f"✅ 固定式懸浮控制器初始化完成。Mocap Index: {self.mocap_index}")

        except ValueError as e:
            print(f"❌ 錯誤: {e}")
            print("     請確保 scene.xml 檔案已正確修改。懸浮功能將被禁用。")

    def toggle_floating_mode(self, current_pos, current_quat):
        """
        切換懸浮模式的啟用狀態。
        """
        if not self.is_functional:
            return False

        is_active = self.data.eq_active[self.weld_id]

        if not is_active:
            # --- 啟用懸浮 ---
            target_pos = np.array([current_pos[0], current_pos[1], self.config.target_height])
            
            # === 關鍵修改：使用正確的 mocap_index ===
            self.data.mocap_pos[self.mocap_index] = target_pos
            self.data.mocap_quat[self.mocap_index] = [1, 0, 0, 0]
            
            self.data.eq_active[self.weld_id] = 1
            print("🚀 已啟用固定懸浮模式。")
            return True
        else:
            # --- 禁用懸浮 ---
            self.data.eq_active[self.weld_id] = 0
            print("🐾 已禁用固定懸浮模式。")
            return False