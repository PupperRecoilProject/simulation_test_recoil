# src/simulation/observation_manager.py (4.3.1 新檔案)

import numpy as np
from typing import TYPE_CHECKING, List, Dict
from src.core.logger import log

# 類型檢查
if TYPE_CHECKING:
    from src.core.state import SimulationState

class ObservationManager:
    """
    [v4.3.1 新增] 觀測向量管理器

    這是觀測向量生成的唯一權威。它從中央的 SimulationState 中讀取
    原始感測器數據，並根據指定的「配方」將它們組合成最終的 ONNX 模型輸入。
    """
    def __init__(self, state: "SimulationState"):
        # 儲存對中央狀態的參考，這是所有數據的來源
        self.state = state
        self.config = state.config
        
        # 來自舊 ObservationBuilder 的配方和維度資訊
        self.recipe: List[str] = []
        self.component_dims: Dict[str, int] = {}

        # 定義所有可能的觀測元件及其維度
        self.ALL_OBS_DIMS: Dict[str, int] = {
            'gravity_vector': 3,
            'commands': 3,
            'joint_positions': 12,
            'joint_velocities': 12,
            'last_action': 12,
            'angular_velocity': 3,
            'linear_velocity': 3,
            'accelerometer': 3,
            # 其他可能的元件...
            'z_angular_velocity': 1,
            'foot_contact_states': 4,
            'phase_signal': 1,
        }
        
        # 註冊所有產生器函式
        self._component_generators = self._register_components()
        log.info("✅ 觀測管理器 (ObservationManager) 初始化完成。")

    def set_recipe(self, recipe: List[str]):
        """動態設定當前要使用的觀測配方。"""
        if self.recipe == recipe:
            return # 如果配方未改變，則不執行任何操作
        self.recipe = recipe
        # 根據新配方，更新當前啟用的元件維度字典
        self.component_dims = {k: self.ALL_OBS_DIMS[k] for k in recipe if k in self.ALL_OBS_DIMS}
        # 檢查是否有未知的元件
        for component in self.recipe:
            if component not in self._component_generators:
                log.warning(f"新配方中的元件 '{component}' 不存在，將被忽略。")
    
    def get_observation(self) -> np.ndarray:
        """
        根據當前設定的配方，依序呼叫產生器函式並拼接成最終的觀測向量。
        此函式不接受任何參數，因為所有需要的數據都來自 self.state。
        """
        obs_list = []
        # 遍歷配方中的每一個元件名稱
        for name in self.recipe:
            if name in self._component_generators:
                # 呼叫對應的產生器函式
                obs_list.append(self._component_generators[name]())
        
        # 如果列表為空，返回一個空的浮點數陣列
        if not obs_list:
            return np.array([], dtype=np.float32)
        
        # 將所有元件的向量拼接成一個長向量
        return np.concatenate(obs_list).astype(np.float32)

    # ------------------- 向量計算輔助函式 -------------------
    # 這些函式是從舊的 ObservationBuilder 遷移而來，並進行了重構。
    # 【核心重構】: 所有數據源都從 self.data 改為 self.state.raw_...

    def _register_components(self) -> Dict[str, Callable]:
        """註冊所有已知的觀察元件及其對應的產生器函式。"""
        return {
            'gravity_vector': self._get_gravity_vector,
            'commands': self._get_commands,
            'joint_positions': self._get_joint_positions,
            'last_action': self._get_last_action,
            'angular_velocity': self._get_full_angular_velocity,
            'joint_velocities': self._get_joint_velocities,
            'accelerometer': self._get_accelerometer,
            'linear_velocity': self._get_linear_velocity,
            # 以下為其他可能的元件，邏輯保持不變
            'z_angular_velocity': self._get_z_angular_velocity,
        }

    def _get_torso_inverse_rotation(self) -> np.ndarray:
        """計算軀幹的逆四元數，用於將世界座標系向量轉換為局部座標系。"""
        # [修改] 數據源變更
        torso_quat = self.state.raw_torso_quat
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8:
            return np.array([1., 0., 0., 0.])
        return np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm

    def _rotate_vec_by_quat_inv(self, v: np.ndarray, q_inv: np.ndarray) -> np.ndarray:
        """使用逆四元數旋轉一個向量。"""
        u, s = q_inv[1:], q_inv[0]
        return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)

    def _get_gravity_vector(self) -> np.ndarray:
        """計算局部座標系下的重力向量。"""
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), inv_torso_rot)

    def _get_commands(self) -> np.ndarray:
        """獲取縮放後的使用者指令。"""
        # [修改] 數據源變更
        return self.state.command * np.array(self.config.command_scaling_factors)

    def _get_joint_positions(self) -> np.ndarray:
        """獲取相對於預設站姿的關節角度。"""
        # [修改] 數據源變更
        return self.state.raw_joint_positions - self.state.sim.default_pose

    def _get_joint_velocities(self) -> np.ndarray:
        """獲取關節角速度。"""
        # [修改] 數據源變更
        return self.state.raw_joint_velocities

    def _get_last_action(self) -> np.ndarray:
        """獲取上一幀的 AI 原始輸出。"""
        # [修改] 數據源變更
        return self.state.raw_last_action

    def _get_linear_velocity(self) -> np.ndarray:
        """獲取局部座標系下的軀幹線速度。"""
        inv_torso_rot = self._get_torso_inverse_rotation()
        # [修改] 數據源變更
        return self._rotate_vec_by_quat_inv(self.state.raw_torso_linear_velocity_world, inv_torso_rot)

    def _get_full_angular_velocity(self) -> np.ndarray:
        """獲取局部座標系下的軀幹角速度。"""
        inv_torso_rot = self._get_torso_inverse_rotation()
        # [修改] 數據源變更
        return self._rotate_vec_by_quat_inv(self.state.raw_torso_angular_velocity_world, inv_torso_rot)

    def _get_accelerometer(self) -> np.ndarray:
        """獲取加速度計讀數。"""
        # [修改] 數據源變更
        return self.state.raw_accelerometer

    def _get_z_angular_velocity(self) -> np.ndarray:
        """獲取局部座標系下 Z 軸的角速度（yaw rate）。"""
        local_ang_vel = self._get_full_angular_velocity()
        return np.array([local_ang_vel[2]]) * 0.25 # 乘以縮放因子