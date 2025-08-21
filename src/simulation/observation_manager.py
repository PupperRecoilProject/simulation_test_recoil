# src/simulation/observation_manager.py (4.3.1 新檔案)
"""
【v4.4.5 修改】觀測向量管理器 (全量數據供給者版)

【v4.3.1 新增】 觀測向量管理器

這是觀測向量生成的唯一權威。它從中央的 SimulationState 中讀取
原始感測器數據，並根據指定的「配方」將它們組合成最終的 ONNX 模型輸入。

【v4.4.5 架構重構說明】
此版本移除了 recipe 驅動的「按需計算」模式，轉而採用「帶緩存的數據供給」模式。
- 它不再由外部通過 set_recipe() 指揮，而是通過 get_component() 為所有消費者提供統一的數據接口。
- 內部緩存機制確保了在單一控制週期內，每個觀測分量只會被計算一次，兼顧了數據的完整性和計算的高效性。
"""

# [v4.3.1 修正] 導入缺失的模組
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Callable
from src.core.logger import log

# 【v4.3.1 新增】 類型檢查區塊
if TYPE_CHECKING:
    from src.core.state import SimulationState

class ObservationManager:
    """
    【v4.4.5 重構】帶內部緩存的數據供給者。
    【v4.3.1 新增】觀測向量管理器。

    這是觀測向量生成的唯一權威。它從中央的 SimulationState 中讀取
    原始感測器數據，並將它們處理成標準化的觀測分量。

    v4.4.5 架構重構說明:
    此版本引入了內部緩存機制，轉為「帶緩存的數據供給」模式，
    以兼顧數據完整性與計算高效性。
    """

    def __init__(self, state: "SimulationState"):
        """【v4.4.5 修改】初始化函式，增加了內部緩存屬性。"""
        self.state = state
        self.config = state.config
        
        # 【v4.4.5 刪除】不再需要 recipe 和 component_dims 公共屬性
        # self.recipe: List[str] = []
        # self.component_dims: Dict[str, int] = {}

        # 【v4.4.5 新增】本週期已計算觀測數據的內部緩存
        # 這個緩存在每個控制週期的開始被清空。
        self._current_frame_cache: Dict[str, np.ndarray] = {}

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
            'z_angular_velocity': 1,
        }
        
        # 註冊所有產生器函式
        self._component_generators = self._register_components()

        # 【v4.4.6 新增】初始化 state.std_obs 字典
        with self.state.lock:
            if not hasattr(self.state, 'std_obs'):
                self.state.std_obs = {}
            for name, dim in self.ALL_OBS_DIMS.items():
                self.state.std_obs[name] = np.zeros(dim)

        log.info("✅ 觀測管理器 (v4.4.6 全局狀態版) 初始化完成。")
    
    # 【v4.4.6 新增】 全量更新方法
    def update_all_observations(self):
        """每個控制週期計算所有觀測分量，並更新到 state.std_obs。"""
        processed_obs = {}
        for name, generator_func in self._component_generators.items():
            processed_obs[name] = generator_func()
        
        with self.state.lock:
            self.state.std_obs.update(processed_obs)


    # 【v4.4.6 刪除】移除 new_frame 和 get_component，因為不再需要內部緩存
    # def new_frame(self): ...
    # def get_component(self, name: str) -> np.ndarray: ...


    # 【v4.4.5 刪除】set_recipe 和 get_observation 方法，它們的職責已被新的架構所取代。
    # def set_recipe(self, recipe: List[str]): ...
    # def get_observation(self) -> np.ndarray: ...


    # ------------------- 向量計算輔助函式 -------------------
    # 這些函式是從舊的 ObservationBuilder 遷移而來，並進行了重構。
    # 【核心重構】: 所有數據源都從 self.data 改為 self.state.raw_...

    # 【v4.3.1 新增】 _register_components 方法
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

    # 【v4.3.1 修改】 _get_torso_inverse_rotation 方法
    def _get_torso_inverse_rotation(self) -> np.ndarray:
        """計算軀幹的逆四元數，用於將世界座標系向量轉換為局部座標系。"""
        # [修改] 數據源變更
        torso_quat = self.state.raw_torso_quat
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8:
            return np.array([1., 0., 0., 0.])
        return np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm

    # 【v4.3.1 新增】 _rotate_vec_by_quat_inv 方法
    def _rotate_vec_by_quat_inv(self, v: np.ndarray, q_inv: np.ndarray) -> np.ndarray:
        """使用逆四元數旋轉一個向量。"""
        u, s = q_inv[1:], q_inv[0]
        return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)

    def _get_gravity_vector(self) -> np.ndarray:
        """
        【v4.4.3 重構】計算局部座標系下的重力向量（模式感知）。

        - 在硬體模式下，直接信任並返回 Teensy 預計算好的重力向量。
        - 在模擬模式下，通過旋轉世界重力向量來計算。
        """
        if self.state.control_mode == "HARDWARE_MODE":
            # 硬體模式：直接返回 Teensy 提供的數據
            return self.state.raw_gravity_vector
        else:
            # 模擬模式：執行基於四元數的計算
            inv_torso_rot = self._get_torso_inverse_rotation()
            return self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), inv_torso_rot)

    # 【v4.3.1 修改】 _get_commands 方法
    def _get_commands(self) -> np.ndarray:
        """獲取縮放後的使用者指令。"""
        # [修改] 數據源變更
        return self.state.command * np.array(self.config.command_scaling_factors)

    # 【v4.3.1 修改】 _get_joint_positions 方法
    def _get_joint_positions(self) -> np.ndarray:
        """獲取相對於預設站姿的關節角度。"""
        # [修改] 數據源變更
        return self.state.raw_joint_positions - self.state.sim.default_pose

    # 【v4.3.1 修改】 _get_joint_velocities 方法
    def _get_joint_velocities(self) -> np.ndarray:
        """獲取關節角速度。"""
        # [修改] 數據源變更
        return self.state.raw_joint_velocities

    # 【v4.3.1 修改】 _get_last_action 方法
    def _get_last_action(self) -> np.ndarray:
        """獲取上一幀的 AI 原始輸出。"""
        # [修改] 數據源變更
        return self.state.raw_last_action

    def _get_linear_velocity(self) -> np.ndarray:
        """
        【v4.4.3 修改】獲取局部座標系下的軀幹線速度（模式感知）。

        - 在硬體模式下，由於 Teensy 未提供線速度數據，返回零向量。
        - 在模擬模式下，通過旋轉世界線速度向量來計算。
        """
        if self.state.control_mode == "HARDWARE_MODE":
            # 硬體模式：Teensy 未提供此數據，返回零
            return np.zeros(3)
        else:
            # 模擬模式：執行基於四元數的計算
            inv_torso_rot = self._get_torso_inverse_rotation()
            return self._rotate_vec_by_quat_inv(self.state.raw_torso_linear_velocity, inv_torso_rot)

    def _get_full_angular_velocity(self) -> np.ndarray:
        """
        【v4.4.3 重構】獲取局部座標系下的軀幹角速度（模式感知）。

        - 在硬體模式下，直接信任並返回 Teensy 提供的機身坐標系角速度。
        - 在模擬模式下，通過旋轉世界角速度向量來計算。
        """
        if self.state.control_mode == "HARDWARE_MODE":
            # 硬體模式：直接返回 Teensy 提供的數據（數據契約定義其為機身坐標系）
            return self.state.raw_torso_angular_velocity
        else:
            # 模擬模式：執行基於四元數的計算
            inv_torso_rot = self._get_torso_inverse_rotation()
            return self._rotate_vec_by_quat_inv(self.state.raw_torso_angular_velocity, inv_torso_rot)

    # 【v4.3.1 修改】 _get_accelerometer 方法
    def _get_accelerometer(self) -> np.ndarray:
        """獲取加速度計讀數。"""
        # [修改] 數據源變更
        return self.state.raw_accelerometer

    # 【v4.3.1 修改】 _get_z_angular_velocity 方法
    def _get_z_angular_velocity(self) -> np.ndarray:
        """獲取局部座標系下 Z 軸的角速度（yaw rate）。"""
        local_ang_vel = self._get_full_angular_velocity()
        return np.array([local_ang_vel[2]]) * 0.25 # 乘以縮放因子