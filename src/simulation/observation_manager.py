# src/simulation/observation_manager.py (4.3.1 新檔案)

# [v4.3.1 修正] 導入缺失的模組
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Callable
from src.core.logger import log

# 【v4.3.1 新增】 類型檢查區塊
if TYPE_CHECKING:
    from src.core.state import SimulationState

# 【v4.3.1 新增】 ObservationManager 類別
class ObservationManager:
    """
    【v4.3.1 新增】 觀測向量管理器
    【v4.4.7 修改】重構為供應驅動模式。

    這是觀測向量生成的唯一權威。它從中央的 SimulationState 中讀取
    原始感測器數據，並根據指定的「配方」將它們組合成最終的 ONNX 模型輸入。
    在新架構下，它負責在每一幀計算所有標準化觀測值，並填充 state.std_obs。
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
            # 【修改】移除舊的 'commands'，加入新的 'commands_3d' 和 'commands_4d'
            'commands_3d': 3,
            'commands_4d': 4,            
            'joint_positions': 12,
            'joint_velocities': 12,
            'last_action': 12,
            'angular_velocity': 3,
            'linear_velocity': 3,
            'accelerometer': 3,
            'current_pitch': 1,
            'firearm_recoil_warming': 1,
            'z_angular_velocity': 1,
            'foot_contact_states': 4,
            'phase_signal': 1,
        }
        
        # 註冊所有產生器函式
        self._component_generators = self._register_components()
        log.info("✅ 觀測管理器 (ObservationManager) 初始化完成。")

    # 【v4.4.7 刪除】 移除舊的、需求驅動的 API
    # def set_recipe(self, recipe: List[str]):

    
    # 【v4.4.7 刪除】 移除舊的、需求驅動的 API
    # def get_observation(self) -> np.ndarray:

    # 【v4.4.7 新增】 新的核心 API，用於供應驅動的數據流
    def update_all_observations(self) -> None:
        """
        【v4.4.7 新增】
        計算所有已知的標準化觀測向量，並將結果存入 state.std_obs。
        此函式應由主控制迴圈 (SimulationController) 在每一幀的末尾調用。
        """
        # 在一次性的鎖定範圍內，計算所有分量
        with self.state.lock:
            # 遍歷所有已註冊的產生器函式
            for name, generator_func in self._component_generators.items():
                # 執行計算並將結果存入 state.std_obs 字典
                self.state.std_obs[name] = generator_func()


    # ------------------- 向量計算輔助函式 -------------------
    # 這些函式是從舊的 ObservationBuilder 遷移而來，並進行了重構。
    # 【核心重構】: 所有數據源都從 self.data 改為 self.state.raw_...

    # 【v4.3.1 新增】 _register_components 方法
    def _register_components(self) -> Dict[str, Callable]:
        """註冊所有已知的觀察元件及其對應的產生器函式。"""
        return {
            'gravity_vector': self._get_gravity_vector,
            'commands_3d': self._get_commands_3d,
            'commands_4d': self._get_commands_4d,
            'joint_positions': self._get_joint_positions,
            'last_action': self._get_last_action,
            'angular_velocity': self._get_full_angular_velocity,
            'joint_velocities': self._get_joint_velocities,
            'accelerometer': self._get_accelerometer,
            'linear_velocity': self._get_linear_velocity,
            'current_pitch': self._get_current_pitch,
            'firearm_recoil_warming': self._get_firearm_recoil_warming,
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
        【v4.4.7 修改】移除模式感知邏輯。
        【v4.9.0 修改】增加坐標系審核註解。
        【v4.11.0 修改】實現模式感知，優先使用硬體提供的直接測量值。

        計算局部座標系下的重力向量。
        - 在硬體模式下，直接信任並返回 Teensy 預計算好的重力向量。
        - 在模擬模式下，通過旋轉世界重力向量來計算。
        """
        # 【v4.11.0 核心修正】
        # 檢查當前是否處於硬體模式 (state.hardware_is_running)，
        # 並且檢查 state.raw_gravity_vector 是否包含有效數據 (不全為零)。
        # np.any() 函數會檢查陣列中是否存在任何非零元素。
        if self.state.hardware_is_running and np.any(self.state.raw_gravity_vector):
            # 硬體模式：直接返回 Teensy 提供的、經過 IMU 融合演算法處理的高品質數據。
            # 這是最直接、最準確的數據源，避免了使用模擬器陳舊姿態的錯誤。
            return self.state.raw_gravity_vector
        else:
            # 模擬模式：執行原始的、基於模擬器姿態的計算。
            # 這個邏輯在模擬環境中是完全正確的。
            inv_torso_rot = self._get_torso_inverse_rotation()
            return self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), inv_torso_rot)

    # 【v4.3.1 修改】 _get_commands 方法
    def _get_commands_3d(self) -> np.ndarray:
        """【新增】獲取縮放後的3維使用者指令 [vy, vx, wz]。"""
        # 從4維指令中只取前3個元素
        cmd_3d = self.state.command[:3]
        # 只對前3個元素進行縮放
        scaled_cmd = cmd_3d * np.array(self.config.command_scaling_factors[:3])
        return scaled_cmd

    def _get_commands_4d(self) -> np.ndarray:
        """【新增】獲取縮放後的4維使用者指令 [vy, vx, wz, pitch]。"""
        # 直接使用完整的4維指令
        scaled_cmd = self.state.command * np.array(self.config.command_scaling_factors)
        return scaled_cmd
    
    def _get_joint_positions(self) -> np.ndarray:
        """
        【v4.11.2 修改】數據源從 sim.default_pose 變更為 config.default_pose。
        
        獲取相對於預設站姿的關節角度。
        """
        # 【v4.11.2 核心修正】
        # 數據源從 self.state.sim.default_pose 更改為 self.config.default_pose。
        # 這使得此函式的運作完全獨立於模擬物件 (sim)，
        # 徹底解耦了 AI 觀測邏輯與模擬環境，增強了模組的獨立性和
        # 對無頭 (--no-sim) 模式的支援。
        return self.state.raw_joint_positions - self.config.default_pose

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
        【v4.4.7 修改】移除模式感知邏輯。

        - 在硬體模式下，由於 Teensy 未提供線速度數據，返回零向量。
        - 在模擬模式下，通過旋轉世界線速度向量來計算。

        
        【v4.9.0 修改】增加坐標系審核註解。

        獲取局部座標系下的軀幹線速度。

        - **輸入**: `state.raw_torso_linear_velocity`
            - 模擬模式: 世界座標系 (World Frame)。
            - 硬體模式: 恆為零，因為 Teensy 不提供此數據。
        - **輸出**: 機體局部座標系 (Body Frame) 下的線速度。
        - **TODO**: 在 v5.0.0+ 的狀態估算器 (State Estimator) 中，
                   需要實現從 IMU 數據估算線速度的功能。
        """
        # 【v4.4.7 修改】 移除模式判斷。
        inv_torso_rot = self._get_torso_inverse_rotation()
        # 注意：在硬體模式下，raw_torso_linear_velocity 預設為零，所以這個計算依然是正確的。
        return self._rotate_vec_by_quat_inv(self.state.raw_torso_linear_velocity, inv_torso_rot)

    def _get_full_angular_velocity(self) -> np.ndarray:
        """
        【v4.4.3 重構】獲取局部座標系下的軀幹角速度（模式感知）。
        【v4.A.B 修改】移除模式感知邏輯。
        【v4.9.0 修改】增加坐標系審核註解。
        【v4.11.0 修改】實現模式感知，避免對硬體數據進行冗餘計算。

        獲取局部座標系下的軀幹角速度。
        - 在硬體模式下，直接信任並返回 Teensy 提供的機身坐標系角速度。
        - 在模擬模式下，通過旋轉世界角速度向量來計算。
        """
        # 【v4.11.0 核心修正】
        # 檢查是否處於硬體運行模式。
        if self.state.hardware_is_running:
            # 硬體模式：直接返回 Teensy 提供的局部座標系角速度。
            # Teensy 的 IMU 數據本身就是相對於機身的，無需任何轉換。
            # 這修正了之前對正確數據進行錯誤二次旋轉的問題。
            return self.state.raw_torso_angular_velocity
        else:
            # 模擬模式：執行原始的坐標系旋轉計算。
            inv_torso_rot = self._get_torso_inverse_rotation()
            return self._rotate_vec_by_quat_inv(self.state.raw_torso_angular_velocity, inv_torso_rot)


    # 【v4.3.1 修改】 _get_accelerometer 方法
    def _get_accelerometer(self) -> np.ndarray:
        """
        【v4.9.0 修改】增加坐標系審核註解。
        
        獲取加速度計讀數。

        - **輸入**: `state.raw_accelerometer`
            - 模擬模式: 機體局部座標系 (Body Frame)。
            - 硬體模式: 機體局部座標系 (Body Frame)。
        - **輸出**: 機體局部座標系 (Body Frame) 下的加速度。
        - **TODO**: 需與 Teensy IMU 的實際輸出坐標系 (Y-fwd, X-right) 進行校對，
                   確認是否需要進行軸交換或符號翻轉。
        """
        # [修改] 數據源變更
        return self.state.raw_accelerometer

    # 【v4.3.1 修改】 _get_z_angular_velocity 方法
    def _get_z_angular_velocity(self) -> np.ndarray:
        """獲取局部座標系下 Z 軸的角速度（yaw rate）。"""
        local_ang_vel = self._get_full_angular_velocity()
        return np.array([local_ang_vel[2]]) * 0.25 # 乘以縮放因子
    
    def _quat_to_rot_matrix(self, q: np.ndarray) -> np.ndarray:
        """輔助函式：將四元數轉換為3x3旋轉矩陣。"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ]) # 實現四元數到旋轉矩陣的轉換

    def _get_current_pitch(self) -> np.ndarray:
        """
        【v4.11.0 修改】實現模式感知，優先使用硬體提供的直接測量值。

        計算當前的俯仰角 (pitch)。
        - 在硬體模式下，直接信任並返回 Teensy 提供的俯仰角。
        - 在模擬模式下，通過姿態四元數進行計算。
        """
        # 【v4.11.0 核心修正】
        # 檢查是否處於硬體模式，並確認 Teensy 提供了有效的俯仰角數據 (不為零)。
        if self.state.hardware_is_running and self.state.raw_pitch_rad != 0.0:
            # 硬體模式：直接返回 Teensy 提供的、經過 IMU 融合演算法計算出的俯仰角。
            # 我們將其封裝在一個單元素的 numpy 陣列中，以符合所有觀測元件的標準格式。
            return np.array([self.state.raw_pitch_rad], dtype=np.float32)
        else:
            # 模擬模式：執行原始的、基於四元數的計算。
            q = self.state.raw_torso_quat
            rot_matrix = self._quat_to_rot_matrix(q)
            pitch = np.arcsin(-rot_matrix[2, 0])
            return np.array([pitch], dtype=np.float32)

    def _get_firearm_recoil_warming(self) -> np.ndarray:
        """【新增】獲取後座力預警訊號。"""
        # 直接從 state 中讀取預警旗標，並轉換為浮點數
        warning_signal = 1.0 if self.state.recoil_warning_active else 0.0
        return np.array([warning_signal], dtype=np.float32) # 返回包含預警信號的單元素陣列