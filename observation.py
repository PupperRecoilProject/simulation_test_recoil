# 內容與原始程式碼的 ObservationBuilder 類別幾乎完全相同
# 只需要做微小的調整，例如 import 的路徑
import numpy as np
import mujoco
import sys
from config import AppConfig

class ObservationBuilder:
    """根據配方從 MuJoCo 環境中提取並組合觀察向量。"""
    def __init__(self, recipe: list, data, model, torso_id, default_pose, config: AppConfig):
        self.recipe = recipe
        self.data = data
        self.model = model
        self.torso_id = torso_id
        self.default_pose = default_pose
        self.config = config
        self._component_generators = self._register_components()

        for component in self.recipe:
            if component not in self._component_generators:
                sys.exit(f"❌ 致命錯誤: 觀察配方中的元件 '{component}' 沒有對應的產生器函式。")

    def _register_components(self):
        """註冊所有已知的觀察元件及其對應的產生器函式。"""
        return {
            'z_angular_velocity': self._get_z_angular_velocity,
            'gravity_vector': self._get_gravity_vector,
            'commands': self._get_commands,
            'joint_positions': self._get_joint_positions,
            'last_action': self._get_last_action,
            'linear_velocity': self._get_linear_velocity,
            'angular_velocity': self._get_full_angular_velocity,
            'joint_velocities': self._get_joint_velocities,
            'foot_contact_states': self._get_foot_contact_states,
            'phase_signal': self._get_phase_signal,
        }

    def get_observation(self, command, last_action) -> np.ndarray:
        """根據配方列表，依序呼叫產生器函式並拼接成最終的觀察向量。"""
        obs_list = []
        for name in self.recipe:
            component_data = self._component_generators[name](command=command, last_action=last_action)
            obs_list.append(component_data)
        return np.concatenate(obs_list).astype(np.float32)

    # --- 這裡貼上原始程式碼中 ObservationBuilder 的所有 _get_... 輔助函式 ---
    # 例如 _get_torso_inverse_rotation, _rotate_vec_by_quat_inv,
    # _get_z_angular_velocity, _get_gravity_vector 等等...
    # (此處省略以節省篇幅，請從原檔複製 붙여넣기)
    # --- 輔助函式：用於多個觀察元件的通用計算 ---
    def _get_torso_inverse_rotation(self):
        """計算軀幹姿態的逆四元數，用於將世界座標系向量轉換到局部座標系。"""
        torso_quat = self.data.xquat[self.torso_id]
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8: torso_quat = np.array([1., 0, 0, 0])
        torso_quat /= np.sqrt(np.sum(np.square(torso_quat)))
        return np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / np.sum(np.square(torso_quat))

    def _rotate_vec_by_quat_inv(self, v, q_inv):
        """使用逆四元數將向量 'v' 從世界座標系旋轉到由 'q_inv' 表示的局部座標系。"""
        u, s = q_inv[1:], q_inv[0]
        return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)

    # --- 觀察元件產生器 ---
    def _get_z_angular_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        local_rpy_rate = self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, 3:], inv_torso_rot)
        return np.array([local_rpy_rate[2]]) * 0.25

    def _get_gravity_vector(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), inv_torso_rot)

    def _get_commands(self, command, **kwargs):
        """獲取縮放後的使用者指令 (vy, vx, wz)。維度: 3"""
        # 現在從 config 中獲取縮放因子
        return command * np.array(self.config.command_scaling_factors) 


    def _get_joint_positions(self, **kwargs):
        return self.data.qpos[7:] - self.default_pose

    def _get_last_action(self, last_action, **kwargs):
        return last_action

    def _get_linear_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, :3], inv_torso_rot)

    def _get_full_angular_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, 3:], inv_torso_rot)

    def _get_joint_velocities(self, **kwargs):
        return self.data.qvel[6:] * 0.05

    def _get_foot_contact_states(self, **kwargs):
        foot_geoms = ['FR_foot_geom', 'FL_foot_geom', 'RR_foot_geom', 'RL_foot_geom']
        foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geoms]
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for foot_idx, foot_geom_id in enumerate(foot_geom_ids):
                if foot_geom_id != -1 and (con.geom1 == foot_geom_id or con.geom2 == foot_geom_id):
                    contacts[foot_idx] = 1.0; break
        return contacts

    def _get_phase_signal(self, **kwargs):
        return np.array([self.data.time % 1.0], dtype=np.float32)