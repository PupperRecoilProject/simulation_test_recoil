# observation.py
import numpy as np
import mujoco
import sys
from config import AppConfig

class ObservationBuilder:
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
        return {
            'z_angular_velocity': self._get_z_angular_velocity, 'gravity_vector': self._get_gravity_vector,
            'commands': self._get_commands, 'joint_positions': self._get_joint_positions,
            'last_action': self._get_last_action, 'linear_velocity': self._get_linear_velocity,
            'angular_velocity': self._get_full_angular_velocity, 'joint_velocities': self._get_joint_velocities,
            'foot_contact_states': self._get_foot_contact_states, 'phase_signal': self._get_phase_signal,
        }

    def get_observation(self, command, last_action) -> np.ndarray:
        obs_list = [gen(command=command, last_action=last_action) for gen in [self._component_generators[name] for name in self.recipe]]
        return np.concatenate(obs_list).astype(np.float32)

    def _get_torso_inverse_rotation(self):
        torso_quat = self.data.xquat[self.torso_id]
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8: torso_quat = np.array([1., 0, 0, 0])
        torso_quat /= np.sqrt(np.sum(np.square(torso_quat)))
        return np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / np.sum(np.square(torso_quat))

    def _rotate_vec_by_quat_inv(self, v, q_inv):
        u, s = q_inv[1:], q_inv[0]
        return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)

    def _get_z_angular_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        local_rpy_rate = self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, 3:], inv_torso_rot)
        return np.array([local_rpy_rate[2]]) * 0.25

    def _get_gravity_vector(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), inv_torso_rot)

    def _get_commands(self, command, **kwargs):
        return command * np.array(self.config.command_scaling_factors) 

    def _get_joint_positions(self, **kwargs):
        return self.data.qpos[7:] - self.default_pose

    def _get_joint_velocities(self, **kwargs):
        return self.data.qvel[6:] * 0.05

    def _get_foot_contact_states(self, **kwargs):
        foot_geom_names = ['FR', 'FL', 'RR', 'RL']
        foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geom_names]
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for foot_idx, foot_geom_id in enumerate(foot_geom_ids):
                if foot_geom_id != -1 and (con.geom1 == foot_geom_id or con.geom2 == foot_geom_id):
                    contacts[foot_idx] = 1.0
                    break
        return contacts

    def _get_last_action(self, last_action, **kwargs):
        return last_action

    def _get_linear_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, :3], inv_torso_rot)

    def _get_full_angular_velocity(self, **kwargs):
        inv_torso_rot = self._get_torso_inverse_rotation()
        return self._rotate_vec_by_quat_inv(self.data.cvel[self.torso_id, 3:], inv_torso_rot)
        
    def _get_phase_signal(self, **kwargs):
        return np.array([self.data.time % 1.0], dtype=np.float32)