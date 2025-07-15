# rendering.py
import mujoco
import numpy as np
from state import SimulationState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

def _get_local_ang_vel_for_display(data, torso_id: int) -> np.ndarray:
    """獲取軀幹在局部座標系的完整角速度，僅用於除錯顯示。"""
    torso_quat = data.xquat[torso_id]
    norm = np.sum(np.square(torso_quat))
    if norm < 1e-8: return np.zeros(3)
    torso_quat /= np.sqrt(norm)
    q_inv = np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm
    u, s = q_inv[1:], q_inv[0]
    world_ang_vel = data.cvel[torso_id, 3:]
    return 2 * np.dot(u, world_ang_vel) * u + (s*s - np.dot(u, u)) * world_ang_vel + 2*s*np.cross(u, world_ang_vel)

class DebugOverlay:
    """在 MuJoCo 視窗上覆蓋顯示除錯資訊。"""
    def __init__(self, recipe: list, recipe_dims: dict):
        self.recipe = recipe
        self.component_dims = recipe_dims
        self.display_pages_content = [
            # 根據您的 48-dim recipe 調整
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands'],
            ['joint_positions', 'joint_velocities', 'last_action'],
            # 您也可以加入其他的，例如 'foot_contact_states', 'phase_signal'
        ]
        # 根據 display_pages_content 的長度動態設定總頁數
        state_for_pages = SimulationState
        state_for_pages.num_display_pages = len(self.display_pages_content)


    def render(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """將所有資訊格式化成文字並渲染到指定的視口上。"""
        # --- 統一樣式輔助函式 ---
        def format_item(label: str, value, label_width=16):
            return f"{label:<{label_width}}{value}"
        
        def format_vec(label: str, vec, precision=3, label_width=24):
            if vec is None or vec.size == 0: return f"{label:<{label_width}}None"
            vec_str = np.array2string(vec, precision=precision, floatmode='fixed', suppress_small=True, threshold=100)
            return f"{label:<{label_width}}{vec_str}"

        # --- TOP RIGHT: Controls Help Text ---
        help_text = (
            "--- CONTROLS ---\n\n"
            "[Universal]\n"
            "  F: Toggle Walk/Float Mode\n"
            "  ESC: Exit  |  R: Reset\n"
            "  M: Toggle Input Mode\n"
            "  TAB: Toggle Info Page\n"
            "  C: Clear Cmd (Keyboard)\n\n"
            "[Keyboard Mode]\n"
            "  WASD/QE: Move/Turn\n"
            "  I/K: Kp +/- | L/J: Kd +/-\n"
            "  Y/H: ActScl +/- | P/;: Bias +/-\n\n"
            "[Gamepad Mode]\n"
            "  L-Stick: Move | R-Stick: Turn\n"
            "  D-Pad U/D: Kp | D-Pad R/L: Kd\n"
            "  LB/RB: ActScl | Y/A: Bias\n"
            "  Select/View: Reset"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)

        # --- TOP LEFT: General Info & Tuning Params ---
        p = state.tuning_params
        top_left_info_text = (
            f"{format_item('Control Mode:', state.control_mode)}\n"
            f"{format_item('Input Mode:', state.input_mode)}\n"
            f"{format_item('UI Mode:', state.sim_mode_text)}\n"
            f"{format_item('Time:', f'{sim.data.time:.2f} s')}\n\n"
            f"--- Tuning Params ---\n"
            f"{format_item('Kp:', f'{p.kp:.1f}')}\n"
            f"{format_item('Kd:', f'{p.kd:.2f}')}\n"
            f"{format_item('Action Scale:', f'{p.action_scale:.3f}')}\n"
            f"{format_item('Bias:', f'{p.bias:.1f}')}\n\n"
            f"--- Command ---\n"
            f"{format_vec('User Command:', state.command, label_width=16)}\n"
        )
        
        if state.control_mode == "FLOATING":
            current_height = sim.data.qpos[2]
            target_height = sim.config.floating_controller.target_height
            float_text = (
                f"\n--- Floating Info ---\n"
                f"{format_item('Target Height:', f'{target_height:.3f} m')}\n"
                f"{format_item('Current Height:', f'{current_height:.3f} m')}\n"
            )
            top_left_info_text += float_text
        
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, top_left_info_text, None, context)

        # --- BOTTOM LEFT: ONNX Inputs ---
        bottom_left_format_vec = lambda name, vec, p=2: format_vec(f"{name} [{vec.size}d]:", vec, precision=p)
        bottom_left_text = f"--- ONNX INPUTS (Page {state.display_page + 1}/{state.num_display_pages}) ---\n"
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0 and state.display_page < len(self.display_pages_content):
            current_page_components = self.display_pages_content[state.display_page]
            base_obs_dim = sum(self.component_dims.values())
            history_len = len(onnx_input_vec) // base_obs_dim if base_obs_dim > 0 else 1 
            current_full_obs_idx = 0
            for comp_name_in_recipe in self.recipe:
                dim = self.component_dims.get(comp_name_in_recipe, 0)
                if dim > 0:
                    if comp_name_in_recipe in current_page_components:
                        start_idx, end_idx = current_full_obs_idx, current_full_obs_idx + dim
                        value_slice = onnx_input_vec[start_idx:end_idx]
                        bottom_left_text += bottom_left_format_vec(comp_name_in_recipe, value_slice) + "\n"
                    current_full_obs_idx += dim
            if history_len > 1:
                bottom_left_text += f"  (From latest frame of {history_len}-frame history)\n"
            bottom_left_text += f"Total Dim: {len(onnx_input_vec)} (Base {base_obs_dim} x Hist {history_len})\n"
        else:
            bottom_left_text += "  (ONNX input not available)\n"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, bottom_left_text, None, context)

        # --- BOTTOM RIGHT: ONNX Outputs & Robot State ---
        torso_lin_vel = sim.data.cvel[sim.torso_id, :3]
        torso_ang_vel_local = _get_local_ang_vel_for_display(sim.data, sim.torso_id)
        bottom_right_detail_text = (
            f"--- ONNX OUTPUTS & CONTROL ---\n"
            f"{format_vec('ONNX Raw Action:', state.latest_action_raw)}\n"
            f"{format_vec('Final Motor Ctrl:', state.latest_final_ctrl)}\n"
            f"--- Robot State ---\n"
            f"{format_vec('Torso Z-Pos (m):', np.array([sim.data.qpos[2]]))}\n"
            f"{format_vec('Torso Lin Vel (World):', torso_lin_vel, precision=2)}\n"
            f"{format_vec('Torso Ang Vel (Local):', torso_ang_vel_local, precision=2)}"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, viewport, bottom_right_detail_text, None, context)