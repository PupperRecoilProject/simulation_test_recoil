import mujoco
import numpy as np
from state import SimulationState

# 輔助函式：僅用於顯示目的
def _get_local_ang_vel_for_display(data, torso_id):
    """獲取軀幹在局部座標系的完整角速度，用於除錯顯示。"""
    torso_quat = data.xquat[torso_id]
    norm = np.sum(np.square(torso_quat))
    if norm < 1e-8:
        return np.zeros(3)
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

    def render(self, viewport, context, state: SimulationState, sim):
        """
        將所有資訊格式化成文字並渲染到指定的視口上。
        
        Args:
            viewport: MuJoCo 視口物件。
            context: MuJoCo 渲染上下文。
            state (SimulationState): 當前的應用程式狀態。
            sim: Simulation 實例，用於獲取 MuJoCo data。
        """
        # --- 右上角的操作說明 ---
        help_text = ("--- CONTROLS ---\n\n"
                     "[Keyboard]\n"
                     "  WASD/QE: Move/Turn\n"
                     "  R: Reset | C: Clear Cmd\n"
                     "  I/K: Kp +/- | L/J: Kd +/-\n"
                     "  Y/H: ActScl +/- | P/;: Bias +/-\n"
                     "  ESC: Exit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                           viewport, help_text, None, context)

        # --- 左上角的除錯資訊 ---
        def format_vec(name, vec, precision=3):
            if vec is None: return f"{name:<22}: None"
            return f"{name:<22}: " + np.array2string(vec, precision=precision, 
                                                    floatmode='fixed', suppress_small=True, threshold=100)

        p = state.tuning_params
        info_text = (
            f"Mode: {state.mode_text} (Time: {sim.data.time:.2f} s)\n"
            f"--- Tuning Params ---\n"
            f"Kp: {p.kp:.1f} | Kd: {p.kd:.2f} | ActScl: {p.action_scale:.3f} | Bias: {p.bias:.1f}\n"
            f"--- Command ---\n"
            f"{format_vec('User Command', state.command)}\n"
        )
        
        # --- ONNX 輸入分解 ---
        onnx_input_text = "--- ONNX INPUTS (Breakdown) ---\n"
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0:
            base_obs_dim = sum(self.component_dims.values())
            history_len = len(onnx_input_vec) // base_obs_dim if base_obs_dim > 0 else 1
            
            current_idx = 0
            for comp_name in self.recipe:
                dim = self.component_dims.get(comp_name, 0)
                if dim > 0:
                    val_slice = onnx_input_vec[current_idx : current_idx + dim]
                    formatted_name = f"{comp_name} [{dim}d]"
                    onnx_input_text += format_vec(formatted_name, val_slice, 2) + "\n"
                    current_idx += dim
            
            if history_len > 1:
                onnx_input_text += f"  ... and {history_len - 1} more historical frames.\n"
            onnx_input_text += f"Total ONNX Input Dim: {len(onnx_input_vec)} (Base {base_obs_dim} x Hist {history_len})\n"

        # --- 輸出與狀態 ---
        torso_lin_vel = sim.data.cvel[sim.torso_id, :3]
        torso_ang_vel_local = _get_local_ang_vel_for_display(sim.data, sim.torso_id)
        output_text = (
            f"--- ONNX OUTPUTS & CONTROL ---\n"
            f"{format_vec('ONNX Raw Action [12d]', state.latest_action_raw)}\n"
            f"{format_vec('Final Motor Ctrl [12d]', state.latest_final_ctrl)}\n"
            f"--- Robot State ---\n"
            f"Torso Z-Pos: {sim.data.qpos[2]:.2f} m\n"
            f"Torso Lin Vel (World): [x={torso_lin_vel[0]:.2f}, y={torso_lin_vel[1]:.2f}, z={torso_lin_vel[2]:.2f}] m/s\n"
            f"Torso Ang Vel (Local): [Rx={torso_ang_vel_local[0]:.2f}, Ry={torso_ang_vel_local[1]:.2f}, Rz={torso_ang_vel_local[2]:.2f}] rad/s\n"
        )
        
        full_text = info_text + onnx_input_text + output_text
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                           viewport, full_text, None, context)