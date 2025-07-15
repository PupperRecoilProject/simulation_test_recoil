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
        # 定義每個顯示頁面包含哪些觀察元件
        # 你可以根據實際模型輸入的順序和長度來調整這些列表
        self.display_pages_content = [
            # Page 0: 常見的基礎感測器和指令
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands'],
            # Page 1: 機器人關節的詳細狀態和前一步動作
            ['joint_positions', 'joint_velocities', 'last_action'],
            # 如果還有其他觀察元件，可以考慮新增 Page 2, Page 3 ...
            # 例如: ['foot_contact_states', 'phase_signal']
        ]

    def render(self, viewport, context, state, sim):
        """
        將所有資訊格式化成文字並渲染到指定的視口上。
        
        Args:
            viewport: MuJoCo 視口物件。
            context: MuJoCo 渲染上下文。
            state (SimulationState): 當前的應用程式狀態。
            sim: Simulation 實例，用於獲取 MuJoCo data。
        """
        # --- Helper for formatting vectors ---
        def format_vec(name, vec, precision=3):
            if vec is None: return f"{name:<22}: None"
            return f"{name:<22}: " + np.array2string(vec, precision=precision, 
                                                    floatmode='fixed', suppress_small=True, threshold=100)

        # --- TOP RIGHT: Controls ---
        help_text = ("--- CONTROLS ---\n\n"
                     "[Keyboard]\n"
                     "  WASD/QE: Move/Turn\n"
                     "  R: Reset | C: Clear Cmd\n"
                     "  I/K: Kp +/- | L/J: Kd +/-\n"
                     "  Y/H: ActScl +/- | P/;: Bias +/-\n"
                     "  ESC: Exit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                           viewport, help_text, None, context)

        # --- TOP LEFT: General Info & Tuning Params ---
        p = state.tuning_params
        top_left_info_text = (
            f"Mode: {state.mode_text} (Time: {sim.data.time:.2f} s)\n"
            f"--- Tuning Params ---\n"
            f"Kp: {p.kp:.1f} | Kd: {p.kd:.2f} | ActScl: {p.action_scale:.3f} | Bias: {p.bias:.1f}\n"
            f"--- Command ---\n"
            f"{format_vec('User Command', state.command)}\n"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                           viewport, top_left_info_text, None, context)

        # --- BOTTOM LEFT: ONNX Inputs (Breakdown) - PAGED VIEW ---
        bottom_left_text = f"--- ONNX INPUTS (Breakdown) - Page {state.display_page + 1}/{state.num_display_pages} ---\n"
        
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0:
            # 獲取當前頁面應該顯示的元件列表
            current_page_components = self.display_pages_content[state.display_page]
            
            base_obs_dim = sum(self.component_dims.values())
            history_len = len(onnx_input_vec) // base_obs_dim if base_obs_dim > 0 else 1 
            
            # 從 ONNX 完整輸入中提取並顯示當前頁面的數據
            # 注意：這裡假設 ONNX 輸入是按 recipe_dims 順序排列的
            # 我們需要找到這些組件在完整 ONNX 輸入中的起始位置
            
            current_full_obs_idx = 0
            for comp_name_in_recipe in self.recipe: # 遍歷所有在 recipe 中的組件以找到索引
                dim = self.component_dims.get(comp_name_in_recipe, 0)
                if dim > 0:
                    if comp_name_in_recipe in current_page_components:
                        # 如果這個組件屬於當前頁面，則顯示它
                        # 取出所有歷史幀中該組件的數據並拼接
                        val_slice_all_history = []
                        for i in range(history_len):
                            start_idx = current_full_obs_idx + i * base_obs_dim
                            end_idx = start_idx + dim
                            if end_idx <= len(onnx_input_vec):
                                val_slice_all_history.append(onnx_input_vec[start_idx : end_idx])
                            else:
                                val_slice_all_history.append(np.zeros(dim)) # 處理不完整的歷史數據
                        
                        # 這裡我們只顯示第一幀（最新幀）的數據，避免過長
                        # 如果需要顯示多幀的數據，可能需要重新考慮格式化
                        val_to_display = val_slice_all_history[0] if val_slice_all_history else None

                        formatted_name = f"{comp_name_in_recipe} [{dim}d]"
                        bottom_left_text += format_vec(formatted_name, val_to_display, 2) + "\n"
                    current_full_obs_idx += dim # 更新總偏移量
                
            if history_len > 1:
                bottom_left_text += f"  ... and {history_len - 1} more historical frames (hidden).\n"
            bottom_left_text += f"Total ONNX Input Dim: {len(onnx_input_vec)} (Base {base_obs_dim} x Hist {history_len})\n"
        else:
            bottom_left_text += "  (ONNX input not yet available or empty)\n" # During warmup

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 
                           viewport, bottom_left_text, None, context)

        # --- BOTTOM RIGHT: ONNX Outputs & Robot State ---
        torso_lin_vel = sim.data.cvel[sim.torso_id, :3]
        torso_ang_vel_local = _get_local_ang_vel_for_display(sim.data, sim.torso_id)
        bottom_right_detail_text = (
            f"--- ONNX OUTPUTS & CONTROL ---\n"
            f"{format_vec('ONNX Raw Action [12d]', state.latest_action_raw)}\n"
            f"{format_vec('Final Motor Ctrl [12d]', state.latest_final_ctrl)}\n"
            f"--- Robot State ---\n"
            f"Torso Z-Pos: {sim.data.qpos[2]:.2f} m\n"
            f"Torso Lin Vel (World): [x={torso_lin_vel[0]:.2f}, y={torso_lin_vel[1]:.2f}, z={torso_lin_vel[2]:.2f}] m/s\n"
            f"Torso Ang Vel (Local): [Rx={torso_ang_vel_local[0]:.2f}, Ry={torso_ang_vel_local[1]:.2f}, Rz={torso_ang_vel_local[2]:.2f}] rad/s\n"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, 
                           viewport, bottom_right_detail_text, None, context)
