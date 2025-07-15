# rendering.py
import mujoco
import numpy as np
from state import SimulationState

def _get_local_ang_vel_for_display(data, torso_id: int) -> np.ndarray:
    """
    獲取軀幹在局部座標系的完整角速度，僅用於除錯顯示。
    
    Args:
        data: MuJoCo data object.
        torso_id: The integer ID of the torso body.

    Returns:
        A numpy array of shape (3,) representing local angular velocity.
    """
    torso_quat = data.xquat[torso_id] # 獲取軀幹的四元數姿態
    norm = np.sum(np.square(torso_quat))
    if norm < 1e-8:
        return np.zeros(3)
    torso_quat /= np.sqrt(norm) # 標準化四元數
    
    # 計算逆四元數，用於將向量從世界座標系轉換到局部座標系
    q_inv = np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm
    u, s = q_inv[1:], q_inv[0]
    world_ang_vel = data.cvel[torso_id, 3:] # 獲取世界座標系下的角速度
    
    # 使用逆四元數旋轉世界角速度向量，得到局部角速度
    return 2 * np.dot(u, world_ang_vel) * u + (s*s - np.dot(u, u)) * world_ang_vel + 2*s*np.cross(u, world_ang_vel)


class DebugOverlay:
    """在 MuJoCo 視窗上覆蓋顯示除錯資訊。"""
    def __init__(self, recipe: list, recipe_dims: dict):
        """
        初始化 DebugOverlay。

        Args:
            recipe (list): 當前使用的觀察配方列表。
            recipe_dims (dict): 配方中各元件及其維度的字典。
        """
        self.recipe = recipe
        self.component_dims = recipe_dims
        # 定義每個顯示頁面包含哪些觀察元件
        # 可以根據實際模型輸入的順序和長度來調整這些列表
        self.display_pages_content = [
            # Page 1: 基礎運動學感測器和指令
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands'],
            # Page 2: 機器人關節的詳細狀態和前一步動作
            ['joint_positions', 'joint_velocities', 'last_action'],
            # Page 3: 接觸和相位等額外資訊
            ['foot_contact_states', 'phase_signal']
        ]

    def render(self, viewport, context, state: SimulationState, sim):
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
            """輔助函式，將向量格式化為單行完整顯示的字串。"""
            if vec is None or vec.size == 0: return f"{name:<22}: None"
            return f"{name:<22}: " + np.array2string(vec, precision=precision, 
                                                    floatmode='fixed', suppress_small=True, threshold=100)

        # --- TOP RIGHT: Controls Help Text ---
        help_text = (
            "--- CONTROLS ---\n\n"
            "[Universal]\n"
            "  ESC: Exit  |  R: Reset\n"
            "  M: Toggle Input Mode\n"
            "  TAB: Toggle Info Page\n"
            "  C: C: Clear Cmd\n\n"
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
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                           viewport, help_text, None, context) # 在右上角繪製幫助文字

        # --- TOP LEFT: General Info & Tuning Params ---
        p = state.tuning_params
        top_left_info_text = (
            f"Mode: {state.sim_mode_text} | Input: {state.input_mode}\n"
            f"Time: {sim.data.time:.2f} s\n"
            f"--- Tuning Params ---\n"
            f"Kp: {p.kp:.1f} | Kd: {p.kd:.2f} | ActScl: {p.action_scale:.3f} | Bias: {p.bias:.1f}\n"
            f"--- Command ---\n"
            f"{format_vec('User Command', state.command)}\n"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                           viewport, top_left_info_text, None, context) # 在左上角繪製通用資訊

        # --- BOTTOM LEFT: ONNX Inputs (Breakdown) - PAGED VIEW ---
        bottom_left_text = f"--- ONNX INPUTS (Page {state.display_page + 1}/{state.num_display_pages}) ---\n"
        
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0:
            # 獲取當前頁面應該顯示的元件列表
            current_page_components = self.display_pages_content[state.display_page]
            
            base_obs_dim = sum(self.component_dims.values())
            history_len = len(onnx_input_vec) // base_obs_dim if base_obs_dim > 0 else 1 
            
            # 從 ONNX 完整輸入中提取並顯示當前頁面的數據
            current_full_obs_idx = 0
            for comp_name_in_recipe in self.recipe: # 遍歷配方中的所有組件以找到其在完整向量中的位置
                dim = self.component_dims.get(comp_name_in_recipe, 0)
                if dim > 0:
                    if comp_name_in_recipe in current_page_components:
                        # 如果這個組件屬於當前頁面，則顯示它
                        # 我們只顯示最新一幀 (history[0]) 的數據以保持簡潔
                        start_idx = current_full_obs_idx
                        end_idx = start_idx + dim
                        value_slice = onnx_input_vec[start_idx : end_idx]

                        formatted_name = f"{comp_name_in_recipe} [{dim}d]"
                        bottom_left_text += format_vec(formatted_name, value_slice, 2) + "\n"
                    current_full_obs_idx += dim # 更新下一個組件的起始偏移量
                
            if history_len > 1:
                bottom_left_text += f"  (From latest frame of {history_len}-frame history)\n"
            bottom_left_text += f"Total Dim: {len(onnx_input_vec)} (Base {base_obs_dim} x Hist {history_len})\n"
        else:
            bottom_left_text += "  (ONNX input not yet available)\n" # 在 Warmup 期間顯示

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 
                           viewport, bottom_left_text, None, context) # 在左下角繪製 ONNX 輸入分解

        # --- BOTTOM RIGHT: ONNX Outputs & Robot State ---
        torso_lin_vel = sim.data.cvel[sim.torso_id, :3]
        torso_ang_vel_local = _get_local_ang_vel_for_display(sim.data, sim.torso_id)
        bottom_right_detail_text = (
            f"--- ONNX OUTPUTS & CONTROL ---\n"
            f"{format_vec('ONNX Raw Action', state.latest_action_raw)}\n"
            f"{format_vec('Final Motor Ctrl', state.latest_final_ctrl)}\n"
            f"--- Robot State ---\n"
            f"Torso Z-Pos: {sim.data.qpos[2]:.3f} m\n"
            f"Torso Lin Vel (World): [{torso_lin_vel[0]:.2f}, {torso_lin_vel[1]:.2f}, {torso_lin_vel[2]:.2f}]\n"
            f"Torso Ang Vel (Local): [{torso_ang_vel_local[0]:.2f}, {torso_ang_vel_local[1]:.2f}, {torso_ang_vel_local[2]:.2f}]"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, 
                           viewport, bottom_right_detail_text, None, context) # 在右下角繪製輸出和機器人狀態