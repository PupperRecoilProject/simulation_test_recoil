# rendering.py
import mujoco
import numpy as np
import time
from state import SimulationState
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from simulation import Simulation

class DebugOverlay:
    """
    負責在 MuJoCo 視窗上渲染所有文字除錯資訊。
    """
    def __init__(self, recipe: List[str], recipe_dims: Dict[str, int]):
        self.recipe = recipe
        self.component_dims = recipe_dims
        self.display_pages_content = [
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands'],
            ['joint_positions', 'joint_velocities', 'last_action'],
        ]
        state_class_ref = SimulationState
        state_class_ref.num_display_pages = len(self.display_pages_content)

    def render(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """根據當前控制模式，選擇並呼叫對應的渲染函式。"""
        if state.control_mode == "SERIAL_MODE":
            self.render_serial_console(viewport, context, state)
        elif state.control_mode == "JOINT_TEST":
            self.render_joint_test_overlay(viewport, context, state, sim)
        else:
            self.render_simulation_overlay(viewport, context, state, sim)

    def render_serial_console(self, viewport, context, state: SimulationState):
        """渲染一個全螢幕的序列埠控制台介面。"""
        mujoco.mjr_rectangle(viewport, 0.2, 0.2, 0.2, 0.9)
        title = "--- SERIAL CONSOLE MODE (Press T to exit) ---"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, title, None, context)
        log_text = "\n".join(state.serial_latest_messages)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "\n\n" + log_text, " ", context)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        buffer_text = f"> {state.serial_command_buffer}{cursor}"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, buffer_text, None, context)
    
    def render_joint_test_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染關節手動測試模式的專用介面，使用分欄顯示。"""
        mujoco.mjr_rectangle(viewport, 0.2, 0.25, 0.3, 0.9)

        help_text = (
            "--- JOINT TEST MODE ---\n\n"
            "Press '1' / '2' to Select Joint\n"
            "Press UP / DOWN to Adjust Offset\n"
            "Press 'C' to Clear All Offsets\n\n"
            "Press 'G' to Return to Walking Mode"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)

        joint_names = [
            "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee",
            "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
            "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee",
            "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
        ]
        
        # --- 修正點：將列表分成兩半，分別建立左右兩欄的文字 ---
        num_joints_per_col = 6
        left_col_text = ""
        right_col_text = ""

        for i, name in enumerate(joint_names):
            prefix = ">> " if i == state.joint_test_index else "   "
            offset_val = state.joint_test_offsets[i]
            final_val = sim.default_pose[i] + offset_val
            line_text = f"{prefix}{name:<15}: Offset={offset_val:+.2f}, Final={final_val:+.2f}\n"
            
            if i < num_joints_per_col:
                left_col_text += line_text
            else:
                right_col_text += line_text
        
        # 分別在左上角和中上部渲染兩欄文字
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left_col_text, None, context)
        
        # 為了在中上部渲染，我們需要手動指定一個矩形區域
        # 這裡我們簡單地將第二欄放在一個固定的偏移位置
        # 獲取視窗寬高以便計算位置
        width = viewport.width
        height = viewport.height
        # 建立一個新的 MjrRect 來定義第二欄的渲染區域
        right_col_rect = mujoco.MjrRect(int(width * 0.3), 0, int(width * 0.3), height) # 從寬度30%處開始
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, right_col_rect, right_col_text, None, context)

    def render_simulation_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染正常的模擬除錯資訊。"""
        # (這個函式保持不變)
        def format_vec(label: str, vec, precision=3, label_width=24):
            if vec is None or vec.size == 0: return f"{label:<{label_width}}None"
            vec_str = np.array2string(vec, precision=precision, floatmode='fixed', suppress_small=True, threshold=100)
            return f"{label:<{label_width}}{vec_str}"

        help_text = (
            "--- CONTROLS ---\n\n"
            "[Universal]\n"
            "  F: Float | G: Joint Test | T: Serial\n"
            "  ESC: Exit       | R: Reset\n"
            "  M: Input Mode   | TAB: Info Page\n"
            "  C: Clear Cmd (Keyboard)\n\n"
            "[Keyboard Mode]\n"
            "  WASD/QE: Move/Turn\n"
            "  I/K: Kp | L/J: Kd\n"
            "  Y/H: ActScl | P/;: Bias\n\n"
            "[Gamepad Mode]\n"
            "  L-Stick: Move | R-Stick: Turn\n"
            "  D-Pad U/D: Kp | D-Pad R/L: Kd\n"
            "  LB/RB: ActScl | Y/A: Bias\n"
            "  Select/View: Reset"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)

        p = state.tuning_params
        top_left_text = (
            f"Mode: {state.control_mode} | Input: {state.input_mode}\n"
            f"Time: {sim.data.time:.2f} s\n\n"
            f"--- Tuning Params ---\n"
            f"{format_vec('Kp:', np.array([p.kp]), 1)}\n"
            f"{format_vec('Kd:', np.array([p.kd]), 2)}\n"
            f"{format_vec('Act Scale:', np.array([p.action_scale]), 3)}\n"
            f"{format_vec('Bias:', np.array([p.bias]), 1)}\n\n"
            f"--- Command ---\n"
            f"{format_vec('User Cmd:', state.command)}\n"
        )
        if state.control_mode == "FLOATING":
            current_height = sim.data.qpos[2]
            target_height = sim.config.floating_controller.target_height
            top_left_text += (
                f"\n--- Floating Info ---\n"
                f"{format_vec('Target H:', np.array([target_height]), 3)}\n"
                f"{format_vec('Current H:', np.array([current_height]), 3)}\n"
            )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, top_left_text, None, context)
        
        bottom_left_text = f"--- ONNX INPUTS (Page {state.display_page + 1}/{state.num_display_pages}) ---\n"
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0 and state.display_page < len(self.display_pages_content):
            current_page_components = self.display_pages_content[state.display_page]
            base_obs_dim = sum(self.component_dims.values()) if self.component_dims else 0
            if base_obs_dim > 0:
                history_len = len(onnx_input_vec) // base_obs_dim
                current_full_obs_idx = 0
                for comp_name_in_recipe in self.recipe:
                    dim = self.component_dims.get(comp_name_in_recipe, 0)
                    if dim > 0:
                        if comp_name_in_recipe in current_page_components:
                            start_idx, end_idx = current_full_obs_idx, current_full_obs_idx + dim
                            value_slice = onnx_input_vec[start_idx:end_idx]
                            bottom_left_text += format_vec(f"{comp_name_in_recipe} [{dim}d]:", value_slice, 2) + "\n"
                        current_full_obs_idx += dim
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, bottom_left_text, None, context)
        
        torso_lin_vel = sim.data.cvel[sim.torso_id, 3:]
        torso_ang_vel_local = self._get_local_ang_vel(sim.data, sim.torso_id)
        bottom_right_text = (
            f"--- ONNX OUTPUTS & STATE ---\n"
            f"{format_vec('Raw Action:', state.latest_action_raw)}\n"
            f"{format_vec('Final Ctrl:', state.latest_final_ctrl)}\n\n"
            f"--- Robot State ---\n"
            f"{format_vec('Torso Z:', np.array([sim.data.qpos[2]]))}\n"
            f"{format_vec('Lin Vel (World):', torso_lin_vel)}\n"
            f"{format_vec('Ang Vel (Local):', torso_ang_vel_local)}"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, viewport, bottom_right_text, None, context)
    
    def _get_local_ang_vel(self, data, torso_id):
        """輔助函式，計算局部角速度用於顯示。"""
        torso_quat = data.xquat[torso_id]
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8: return np.zeros(3)
        torso_quat /= np.sqrt(norm)
        q_inv = np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm
        u, s = q_inv[1:], q_inv[0]
        world_ang_vel = data.cvel[torso_id, :3]
        return 2 * np.dot(u, world_ang_vel) * u + (s*s - np.dot(u, u)) * world_ang_vel + 2*s*np.cross(u, world_ang_vel)