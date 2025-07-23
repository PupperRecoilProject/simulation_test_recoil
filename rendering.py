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
    def __init__(self):
        # 【修改】初始化時不再接收固定的 recipe 和 dims
        self.recipe: List[str] = [] # 初始化一個空列表，用於儲存當前觀察配方
        self.component_dims: Dict[str, int] = {} # 初始化一個空字典，儲存配方中各元件的維度
        
        # 顯示頁面的定義保持不變
        self.display_pages_content = [
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands', 'accelerometer'], # 新增 accelerometer
            ['joint_positions', 'joint_velocities', 'last_action'],
        ]
        state_class_ref = SimulationState # 引用 SimulationState 類別
        state_class_ref.num_display_pages = len(self.display_pages_content) # 設定總顯示頁數

    def set_recipe(self, recipe: List[str]):
        """【新增】動態設定當前要顯示的觀察配方。"""
        self.recipe = recipe # 更新當前配方
        # 根據新配方，更新 component_dims 以便計算
        ALL_OBS_DIMS = {'z_angular_velocity':1, 'gravity_vector':3, 'commands':3, 
                        'joint_positions':12, 'joint_velocities':12, 'foot_contact_states':4, 
                        'linear_velocity':3, 'angular_velocity':3, 'last_action':12, 
                        'phase_signal':1, 'accelerometer': 3}
        # 從總維度字典中，挑出新配方裡存在的元件及其維度
        self.component_dims = {k: ALL_OBS_DIMS[k] for k in recipe if k in ALL_OBS_DIMS}
        print(f"  -> DebugOverlay 切換配方至: {self.recipe}")

    def render(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """根據當前控制模式，選擇並呼叫對應的渲染函式。"""
        if state.control_mode == "HARDWARE_MODE":
            self.render_hardware_overlay(viewport, context, state)
        elif state.control_mode == "SERIAL_MODE":
            self.render_serial_console(viewport, context, state)
        elif state.control_mode == "JOINT_TEST":
            self.render_joint_test_overlay(viewport, context, state, sim)
        elif state.control_mode == "MANUAL_CTRL":
            self.render_manual_ctrl_overlay(viewport, context, state, sim)
        else:
            self.render_simulation_overlay(viewport, context, state, sim)

    def render_hardware_overlay(self, viewport, context, state: SimulationState):
        """渲染硬體控制模式的專用介面。"""
        mujoco.mjr_rectangle(viewport, 0.1, 0.1, 0.1, 0.95)
        ai_status = "啟用" if state.hardware_ai_is_active else "禁用"
        title = f"--- HARDWARE CONTROL MODE (AI: {ai_status}) ---"
        help_text = "Press 'H' to exit | Press 'K' to toggle AI | Press 1-4 to select policy"

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, title, None, context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "\n\n" + help_text, " ", context)
        
        # 動態顯示策略融合狀態
        policy_text = ""
        pm = state.policy_manager_ref
        if pm:
            if pm.is_transitioning:
                source = pm.source_policy_name
                target = pm.target_policy_name
                alpha_percent = pm.transition_alpha * 100
                policy_text = f"Active Policy: Blending {source} -> {target} ({alpha_percent:.0f}%)"
            else:
                policy_text = f"Active Policy: {pm.primary_policy_name}"

        status_text = f"\n\n\n\n--- Real-time Hardware Status ---\n{policy_text}\n{state.hardware_status_text}"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, status_text, None, context)
        
        user_cmd_text = f"\n--- User Command ---\nvy: {state.command[0]:.2f}, vx: {state.command[1]:.2f}, wz: {state.command[2]:.2f}"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, user_cmd_text, None, context)

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
        """渲染關節手動測試模式的專用介面。"""
        mujoco.mjr_rectangle(viewport, 0.2, 0.25, 0.3, 0.9)
        help_text = (
            "--- JOINT TEST MODE ---\n\n"
            "Press '[ / ]' to Select Joint\n"
            "Press UP / DOWN to Adjust Offset\n"
            "Press 'C' to Clear All Offsets\n\n"
            "Press 'G' to Return to Walking Mode"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)
        joint_names = [
            "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee", "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
            "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee", "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
        ]
        num_joints_per_col = 6
        left_col_text, right_col_text = "", ""
        for i, name in enumerate(joint_names):
            prefix = ">> " if i == state.joint_test_index else "   "
            offset_val = state.joint_test_offsets[i]
            final_val = sim.default_pose[i] + offset_val
            line_text = f"{prefix}{name:<15}: Offset={offset_val:+.2f}, Final={final_val:+.2f}\n"
            if i < num_joints_per_col: left_col_text += line_text
            else: right_col_text += line_text
        
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left_col_text, None, context)
        right_col_rect = mujoco.MjrRect(int(viewport.width * 0.45), 0, int(viewport.width * 0.55), viewport.height)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, right_col_rect, right_col_text, None, context)

    def render_manual_ctrl_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染手動 Final Ctrl 模式的專用介面。"""
        floating_status = "Floating" if state.manual_mode_is_floating else "On Ground"
        help_title = f"--- MANUAL CTRL MODE ({floating_status}) ---"
        help_text = (
            f"{help_title}\n\n"
            "Press 'F' to Toggle Floating\n\n"
            "Press '[ / ]' to Select Joint\n"
            "Press UP / DOWN to Adjust Target Angle\n"
            "Press 'C' to Reset All Targets to 0\n\n"
            "Press 'G' to Return to Walking Mode"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)
        joint_names = [
            "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee", "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
            "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee", "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
        ]
        num_joints_per_col = 6
        left_col_text, right_col_text = "", ""
        current_joint_positions = sim.data.qpos[7:]
        for i, name in enumerate(joint_names):
            prefix = ">> " if i == state.manual_ctrl_index else "   "
            target_val = state.manual_final_ctrl[i]
            actual_val = current_joint_positions[i]
            error = target_val - actual_val
            line_text = f"{prefix}{name:<15}: Target={target_val:+.2f}, Actual={actual_val:+.2f}, Err={error:+.2f}\n"
            if i < num_joints_per_col: left_col_text += line_text
            else: right_col_text += line_text
        
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left_col_text, None, context)
        right_col_rect = mujoco.MjrRect(int(viewport.width * 0.40), 0, int(viewport.width * 0.60), viewport.height)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, right_col_rect, right_col_text, None, context)

    def render_simulation_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染正常的模擬除錯資訊。"""
        def format_vec(label: str, vec, precision=3, label_width=24):
            if vec is None or vec.size == 0: return f"{label:<{label_width}}None"
            vec_str = np.array2string(vec, precision=precision, floatmode='fixed', suppress_small=True, threshold=100)
            return f"{label:<{label_width}}{vec_str}"

        # --- 【修改】更新幫助文本 ---
        help_text = (
            "--- CONTROLS ---\n\n"
            "[Universal]\n"
            "  SPACE: Pause/Play | N: Next Step\n"
            "  F: Float | G: Joint Test/Exit | B: Manual Ctrl\n"
            "  ESC: Exit       | R: Hard Reset  | T: Serial Console\n"
            "  X: Soft Reset   | Y: Regen Terrain | H: Hardware Mode\n"
            "  P: Save Terrain PNG\n\n"
            "[Input & Policy]\n"
            "  M: Input Mode   | C: Clear Cmd   | 1-4: Select Policy\n"
            "  U: Scan Serial  | J: Scan Gamepad| K: Toggle HW AI\n"
            "  V: Cycle Terrain Mode\n\n" 
            "[Keyboard Mode]\n"
            "  WASD/QE: Move/Turn\n"
            "  [/]: Select Param | UP/DOWN: Adjust Value\n\n"
            "[Gamepad Mode]\n"
            "  L-Stick: Move | R-Stick: Turn\n"
            "  LB/RB: Select Param | D-Pad U/D: Adjust Value\n"
            "  Select/View: Reset"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)
        
        serial_status = "Connected" if state.serial_is_connected else "Disconnected (U to Scan)"
        gamepad_status = "Connected" if state.gamepad_is_connected else "Disconnected (J to Scan)"
        # 【修改】從 terrain_manager 獲取新的、包含模式的名稱
        terrain_name = state.terrain_manager_ref.get_current_terrain_name(state) if state.terrain_manager_ref else "N/A"

        policy_text = ""
        pm = state.policy_manager_ref
        if pm:
            if pm.is_transitioning:
                source = pm.source_policy_name
                target = pm.target_policy_name
                alpha_percent = pm.transition_alpha * 100
                policy_text = f"Policy: Blending {source} -> {target} ({alpha_percent:.0f}%)"
            else:
                policy_text = f"Policy: {pm.primary_policy_name}"

        p = state.tuning_params
        prefixes = ["   "] * 4
        prefixes[state.tuning_param_index] = ">> "

        top_left_text = (
            f"Mode: {state.control_mode} | Input: {state.input_mode}\n"
            f"{policy_text}\n"
            f"Time: {sim.data.time:.2f} s\n"
            f"Terrain: {terrain_name}\n\n"
            f"--- Devices ---\n"
            f"Serial Console: {serial_status}\n"
            f"Gamepad: {gamepad_status}\n\n"
            f"--- Tuning Params ---\n"
            f"{prefixes[0]}{format_vec('Kp:', np.array([p.kp]), 1)}\n"
            f"{prefixes[1]}{format_vec('Kd:', np.array([p.kd]), 2)}\n"
            f"{prefixes[2]}{format_vec('Act Scale:', np.array([p.action_scale]), 3)}\n"
            f"{prefixes[3]}{format_vec('Bias:', np.array([p.bias]), 1)}\n\n"
            f"--- Command ---\n"
            f"{format_vec('User Cmd:', state.command)}\n"
        )
        if state.control_mode == "FLOATING":
            current_height = sim.data.qpos[2]
            # 顯示目標世界Z座標，而不是相對高度，這樣更直觀
            target_world_z = state.floating_controller_ref.data.mocap_pos[state.floating_controller_ref.mocap_index][2]
            top_left_text += (
                f"\n--- Floating Info ---\n"
                f"{format_vec('Target World Z:', np.array([target_world_z]), 3)}\n"
                f"{format_vec('Current Z:', np.array([current_height]), 3)}\n"
            )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, top_left_text, None, context)
        
        bottom_left_text = f"--- ONNX INPUTS (Page {state.display_page + 1}/{state.num_display_pages}) ---\n"
        onnx_input_vec = state.latest_onnx_input
        if onnx_input_vec.size > 0 and self.recipe and state.display_page < len(self.display_pages_content):
            current_page_components = self.display_pages_content[state.display_page]
            base_obs_dim = sum(self.component_dims.values()) if self.component_dims else 0
            if base_obs_dim > 0:
                history_len = len(onnx_input_vec) // base_obs_dim
                current_frame_obs = onnx_input_vec[-base_obs_dim:]
                
                current_full_obs_idx = 0
                for comp_name_in_recipe in self.recipe:
                    dim = self.component_dims.get(comp_name_in_recipe, 0)
                    if dim > 0:
                        if comp_name_in_recipe in current_page_components:
                            start_idx, end_idx = current_full_obs_idx, current_full_obs_idx + dim
                            value_slice = current_frame_obs[start_idx:end_idx]
                            bottom_left_text += format_vec(f"{comp_name_in_recipe} [{dim}d]:", value_slice, 2) + "\n"
                        current_full_obs_idx += dim
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, bottom_left_text, None, context)
        
        torso_lin_vel = sim.data.cvel[sim.torso_id, 3:]
        torso_ang_vel_local = self._get_local_ang_vel(sim.data, sim.torso_id)
        bottom_right_text = (
            f"--- ONNX OUTPUTS & STATE ---\n"
            f"{format_vec('Final Action:', state.latest_action_raw)}\n"
            f"{format_vec('Final Ctrl:', state.latest_final_ctrl)}\n\n"
            f"--- Robot State (Sim) ---\n"
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