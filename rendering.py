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
        self.recipe: List[str] = [] # 儲存當前模型使用的觀察配方
        self.component_dims: Dict[str, int] = {} # 儲存配方中各元件的維度
        
        # 定義不同顯示頁面對應的觀察元件
        self.display_pages_content = [
            ['linear_velocity', 'angular_velocity', 'gravity_vector', 'commands', 'accelerometer'],
            ['joint_positions', 'joint_velocities', 'last_action'],
        ]
        state_class_ref = SimulationState # 獲取 SimulationState 類別的參考
        state_class_ref.num_display_pages = len(self.display_pages_content) # 將總頁數設定到 State 類別中

    def set_recipe(self, recipe: List[str]):
        """動態設定當前要顯示的觀察配方。"""
        self.recipe = recipe # 更新當前配方
        # 所有可能的觀察元件及其維度
        ALL_OBS_DIMS = {'z_angular_velocity':1, 'gravity_vector':3, 'commands':3, 
                        'joint_positions':12, 'joint_velocities':12, 'foot_contact_states':4, 
                        'linear_velocity':3, 'angular_velocity':3, 'last_action':12, 
                        'phase_signal':1, 'accelerometer': 3}
        # 根據傳入的配方，建立一個僅包含當前所需元件維度的字典
        self.component_dims = {k: ALL_OBS_DIMS[k] for k in recipe if k in ALL_OBS_DIMS}
        print(f"  -> DebugOverlay 切換配方至: {self.recipe}") # 在控制台輸出提示

    def render(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """
        【核心修改】統一渲染邏輯。
        無論在哪種模式下，都會先渲染3D場景，然後再疊加對應模式的文字資訊。
        """
        # --- 步驟 1: 始終更新和渲染 3D 場景 ---
        # 確保攝影機追蹤機器人 (除非使用者正在手動操作視角)
        if not (sim.mouse_button_left or sim.mouse_button_right): # 檢查滑鼠左右鍵是否被按下
             sim.cam.lookat = sim.data.body('torso').xpos # 將攝影機焦點設定為軀幹位置

        # 如果地形被更新，則將新數據上傳到GPU
        terrain_manager = getattr(state, 'terrain_manager_ref', None) # 從 state 安全地獲取地形管理器參考
        if terrain_manager and terrain_manager.needs_scene_update: # 檢查地形管理器是否存在且需要更新
            mujoco.mjr_uploadHField(sim.model, sim.context, terrain_manager.hfield_id) # 上傳高度場數據到渲染上下文
            terrain_manager.needs_scene_update = False # 重置更新旗標
            print("🔄 地形幾何已上傳至 GPU 進行渲染。")
        
        # 更新場景物件並進行渲染
        mujoco.mjv_updateScene(sim.model, sim.data, sim.opt, None, sim.cam, mujoco.mjtCatBit.mjCAT_ALL, sim.scene) # 更新 MuJoCo 渲染場景
        mujoco.mjr_render(viewport, sim.scene, sim.context) # 執行渲染
        
        # --- 步驟 2: 根據當前模式，選擇並疊加對應的文字資訊 ---
        if state.control_mode == "HARDWARE_MODE": # 如果是硬體模式
            self.render_hardware_overlay(viewport, context, state) # 呼叫硬體模式的渲染函式
        elif state.control_mode == "SERIAL_MODE": # 如果是序列埠模式
            self.render_serial_console(viewport, context, state) # 呼叫序列埠模式的渲染函式
        elif state.control_mode == "JOINT_TEST": # 如果是關節測試模式
            self.render_joint_test_overlay(viewport, context, state, sim) # 呼叫關節測試模式的渲染函式
        elif state.control_mode == "MANUAL_CTRL": # 如果是手動控制模式
            self.render_manual_ctrl_overlay(viewport, context, state, sim) # 呼叫手動控制模式的渲染函式
        else: # 其他所有模式（如 WALKING, FLOATING）
            self.render_simulation_overlay(viewport, context, state, sim) # 呼叫預設的模擬資訊渲染函式

    def render_hardware_overlay(self, viewport, context, state: SimulationState):
        """【介面修正】渲染硬體控制模式的專用介面，使用 MjrRect 進行精確排版。"""
        # --- 定義主狀態面板 (左上角) ---
        padding = 10 # 定義面板與視窗邊緣的間距
        panel_width = int(viewport.width * 0.45) # 面板寬度為視窗的 45%
        panel_height = int(viewport.height * 0.6) # 面板高度為視窗的 60%
        top_left_rect = mujoco.MjrRect(padding, viewport.height - panel_height - padding, panel_width, panel_height) # 建立左上角矩形區域

        # --- 繪製主狀態面板背景 ---
        mujoco.mjr_rectangle(top_left_rect, 0.1, 0.1, 0.1, 0.8) # 在定義的矩形區域內繪製半透明黑色背景

        # --- 準備並繪製主狀態面板文字 ---
        # 【中文化修正】將狀態文字從中文 "啟用/禁用" 改為英文，以確保在 MuJoCo 視窗中正確顯示。
        ai_status = "Enabled" if state.hardware_ai_is_active else "Disabled" # 根據狀態決定 AI 狀態文字
        title = f"--- HARDWARE CONTROL MODE (AI: {ai_status}) ---" # 組合標題文字
        help_text = "Press 'H' to exit | 'K': Toggle AI | 'G': Joint Test | 1..: Select Policy" # 幫助文字

        policy_text = "" # 初始化策略文字
        pm = state.policy_manager_ref # 獲取策略管理器
        if pm: # 如果策略管理器存在
            if pm.is_transitioning: # 如果正在切換策略
                source = pm.source_policy_name # 來源策略名稱
                target = pm.target_policy_name # 目標策略名稱
                alpha_percent = pm.transition_alpha * 100 # 計算切換進度百分比
                policy_text = f"Active Policy: Blending {source} -> {target} ({alpha_percent:.0f}%)" # 組合策略切換狀態文字
            else: # 如果不在切換中
                policy_text = f"Active Policy: {pm.primary_policy_name}" # 顯示當前主要策略

        status_text = f"--- Real-time Hardware Status ---\n{state.hardware_status_text}" # 組合硬體狀態文字

        sensor_text = "" # 初始化感測器文字
        hw_ctrl = state.hardware_controller_ref # 獲取硬體控制器
        if hw_ctrl and hw_ctrl.is_running: # 如果硬體控制器存在且在運行中
            with hw_ctrl.lock: # 使用執行緒鎖確保資料安全
                imu_acc_str = np.array2string(hw_ctrl.hw_state.imu_acc_g, precision=2, suppress_small=True) # 格式化 IMU 加速度數據
                joint_pos_str = np.array2string(hw_ctrl.hw_state.joint_positions_rad, precision=2, suppress_small=True, max_line_width=80) # 格式化關節角度數據
                sensor_text = ( # 組合感測器讀數文字
                    f"\n\n--- Sensor Readings (from Robot) ---\n"
                    f"IMU Acc (g): {imu_acc_str}\n"
                    f"Joint Pos (rad):\n{joint_pos_str}"
                )
        
        # 將所有文字組合在一起，用換行符分隔
        full_text = f"{title}\n\n{help_text}\n\n{policy_text}\n\n{status_text}{sensor_text}"
        # 在左上角矩形內繪製所有文字
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, top_left_rect, full_text, " ", context)

        # --- 定義並繪製使用者命令面板 (左下角) ---
        cmd_panel_height = int(viewport.height * 0.1) # 命令面板高度為視窗的 10%
        bottom_left_rect = mujoco.MjrRect(padding, padding, panel_width, cmd_panel_height) # 建立左下角矩形區域
        mujoco.mjr_rectangle(bottom_left_rect, 0.1, 0.1, 0.1, 0.8) # 繪製背景

        user_cmd_text = f"--- User Command ---\nvy: {state.command[0]:.2f}, vx: {state.command[1]:.2f}, wz: {state.command[2]:.2f}" # 組合使用者命令文字
        # 在左下角矩形內繪製命令文字
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, bottom_left_rect, user_cmd_text, " ", context)


    def render_serial_console(self, viewport, context, state: SimulationState):
        """【介面修正】渲染序列埠控制台介面，使其置中且大小適中。"""
        # --- 定義控制台面板 ---
        panel_width = int(viewport.width * 0.8) # 面板寬度為視窗的 80%
        panel_height = int(viewport.height * 0.9) # 面板高度為視窗的 90%
        left = (viewport.width - panel_width) // 2 # 計算左邊界以使其水平置中
        bottom = (viewport.height - panel_height) // 2 # 計算下邊界以使其垂直置中
        console_rect = mujoco.MjrRect(left, bottom, panel_width, panel_height) # 建立置中的矩形區域

        # --- 繪製背景和文字 ---
        mujoco.mjr_rectangle(console_rect, 0.2, 0.2, 0.2, 0.9) # 繪製半透明背景

        title = "--- SERIAL CONSOLE MODE (Press T to exit) ---" # 標題文字
        # 在矩形頂部繪製標題
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, console_rect, title, " ", context)
        
        log_text = "\n".join(state.serial_latest_messages) # 將訊息日誌列表轉換為單一字串
        
        # 為日誌內容定義一個新的、稍微偏移的矩形，以產生邊距效果
        log_rect = mujoco.MjrRect(console_rect.left + 10, console_rect.bottom, console_rect.width - 20, console_rect.height - 50)
        # 在標題下方繪製日誌內容
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, log_rect, "\n\n" + log_text, " ", context)

        cursor = "_" if int(time.time() * 2) % 2 == 0 else " " # 產生閃爍的游標效果
        buffer_text = f"> {state.serial_command_buffer}{cursor}" # 組合輸入緩衝區文字
        # 在矩形底部繪製輸入提示符和緩衝區內容
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, console_rect, buffer_text, " ", context)
    
    def render_joint_test_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染關節手動測試模式的專用介面。"""
        mujoco.mjr_rectangle(viewport, 0.2, 0.25, 0.3, 0.9) # 繪製背景
        # 幫助文字
        help_text = (
            "--- JOINT TEST MODE ---\n\n"
            "Press '[ / ]' to Select Joint\n"
            "Press UP / DOWN to Adjust Offset\n"
            "Press 'C' to Clear All Offsets\n\n"
            "Press 'G' to Return to Walking Mode"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context) # 繪製幫助文字
        # 關節名稱列表
        joint_names = [
            "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee", "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
            "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee", "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
        ]
        num_joints_per_col = 6 # 每列顯示的關節數
        left_col_text, right_col_text = "", "" # 初始化左右兩列的文字
        for i, name in enumerate(joint_names): # 遍歷所有關節
            prefix = ">> " if i == state.joint_test_index else "   " # 如果是當前選中的關節，則加上前綴
            offset_val = state.joint_test_offsets[i] # 獲取偏移值
            final_val = sim.default_pose[i] + offset_val # 計算最終角度
            line_text = f"{prefix}{name:<15}: Offset={offset_val:+.2f}, Final={final_val:+.2f}\n" # 格式化單行文字
            if i < num_joints_per_col: left_col_text += line_text # 加入左列
            else: right_col_text += line_text # 加入右列
        
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left_col_text, None, context) # 繪製左列文字
        right_col_rect = mujoco.MjrRect(int(viewport.width * 0.45), 0, int(viewport.width * 0.55), viewport.height) # 為右列定義一個新的矩形區域
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, right_col_rect, right_col_text, None, context) # 在新區域中繪製右列文字

    def render_manual_ctrl_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染手動 Final Ctrl 模式的專用介面。"""
        floating_status = "Floating" if state.manual_mode_is_floating else "On Ground" # 獲取懸浮狀態
        help_title = f"--- MANUAL CTRL MODE ({floating_status}) ---" # 組合標題
        # 幫助文字
        help_text = (
            f"{help_title}\n\n"
            "Press 'F' to Toggle Floating\n\n"
            "Press '[ / ]' to Select Joint\n"
            "Press UP / DOWN to Adjust Target Angle\n"
            "Press 'C' to Reset All Targets to 0\n\n"
            "Press 'G' to Return to Walking Mode"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context) # 繪製幫助文字
        # 關節名稱
        joint_names = [
            "0: FR_Abduction", "1: FR_Hip", "2: FR_Knee", "3: FL_Abduction", "4: FL_Hip", "5: FL_Knee",
            "6: RR_Abduction", "7: RR_Hip", "8: RR_Knee", "9: RL_Abduction", "10: RL_Hip", "11: RL_Knee"
        ]
        num_joints_per_col = 6 # 每列顯示的關節數
        left_col_text, right_col_text = "", "" # 初始化左右列文字
        current_joint_positions = sim.data.qpos[7:] # 獲取當前實際關節角度
        for i, name in enumerate(joint_names): # 遍歷所有關節
            prefix = ">> " if i == state.manual_ctrl_index else "   " # 選中關節的前綴
            target_val = state.manual_final_ctrl[i] # 目標角度
            actual_val = current_joint_positions[i] # 實際角度
            error = target_val - actual_val # 計算誤差
            line_text = f"{prefix}{name:<15}: Target={target_val:+.2f}, Actual={actual_val:+.2f}, Err={error:+.2f}\n" # 格式化單行文字
            if i < num_joints_per_col: left_col_text += line_text # 加入左列
            else: right_col_text += line_text # 加入右列
        
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left_col_text, None, context) # 繪製左列
        right_col_rect = mujoco.MjrRect(int(viewport.width * 0.40), 0, int(viewport.width * 0.60), viewport.height) # 為右列定義矩形區域
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, right_col_rect, right_col_text, None, context) # 繪製右列

    def render_simulation_overlay(self, viewport, context, state: SimulationState, sim: "Simulation"):
        """渲染正常的模擬除錯資訊。"""
        # 輔助函式，用於格式化向量為字串
        def format_vec(label: str, vec, precision=3, label_width=24):
            if vec is None or vec.size == 0: return f"{label:<{label_width}}None" # 處理空向量
            vec_str = np.array2string(vec, precision=precision, floatmode='fixed', suppress_small=True, threshold=100) # numpy 陣列轉字串
            return f"{label:<{label_width}}{vec_str}" # 回傳格式化後的字串

        # 幫助文字
        help_text = (
            "--- CONTROLS ---\n\n"
            "[Universal]\n"
            "  SPACE: Pause/Play | N: Next Step\n"
            "  F: Float | G: Joint Test/Exit | B: Manual Ctrl\n"
            "  ESC: Exit       | R: Hard Reset  | T: Serial Console\n"
            "  X: Soft Reset   | Y: Regen Infinite | H: Hardware Mode\n"
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
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context) # 在右上角繪製幫助文字
        
        serial_status = "Connected" if state.serial_is_connected else "Disconnected (U to Scan)" # 序列埠連接狀態
        gamepad_status = "Connected" if state.gamepad_is_connected else "Disconnected (J to Scan)" # 遊戲搖桿連接狀態
        terrain_name = state.terrain_manager_ref.get_current_terrain_name(state) if state.terrain_manager_ref else "N/A" # 地形名稱

        policy_text = "" # 初始化策略文字
        pm = state.policy_manager_ref # 獲取策略管理器
        if pm: # 如果存在
            if pm.is_transitioning: # 如果正在切換
                source = pm.source_policy_name
                target = pm.target_policy_name
                alpha_percent = pm.transition_alpha * 100
                policy_text = f"Policy: Blending {source} -> {target} ({alpha_percent:.0f}%)" # 顯示切換進度
            else: # 否則
                policy_text = f"Policy: {pm.primary_policy_name}" # 顯示當前策略

        p = state.tuning_params # 獲取調校參數
        prefixes = ["   "] * 4 # 初始化參數選擇前綴
        prefixes[state.tuning_param_index] = ">> " # 為當前選中的參數加上前綴

        # 組合左上角的資訊文字
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
        if state.control_mode == "FLOATING": # 如果是懸浮模式
            current_height = sim.data.qpos[2] # 獲取當前高度
            target_world_z = state.floating_controller_ref.data.mocap_pos[state.floating_controller_ref.mocap_index][2] # 獲取目標世界Z座標
            # 增加懸浮模式的資訊
            top_left_text += (
                f"\n--- Floating Info ---\n"
                f"{format_vec('Target World Z:', np.array([target_world_z]), 3)}\n"
                f"{format_vec('Current Z:', np.array([current_height]), 3)}\n"
            )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, top_left_text, None, context) # 繪製左上角文字
        
        bottom_left_text = f"--- ONNX INPUTS (Page {state.display_page + 1}/{state.num_display_pages}) ---\n" # ONNX 輸入標題
        onnx_input_vec = state.latest_onnx_input # 獲取最新的 ONNX 輸入
        if onnx_input_vec.size > 0 and self.recipe and state.display_page < len(self.display_pages_content): # 檢查是否有數據可顯示
            current_page_components = self.display_pages_content[state.display_page] # 獲取當前頁面應顯示的元件
            base_obs_dim = sum(self.component_dims.values()) if self.component_dims else 0 # 計算單幀觀察的總維度
            if base_obs_dim > 0: # 如果維度大於0
                history_len = len(onnx_input_vec) // base_obs_dim # 計算歷史幀數
                current_frame_obs = onnx_input_vec[-base_obs_dim:] # 取出最新一幀的觀察數據
                
                current_full_obs_idx = 0 # 初始化索引
                for comp_name_in_recipe in self.recipe: # 遍歷配方中的所有元件
                    dim = self.component_dims.get(comp_name_in_recipe, 0) # 獲取元件維度
                    if dim > 0: # 如果維度大於0
                        if comp_name_in_recipe in current_page_components: # 如果該元件應在當前頁面顯示
                            start_idx, end_idx = current_full_obs_idx, current_full_obs_idx + dim # 計算數據切片索引
                            value_slice = current_frame_obs[start_idx:end_idx] # 切片
                            bottom_left_text += format_vec(f"{comp_name_in_recipe} [{dim}d]:", value_slice, 2) + "\n" # 格式化並加入顯示字串
                        current_full_obs_idx += dim # 更新索引
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, bottom_left_text, None, context) # 繪製左下角文字
        
        torso_lin_vel = sim.data.cvel[sim.torso_id, 3:] # 獲取軀幹世界座標系下的線速度
        torso_ang_vel_local = self._get_local_ang_vel(sim.data, sim.torso_id) # 獲取軀幹局部座標系下的角速度
        # 組合右下角的資訊文字
        bottom_right_text = (
            f"--- ONNX OUTPUTS & STATE ---\n"
            f"{format_vec('Final Action:', state.latest_action_raw)}\n"
            f"{format_vec('Final Ctrl:', state.latest_final_ctrl)}\n\n"
            f"--- Robot State (Sim) ---\n"
            f"{format_vec('Torso Z:', np.array([sim.data.qpos[2]]))}\n"
            f"{format_vec('Lin Vel (World):', torso_lin_vel)}\n"
            f"{format_vec('Ang Vel (Local):', torso_ang_vel_local)}"
        )
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, viewport, bottom_right_text, None, context) # 繪製右下角文字
    
    def _get_local_ang_vel(self, data, torso_id):
        """輔助函式，計算局部角速度用於顯示。"""
        torso_quat = data.xquat[torso_id] # 獲取軀幹的四元數
        norm = np.sum(np.square(torso_quat)) # 計算四元數的模長平方
        if norm < 1e-8: return np.zeros(3) # 如果模長過小，返回零向量
        torso_quat /= np.sqrt(norm) # 標準化四元數
        q_inv = np.array([torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]) / norm # 計算共軛四元數（即逆）
        u, s = q_inv[1:], q_inv[0] # 分解為向量部分和純量部分
        world_ang_vel = data.cvel[torso_id, :3] # 獲取世界座標系下的角速度
        # 使用四元數旋轉公式將世界角速度轉換為局部角速度
        return 2 * np.dot(u, world_ang_vel) * u + (s*s - np.dot(u, u)) * world_ang_vel + 2*s*np.cross(u, world_ang_vel)