# main.py
import sys
import numpy as np
import mujoco

from config import load_config
from state import SimulationState
from simulation import Simulation
from policy import ONNXPolicy
from observation import ObservationBuilder
from rendering import DebugOverlay
from keyboard_input_handler import KeyboardInputHandler
from xbox_input_handler import XboxInputHandler
from floating_controller import FloatingController
from serial_communicator import SerialCommunicator

def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    from xbox_controller import XboxController 
    print("\n--- 機器人模擬控制器 (多輸入模式版) ---")
    
    config = load_config()
    state = SimulationState(config)
    sim = Simulation(config)
    
    floating_controller = FloatingController(config, sim.model, sim.data)
    state.floating_controller_ref = floating_controller
    serial_comm = SerialCommunicator()
    
    keyboard_handler = KeyboardInputHandler(state)
    keyboard_handler.register_callbacks(sim.window)
    xbox_handler = XboxInputHandler(state)
    if xbox_handler.is_available():
        state.toggle_input_mode("GAMEPAD")
    
    try:
        assumed_dim = next(iter(config.observation_recipes))
        recipe = config.observation_recipes[assumed_dim]
    except StopIteration:
        sys.exit("❌ 錯誤: 在 config.yaml 中沒有定義任何 observation_recipes。")

    obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    base_obs_dim = len(obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors)))
    policy = ONNXPolicy(config, base_obs_dim)
    
    if policy.model_input_dim != base_obs_dim:
        if policy.model_input_dim in config.observation_recipes:
            print(f"⚠️ 維度不匹配，自動切換到維度 {policy.model_input_dim} 的正確配方...")
            recipe = config.observation_recipes[policy.model_input_dim]
            obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)
            base_obs_dim = len(obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors)))
        else:
            sys.exit(f"❌ 致命錯誤: 模型期望維度 ({policy.model_input_dim}) 與配方產生的觀察維度 ({base_obs_dim}) 不符，且找不到匹配配方！")

    ALL_OBS_DIMS = {'z_angular_velocity':1, 'gravity_vector':3, 'commands':3, 'joint_positions':12, 'joint_velocities':12, 'foot_contact_states':4, 'linear_velocity':3, 'angular_velocity':3, 'last_action':12, 'phase_signal':1}
    used_dims = {k: ALL_OBS_DIMS[k] for k in recipe if k in ALL_OBS_DIMS}
    overlay = DebugOverlay(recipe, used_dims)

    def hard_reset():
        """完全重置整個模擬環境。"""
        print("\n--- 正在執行完全重置 (Hard Reset) ---")
        sim.reset()
        policy.reset()
        if state.control_mode == "FLOATING":
            state.set_control_mode("WALKING")
        state.reset_control_state(sim.data.time)
        state.clear_command()
        
        # =========================================================================
        # === 【新增】在重置時，清理所有模式特定的狀態，增強穩健性              ===
        # =========================================================================
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        # =========================================================================
        
        state.hard_reset_requested = False

    def soft_reset():
        """僅重置機器人姿態和控制器狀態，不重置模擬時間和物理世界。"""
        print("\n--- 正在執行空中姿態重置 (Soft Reset) ---")
        sim.data.qpos[7:] = sim.default_pose
        sim.data.qvel[6:] = 0
        policy.reset()
        state.clear_command()

        # =========================================================================
        # === 【新增】在重置時，清理所有模式特定的狀態，增強穩健性              ===
        # =========================================================================
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        # =========================================================================

        mujoco.mj_forward(sim.model, sim.data)
        state.soft_reset_requested = False

    hard_reset()
    print("\n--- 模擬開始 (SPACE: 暫停, N:下一步) ---")
    print("    (F: 懸浮, G: 關節測試, B: 手動控制, T: 序列埠, M: 輸入模式, R: 硬重置, X: 軟重置)")

    state.execute_one_step = False

    while not sim.should_close():
        if state.single_step_mode and not state.execute_one_step:
            sim.render(state, overlay)
            continue
        if state.execute_one_step:
            state.execute_one_step = False

        if state.input_mode == "GAMEPAD": xbox_handler.update_state()
        
        # --- 修改：區分兩種重置 ---
        if state.hard_reset_requested: hard_reset()
        if state.soft_reset_requested: soft_reset()

        state.latest_pos = sim.data.body('torso').xpos.copy()
        state.latest_quat = sim.data.body('torso').xquat.copy()
        if serial_comm.is_connected: state.serial_latest_messages = serial_comm.get_latest_messages()
        if state.serial_command_to_send:
            serial_comm.send_command(state.serial_command_to_send)
            state.serial_command_to_send = ""

        if state.control_mode != "SERIAL_MODE":
            if state.single_step_mode:
                print("\n" + "="*20 + f" STEP AT TIME {sim.data.time:.4f} " + "="*20)

            base_obs = obs_builder.get_observation(state.command, policy.last_action)
            if state.single_step_mode:
                current_joint_angles = sim.data.qpos[7:]
                current_joint_positions = current_joint_angles - sim.default_pose
                print(f"1. [OBSERVED] joint_positions: {np.array2string(current_joint_positions, precision=3, suppress_small=True)}")

            onnx_input, action_raw = policy.get_action(base_obs)
            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_raw
            if state.single_step_mode:
                 print(f"2. [AI DECISION] Raw Action:      {np.array2string(action_raw, precision=3, suppress_small=True)}")

            # =========================================================================
            # === 【核心修復】為 MANUAL_CTRL 模式增加專門的邏輯分支                  ===
            # =========================================================================
            if state.control_mode == "MANUAL_CTRL":
                state.sim_mode_text = "Manual Ctrl"
                # 在手動模式下，直接使用 state.manual_final_ctrl 作為目標
                final_ctrl = state.manual_final_ctrl.copy() 
            elif state.control_mode == "JOINT_TEST":
                state.sim_mode_text = "Joint Test"
                final_ctrl = sim.default_pose + state.joint_test_offsets
            else: # 預設情況，包含 "WALKING" 和 "FLOATING" 模式
                state.sim_mode_text = state.control_mode
                final_ctrl = sim.default_pose + action_raw * state.tuning_params.action_scale
            # =========================================================================

            state.latest_final_ctrl = final_ctrl
            if state.single_step_mode:
                print(f"3. [COMMAND] Final Ctrl:          {np.array2string(final_ctrl, precision=3, suppress_small=True)}")

            sim.apply_position_control(final_ctrl, state.tuning_params)
            
            target_time = sim.data.time + config.control_dt
            while sim.data.time < target_time:
                mujoco.mj_step(sim.model, sim.data)

            if state.single_step_mode:
                next_joint_angles = sim.data.qpos[7:]
                print(f"4. [RESULT] Next actual angles: {np.array2string(next_joint_angles, precision=3, suppress_small=True)}")
        
        sim.render(state, overlay)

    sim.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n模擬結束，程式退出。")

if __name__ == "__main__":
    main()