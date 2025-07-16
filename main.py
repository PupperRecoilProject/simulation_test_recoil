# main.py
import sys
import numpy as np
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
            policy = ONNXPolicy(config, base_obs_dim)
        else:
            sys.exit(f"❌ 致命錯誤: 模型期望維度 ({policy.model_input_dim}) 與配方產生的觀察維度 ({base_obs_dim}) 不符，且找不到匹配配方！")

    ALL_OBS_DIMS = {'z_angular_velocity':1, 'gravity_vector':3, 'commands':3, 'joint_positions':12, 'joint_velocities':12, 'foot_contact_states':4, 'linear_velocity':3, 'angular_velocity':3, 'last_action':12, 'phase_signal':1}
    used_dims = {k: ALL_OBS_DIMS[k] for k in recipe if k in ALL_OBS_DIMS}
    overlay = DebugOverlay(recipe, used_dims)

    def reset_all():
        print("\n--- 正在重置模擬 ---")
        sim.reset()
        policy.reset()
        if state.control_mode == "FLOATING":
            state.set_control_mode("WALKING")
        elif state.control_mode == "JOINT_TEST":
            state.set_control_mode("WALKING")
        state.reset_control_state(sim.data.time)
        state.clear_command()

    reset_all()
    print("\n--- 模擬開始 (F: 懸浮, G: 關節測試, T: 序列埠, M: 輸入模式) ---")

    while not sim.should_close():
        if state.input_mode == "GAMEPAD": xbox_handler.update_state()
        if state.reset_requested: reset_all()

        state.latest_pos = sim.data.body('torso').xpos.copy()
        state.latest_quat = sim.data.body('torso').xquat.copy()
        if serial_comm.is_connected: state.serial_latest_messages = serial_comm.get_latest_messages()
        if state.serial_command_to_send:
            serial_comm.send_command(state.serial_command_to_send)
            state.serial_command_to_send = ""

        if state.control_mode != "SERIAL_MODE":
            sim_time = sim.data.time
            if state.control_timer <= sim_time:
                final_ctrl = np.zeros(config.num_motors)
                
                if state.control_mode == "WALKING" or state.control_mode == "FLOATING":
                    state.sim_mode_text = state.control_mode
                    base_obs = obs_builder.get_observation(state.command, policy.last_action)
                    onnx_input, action_raw = policy.get_action(base_obs)
                    final_ctrl = sim.default_pose + action_raw * state.tuning_params.action_scale
                    state.latest_onnx_input = onnx_input.flatten()
                    state.latest_action_raw = action_raw
                
                elif state.control_mode == "JOINT_TEST":
                    state.sim_mode_text = "Joint Test"
                    final_ctrl = sim.default_pose + state.joint_test_offsets
                
                sim.apply_control(final_ctrl, state.tuning_params)
                state.latest_final_ctrl = final_ctrl
                state.control_timer += config.control_dt

            sim.step(state)
        
        sim.render(state, overlay)

    sim.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n模擬結束，程式退出。")

if __name__ == "__main__":
    main()