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

def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    print("\n--- 機器人模擬控制器 (多輸入模式版) ---")

    config = load_config()
    state = SimulationState(config)
    sim = Simulation(config)
    
    # 初始化控制器並將其引用存入 state
    floating_controller = FloatingController(config, sim.model, sim.data)
    state.floating_controller_ref = floating_controller

    # 初始化輸入處理器
    keyboard_handler = KeyboardInputHandler(state)
    sim.register_callbacks(keyboard_handler)
    xbox_handler = XboxInputHandler(state)
    
    if xbox_handler.is_available():
        state.input_mode = "GAMEPAD"
        print("🎮 搖桿已就緒，預設為搖桿控制模式。按 'M' 鍵切換。")
    else:
        state.input_mode = "KEYBOARD"
        print("⚠️ 未偵測到搖桿，使用鍵盤控制模式。")
    
    try:
        temp_policy = ONNXPolicy(config, 1)
        model_input_dim = temp_policy.model_input_dim
        del temp_policy
    except Exception as e:
        sys.exit(f"❌ 策略初始化失敗: {e}")
    
    if model_input_dim not in config.observation_recipes:
        sys.exit(f"❌ 錯誤: 在 config.yaml 中找不到適用於維度 {model_input_dim} 的觀察配方。")
    recipe = config.observation_recipes[model_input_dim]
    print(f"🔍 找到匹配配方! 使用以下元件建構觀察向量:\n -> {recipe}")

    temp_obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    dummy_obs = temp_obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors))
    base_obs_dim = len(dummy_obs)
    del temp_obs_builder

    policy = ONNXPolicy(config, base_obs_dim)
    obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)

    ALL_OBS_DIMS = {'z_angular_velocity':1, 'gravity_vector':3, 'commands':3, 'joint_positions':12, 'joint_velocities':12, 'foot_contact_states':4, 'linear_velocity':3, 'angular_velocity':3, 'last_action':12, 'phase_signal':1}
    used_dims = {k: ALL_OBS_DIMS[k] for k in recipe if k in ALL_OBS_DIMS}
    overlay = DebugOverlay(recipe, used_dims)

    def reset_all():
        print("\n--- 正在重置模擬 ---")
        sim.reset()
        policy.reset()
        # 重置時確保關閉懸浮模式
        if floating_controller.is_functional:
            sim.data.eq_active[floating_controller.weld_id] = 0
        state.control_mode = "WALKING"
        state.reset_control_state(sim.data.time)
        state.clear_command()

    reset_all()
    print("\n--- 模擬開始 ---")

    while not sim.should_close():
        if state.input_mode == "GAMEPAD":
            xbox_handler.update_state()

        if state.reset_requested:
            reset_all()

        # 每幀都更新機器人姿態到 state，供輸入處理器使用
        state.latest_pos = sim.data.body('torso').xpos.copy()
        state.latest_quat = sim.data.body('torso').xquat.copy()

        sim_time = sim.data.time
        if state.control_timer <= sim_time:
            # 根據控制模式設定UI顯示文字
            if state.control_mode == "WALKING":
                state.sim_mode_text = "ONNX Control"
            else: # FLOATING
                state.sim_mode_text = "Floating (Fixed)"
            
            # 無論何種模式，ONNX策略都在背景運行，以保持其內部狀態(如last_action)的連續性
            base_obs = obs_builder.get_observation(state.command, policy.last_action)
            onnx_input, action_raw = policy.get_action(base_obs)
            
            final_ctrl = sim.default_pose + action_raw * state.tuning_params.action_scale
            sim.apply_control(final_ctrl, state.tuning_params)

            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_raw
            state.latest_final_ctrl = final_ctrl
            
            state.control_timer += config.control_dt

        sim.step(state)
        sim.render(state, overlay)

    sim.close()
    xbox_handler.close()
    print("\n模擬結束，程式退出。")

if __name__ == "__main__":
    main()