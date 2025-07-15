import sys
import numpy as np
from config import load_config
from state import SimulationState
from simulation import Simulation
from input_handler import InputHandler
from policy import ONNXPolicy
from observation import ObservationBuilder
from rendering import DebugOverlay

def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    print("\n--- 機器人模擬控制器 (重構版) ---")

    # 1. 載入設定
    config = load_config()

    # 2. 初始化核心組件
    state = SimulationState(config)
    sim = Simulation(config)
    input_handler = InputHandler(state)
    sim.register_callbacks(input_handler)
    
    # 3. 智慧觀察與策略初始化
    # 3.1 確定觀察配方
    #    (注意: ONNXPolicy 在內部會讀取模型維度，但我們需要先找到對應的配方來計算 base_obs_dim)
    temp_policy = ONNXPolicy(config, 1) # 暫時用 base_obs_dim=1 初始化來讀取模型
    model_input_dim = temp_policy.model_input_dim
    del temp_policy # 刪除臨時實例

    if model_input_dim not in config.observation_recipes:
        sys.exit(f"❌ 錯誤: 在 config.yaml 中找不到適用於維度 {model_input_dim} 的觀察配方。")
    
    recipe = config.observation_recipes[model_input_dim]
    print(f"🔍 找到匹配配方! 使用以下元件建構觀察向量:\n -> {recipe}")

    # 3.2 計算基礎觀察維度
    #     我們需要一個 "虛擬" 的 ObservationBuilder 來計算維度
    temp_obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    dummy_obs = temp_obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors))
    base_obs_dim = len(dummy_obs)
    del temp_obs_builder

    # 3.3 正式初始化策略和觀察建構器
    policy = ONNXPolicy(config, base_obs_dim)
    obs_builder = ObservationBuilder(recipe, sim.data, sim.model, sim.torso_id, sim.default_pose, config)

    # 4. 初始化渲染器
    #    計算配方中各元件的維度，供 DebugOverlay 使用
    #    這是所有可能作為觀察元件的名稱及其預期維度
    ALL_OBS_COMPONENT_DIMS = { 
        'z_angular_velocity': 1, 'gravity_vector': 3, 'commands': 3,
        'joint_positions': 12, 'joint_velocities': 12, 'foot_contact_states': 4,
        'linear_velocity': 3, 'angular_velocity': 3,
        'last_action': 12, 'phase_signal': 1
    }
    # 只取出當前配方中實際使用的元件及其維度
    used_recipe_dims = {key: ALL_OBS_COMPONENT_DIMS[key] for key in recipe if key in ALL_OBS_COMPONENT_DIMS}
    overlay = DebugOverlay(recipe, used_recipe_dims)

    # 5. 定義重置函式
    def reset_all():
        """重置所有相關的模擬和控制狀態 (不包含清除指令)。"""
        print("\n--- 正在重置模擬 ---")
        sim.reset() # 重置 MuJoCo 物理世界
        policy.reset() # 重置 ONNX 策略的內部狀態 (歷史觀察、上次動作)
        state.reset_control_state(sim.data.time) # 重置控制計時器和請求狀態
        # 注意: 此處不再包含 state.command.fill(0.0)

    # 首次運行時執行重置
    reset_all()
    print("\n--- 模擬開始 ---")

    # 6. 主模擬迴圈
    while not sim.should_close():
        
        if state.reset_requested:
            reset_all()

        sim_time = sim.data.time
        
        # 控制邏輯 (僅在達到控制頻率時執行)
        if state.control_timer <= sim_time:
            # a. 判斷模式
            if sim_time < config.warmup_duration:
                state.mode_text = "Warmup"
                action_raw = np.zeros(config.num_motors)
                onnx_input = np.array([])
            else:
                state.mode_text = "ONNX Control"
                # b. 產生觀察
                base_obs = obs_builder.get_observation(state.command, policy.last_action)
                # c. 獲取動作
                onnx_input, action_raw = policy.get_action(base_obs)
            
            # d. 應用控制
            final_ctrl = sim.default_pose + action_raw * state.tuning_params.action_scale
            sim.apply_control(final_ctrl, state.tuning_params)

            # e. 更新狀態以供渲染
            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_raw
            state.latest_final_ctrl = final_ctrl
            
            # f. 更新下一個控制時間點
            state.control_timer += config.control_dt

        # 物理步進
        sim.step(state)

        # 渲染
        sim.render(state, overlay)

    # 7. 清理
    sim.close()
    print("\n模擬結束，程式退出。")

if __name__ == "__main__":
    main()