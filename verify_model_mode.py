# verify_model_mode.py
import numpy as np
import mujoco
import sys
import time
from pathlib import Path

# --- 導入您專案的模組 ---
from config import load_config
from policy import ONNXPolicy
from observation import ObservationBuilder # 我們將使用您修改後的版本

# --- 腳本設定 ---
SIMULATION_DURATION = 3.0
PERTURBATION_VALUE = 0.3
STABILITY_THRESHOLD = 0.05
HIP_JOINT_INDICES = [1, 4, 7, 10]

def run_simulation(model, data, policy, obs_builder, duration):
    """
    運行一個模擬片段並收集最後的 Raw Action 數據。
    這個版本假設是「絕對角度模式」。
    """
    # 在這個測試腳本中，我們直接使用一個固定的PD增益
    model.actuator_gainprm[:, 0] = 5.0
    model.dof_damping[6:] = 0.5
    
    start_time = data.time
    recent_actions = []

    # 簡單的熱身
    warmup_duration = 1.0
    while data.time - start_time < warmup_duration:
        base_obs = obs_builder.get_observation(np.zeros(3), policy.last_action)
        _, action_raw = policy.get_action(base_obs)
        # 【核心】使用絕對角度模式計算控制指令
        final_ctrl = action_raw * 1.0 # action_scale 設為 1.0
        data.ctrl[:] = final_ctrl
        mujoco.mj_step(model, data)

    # 真正開始收集數據
    collection_start_time = data.time
    while data.time - collection_start_time < (duration - warmup_duration):
        base_obs = obs_builder.get_observation(np.zeros(3), policy.last_action)
        _, action_raw = policy.get_action(base_obs)
        recent_actions.append(action_raw.copy())
        
        final_ctrl = action_raw * 1.0
        data.ctrl[:] = final_ctrl
        mujoco.mj_step(model, data)

    if not recent_actions:
        print("❌ 錯誤：未能收集到任何 action 數據。")
        return None
        
    return np.mean(recent_actions, axis=0)

def reset_to_key(model, data, key_id, perturbation=None):
    """
    手動將模擬重置到指定的 keyframe，並可選擇性地施加擾動。
    """
    mujoco.mj_resetData(model, data)
    qpos = model.key_qpos[key_id].copy()
    if perturbation is not None:
        qpos[7:] += perturbation
    data.qpos[:] = qpos
    data.qvel[:] = model.key_qvel[key_id]
    mujoco.mj_forward(model, data)


def verify():
    """執行驗證的主函式。"""
    print("=" * 60)
    print("🤖 模型輸出模式驗證工具 (絕對角度模式驗證版) 🤖")
    print("=" * 60)

    try:
        print("1. 載入設定與模型...")
        config = load_config()
        if not Path(config.mujoco_model_file).exists():
            print(f"❌ 錯誤：找不到模型檔案 '{config.mujoco_model_file}'")
            return

        model = mujoco.MjModel.from_xml_path(config.mujoco_model_file)
        data = mujoco.MjData(model)
        
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id == -1:
            print("❌ 錯誤：在 XML 中找不到名為 'home' 的 keyframe。")
            return
        
        default_pose_from_key = model.key_qpos[home_key_id][7:].copy()
        
        # 確保使用與模型匹配的觀察配方 (假設為48維)
        obs_dim = 48
        if obs_dim not in config.observation_recipes:
            print(f"❌ 錯誤: config.yaml 中缺少維度為 {obs_dim} 的 observation_recipes。")
            return
        recipe = config.observation_recipes[obs_dim]
             
        # 【核心】我們在這裡實例化的 obs_builder 會使用您修改後的 absolute mode 版本
        obs_builder = ObservationBuilder(recipe, data, model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso'), default_pose_from_key, config)
        base_obs_dim = len(obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors)))
        
        policy_config = config
        policy_config.initial_tuning_params.action_scale = 1.0 # 測試時固定為1.0
        policy = ONNXPolicy(policy_config, base_obs_dim)
        print("✅ 資源載入成功！")
        print("-" * 60)

        # --- 實驗一：基準測試 (Baseline Test) ---
        print("2. 執行【實驗一：基準測試】")
        print("   - 從標準的 'home' 姿態開始。")
        
        reset_to_key(model, data, home_key_id)
        policy.reset()
        
        stable_action_base = run_simulation(model, data, policy, obs_builder, SIMULATION_DURATION)
        if stable_action_base is None: return

        hip_action_base = np.mean(stable_action_base[HIP_JOINT_INDICES])
        print(f"   📊 基準穩定後 Raw Action (髖關節平均值): {hip_action_base:.4f}")
        print("-" * 60)
        time.sleep(1)

        # --- 實驗二：擾動初始姿態測試 (Perturbation Test) ---
        print("3. 執行【實驗二：擾動測試】")
        print(f"   - 從一個被擾動過的初始姿態開始 (髖關節增加 {PERTURBATION_VALUE})。")
        
        perturbation_vector = np.zeros(12)
        perturbation_vector[HIP_JOINT_INDICES] = PERTURBATION_VALUE
        reset_to_key(model, data, home_key_id, perturbation=perturbation_vector)
        policy.reset()

        stable_action_perturbed = run_simulation(model, data, policy, obs_builder, SIMULATION_DURATION)
        if stable_action_perturbed is None: return
        
        hip_action_perturbed = np.mean(stable_action_perturbed[HIP_JOINT_INDICES])
        print(f"   📊 擾動穩定後 Raw Action (髖關節平均值): {hip_action_perturbed:.4f}")
        print("-" * 60)

        # --- 4. 分析與結論 ---
        print("4. 分析結果與結論...")
        
        # 在絕對角度模式下，兩個實驗的輸出應該幾乎相同
        diff = abs(hip_action_perturbed - hip_action_base)
        print(f"   - 兩個實驗的 Raw Action 穩定值之差: {diff:.4f}")
        print("-" * 60)

        if diff < STABILITY_THRESHOLD:
            print("✅ 【結論】驗證成功！模型的行為與【絕對角度模式 (Absolute-based)】的預期相符。")
            print("   無論從哪個初始姿態開始，模型都能收斂到幾乎相同的目標角度輸出。")
            print("   您在 main.py 和 observation.py 中的絕對角度模式修改是【正確的】。")
        else:
            print("❌ 【結論】驗證失敗！模型的行為與【絕對角度模式】的預期不符。")
            print("   模型的輸出會因為初始姿態的不同而產生巨大差異，這不符合絕對角度模型的特徵。")
            print("   這可能意味著模型實際上是「偏移量模式」，或者模型本身不夠穩定。")
            print("   建議與模型作者確認訓練時的觀察空間和動作空間定義。")
        
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 驗證過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()