# test/verify_model_mode.py

# 【v4.3.3 修改】 導入必要的模組 (保留此註解)
import os
import sys
import numpy as np
import onnxruntime as ort
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# 【v4.3.4 修正】 導入 deque
from collections import deque
# ✅ 新增：共用 ORT 工廠（自動選 TRT→CUDA→DML→CPU）
from src.core.ort_provider import create_session, diag_print

# 【v4.3.3 修改】 調整 sys.path 以便能夠導入 src 下的模組 (保留此註解)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 【v4.3.4 修正】 導入 load_config
from src.core.config import load_config
# 【v4.3.3 修改】 導入 PolicyManager 和 ObservationManager (保留此註解)
from src.hardware.policy import PolicyManager
from src.simulation.observation_manager import ObservationManager

# 【v4.3.3 新增】 為了在獨立測試中運行 ObservationManager，需要一個最小化的 MockState 和 MockSim (保留此註解)
@dataclass
class _MockSim:
    """_MockSim: 最小化的模擬器 Mock，只提供 ObservationManager 需要的屬性。"""
    default_pose: np.ndarray = field(default_factory=lambda: np.zeros(12))
    torso_id: int = 1
    accelerometer_id: int = -1 # 假設存在 accelerometer 傳感器

@dataclass
class _MockState:
    """_MockState: 最小化的 SimulationState Mock，只提供 ObservationManager 需要的屬性。"""
    # 【v4.3.4 修正】 config 應為 AppConfig 類型
    config: Any # 這裡將接收 AppConfig 物件
    sim: _MockSim = field(default_factory=_MockSim) # 包含一個 MockSim
    
    # 【v4.3.3 新增】 提供 ObservationManager 所需的 raw 數據屬性，全部初始化為零向量 (保留此註解)
    raw_torso_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    raw_torso_linear_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_torso_angular_velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(12))
    raw_accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_last_action: np.ndarray = field(default_factory=lambda: np.zeros(12))
    command: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 【v4.3.3 新增】 為了 PolicyManager 初始化 PolicyManager.reset() 中對它的調用 (保留此註解)
    observation_manager_ref: Optional[ObservationManager] = None 


# 【v4.3.3 修改】 verify_model 函式 (保留此註解)
def verify_model(model_name: str, config_path: str = "config.yaml"):
    """
    【v4.3.3 修改】
    驗證指定的 ONNX 模型是否能正確載入並進行推論。
    此函式已重構為使用 ObservationManager。
    """
    try:
        # 【v4.3.4 修正】 使用 load_config 載入 AppConfig 物件
        app_config = load_config(config_path)

        if model_name not in app_config.onnx_models: # 【v4.3.4 修正】 訪問 app_config.onnx_models
            print(f"❌ 錯誤: 配置檔案中找不到模型 '{model_name}'。")
            print(f"可用的模型: {list(app_config.onnx_models.keys())}")
            return False

        model_info = app_config.onnx_models[model_name] # 【v4.3.4 修正】 訪問 app_config.onnx_models
        model_path = model_info['path']
        observation_recipe = model_info['observation_recipe']

        print(f"\n--- 正在驗證模型: '{model_name}' ---")
        print(f"模型路徑: {model_path}")
        print(f"觀測配方: {observation_recipe}")

        if not os.path.exists(model_path):
            print(f"❌ 錯誤: 模型檔案不存在: {model_path}")
            return False

        # 【v4.3.3 修改】 準備 Mock 環境 (保留此註解)
        # 【v4.3.4 修正】 將實際的 app_config 物件傳遞給 _MockState
        mock_sim = _MockSim()
        mock_state = _MockState(config=app_config, sim=mock_sim) # 【v4.3.4 修正】 傳入 app_config
        
        # 【v4.3.3 新增】 實例化 ObservationManager (保留此註解)
        observation_manager = ObservationManager(mock_state)
        mock_state.observation_manager_ref = observation_manager # 將參考注入 mock_state

        # 【v4.3.3 新增】 實例化 PolicyManager (需要 PolicyManager 才能計算 base_obs_dim 和管理歷史) (保留此註解)
        # 這裡 PolicyManager 只是用於測試觀測生成和基本推論，所以 overlay 設為 None
        # 【v4.3.4 修正】 將實際的 app_config 物件傳遞給 PolicyManager
        policy_manager = PolicyManager(app_config, observation_manager, None) # 【v4.3.4 修正】 傳入 app_config

        # 【v4.3.3 修改】 設置 ObservationManager 的配方 (保留此註解)
        observation_manager.set_recipe(observation_recipe)

        # 【v4.3.3 修改】 從 ObservationManager 獲取觀測數據 (保留此註解)
        # 這裡的觀測數據會全部是零，因為 MockState 的 raw 數據都是零
        dummy_base_obs = observation_manager.get_observation()

        if dummy_base_obs.size == 0:
            print("❌ 錯誤: 生成的基礎觀測向量為空。請檢查觀測配方和 ALL_OBS_DIMS 配置。")
            return False

        # 推斷模型輸入維度和歷史長度（這部分現在主要由 PolicyManager 處理，但我們可以重新驗證）
        base_obs_dim = dummy_base_obs.shape[0]
        
        sess_options = ort.SessionOptions()
        # ✅ 改用共用工廠：自動挑選最適合的 EP（Jetson 用 TensorRT/CUDA；Windows 用 CUDA/DML；否則 CPU）
        session = create_session(model_path, sess_options=sess_options)
                
        model_input_dim = session.get_inputs()[0].shape[1]
        history_len = 1
        if base_obs_dim > 0 and model_input_dim % base_obs_dim == 0:
            history_len = model_input_dim // base_obs_dim
        else:
            print(f"❌ 錯誤: 模型輸入維度 ({model_input_dim}) 與基礎觀測維度 ({base_obs_dim}) 不兼容。")
            return False

        print(f"推斷基礎觀測維度: {base_obs_dim}")
        print(f"模型輸入總維度: {model_input_dim}")
        print(f"模型歷史長度: {history_len}")
        
        # 準備一個符合模型歷史長度的零填充觀測歷史
        obs_history_deque = deque(
            [np.zeros(base_obs_dim, dtype=np.float32)] * history_len,
            maxlen=history_len
        )
        
        # 將最新的零觀測加入歷史
        obs_history_deque.append(dummy_base_obs)
        
        # 拼接成最終的 ONNX 輸入
        onnx_input = np.concatenate(list(obs_history_deque)).astype(np.float32).reshape(1, -1)

        if onnx_input.shape[1] != model_input_dim:
            print(f"❌ 錯誤: 最終 ONNX 輸入維度 ({onnx_input.shape[1]}) 與模型預期 ({model_input_dim}) 不符。")
            return False

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # 執行推論
        _ = session.run([output_name], {input_name: onnx_input})

        print(f"✅ 模型 '{model_name}' 載入成功，並通過推論測試。")
        return True

    except Exception as e:
        print(f"❌ 驗證模型 '{model_name}' 時發生錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印完整堆棧信息
        return False

if __name__ == "__main__":
    # 【v4.3.3 修改】 測試時，使用 config.yaml 中的第一個模型作為默認測試對象 (保留此註解)
    # 或者您可以手動指定一個模型名稱，例如: verify_model("agile_model") (保留此註解)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    os.chdir(project_root) # 確保腳本在專案根目錄運行

    config_file = "config.yaml"
    
    # 【v4.3.4 修正】 載入 config 以獲取模型列表 (使用 load_config)
    try:
        # 使用 load_config 獲取 AppConfig 物件
        app_config_obj = load_config(config_file)
        all_models = list(app_config_obj.onnx_models.keys()) # 從 AppConfig 物件中訪問 onnx_models 屬性
        if not all_models:
            print("未在 config.yaml 中找到任何模型可供驗證。")
            sys.exit(1)
        
        default_test_model = all_models[0] # 默認測試第一個模型
        
        # 讓使用者選擇或使用默認模型
        print("\n--- 模型驗證工具 ---")
        print(f"默認驗證模型: '{default_test_model}'")
        chosen_model = input("輸入要驗證的模型名稱 (留空使用默認): ").strip()
        if not chosen_model:
            chosen_model = default_test_model
            
        if chosen_model not in all_models:
            print(f"❌ 錯誤: 模型 '{chosen_model}' 不在 config.yaml 中。")
            sys.exit(1)

    except FileNotFoundError:
        print(f"錯誤: 未找到 config.yaml 檔案。請確保在正確的目錄中運行。")
        sys.exit(1)
    except Exception as e:
        print(f"載入 config.yaml 時發生錯誤: {e}")
        sys.exit(1)

    if verify_model(chosen_model, config_file):
        print("\n所有模型驗證完成，無錯誤。")
    else:
        print("\n模型驗證失敗。請檢查日誌了解詳情。")
        sys.exit(1)