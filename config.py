import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class TuningParamsConfig:
    """從設定檔載入的初始調校參數資料類別。"""
    kp: float
    kd: float
    action_scale: float
    bias: float

@dataclass
class AppConfig:
    """儲存所有應用程式設定的資料類別。"""
    mujoco_model_file: str
    onnx_model_path: str
    num_motors: int
    physics_timestep: float
    control_freq: float
    control_dt: float
    warmup_duration: float
    velocity_adjust_step: float
    command_scaling_factors: List[float]
    initial_tuning_params: TuningParamsConfig
    observation_recipes: Dict[int, List[str]]

def load_config(path: str = "config.yaml") -> AppConfig:
    """
    從 YAML 檔案載入設定並回傳一個 AppConfig 物件。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"設定檔 '{path}' 不存在。請確保檔案路徑正確。")
    except Exception as e:
        raise IOError(f"讀取或解析設定檔 '{path}' 時發生錯誤: {e}")

    # 將字典轉換為 dataclass 物件，提供更好的型別提示和屬性存取
    tuning_params = TuningParamsConfig(**config_data['initial_tuning_params'])
    
    config_obj = AppConfig(
        mujoco_model_file=config_data['mujoco_model_file'],
        onnx_model_path=config_data['onnx_model_path'],
        num_motors=config_data['num_motors'],
        physics_timestep=config_data['physics_timestep'],
        control_freq=config_data['control_freq'],
        control_dt=1.0 / config_data['control_freq'],
        warmup_duration=config_data['warmup_duration'],
        velocity_adjust_step=config_data['velocity_adjust_step'],
        command_scaling_factors=config_data['command_scaling_factors'],
        initial_tuning_params=tuning_params,
        observation_recipes=config_data['observation_recipes']
    )
    
    print("✅ 設定檔載入成功。")
    return config_obj

# 允許直接執行此檔案來測試設定檔是否能被正確載入和解析
if __name__ == '__main__':
    config = load_config()
    print("--- 載入的設定 ---")
    print(config)
    print("\n存取範例:")
    print(f"  模型路徑: {config.onnx_model_path}")
    print(f"  初始 Kp: {config.initial_tuning_params.kp}")
    print(f"  48維配方: {config.observation_recipes[48]}")