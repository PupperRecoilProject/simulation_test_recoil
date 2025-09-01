# config.py
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any

# 【v4.10.0 新增】為地形生成參數建立一個新的 dataclass
@dataclass
class TerrainGenerationConfig:
    """地形生成器的參數。"""
    sine_waves_amplitude: float
    steps_height: float
    random_noise_amplitude: float
    pyramid_max_height: float
    stepped_pyramid_max_height: float
    stepped_pyramid_steps_min: int
    stepped_pyramid_steps_max: int

@dataclass
class TuningParamsConfig:
    """從設定檔載入的初始調校參數資料類別。"""
    kp: float
    kd: float
    action_scale: float
    bias: float

@dataclass
class FloatingControllerConfig:
    """懸浮控制器的設定。"""
    target_height: float
    kp_vertical: float
    kd_vertical: float
    kp_attitude: float
    kd_attitude: float

@dataclass
class AppConfig:
    """儲存所有應用程式設定的資料類別。"""
    mujoco_model_file: str
    
    # 【修改】使用新的 onnx_models 結構，可以包含路徑和配方
    onnx_models: Dict[str, Dict[str, Any]]
    policy_transition_duration: float
    
    num_motors: int
    physics_timestep: float
    control_freq: float
    control_dt: float
    warmup_duration: float
    command_scaling_factors: List[float]
    
    keyboard_velocity_adjust_step: float
    gamepad_sensitivity: Dict[str, float]
    param_adjust_steps: Dict[str, float]

    initial_tuning_params: TuningParamsConfig
    floating_controller: FloatingControllerConfig
    # 【v4.10.0 新增】將地形設定加入到主設定物件中
    terrain_generation: TerrainGenerationConfig


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

    tuning_params = TuningParamsConfig(**config_data['initial_tuning_params'])
    floating_config = FloatingControllerConfig(**config_data['floating_controller'])
    # 【v4.10.0 新增】載入地形生成設定
    terrain_gen_config = TerrainGenerationConfig(**config_data['terrain_generation'])
    
    config_obj = AppConfig(
        mujoco_model_file=config_data['mujoco_model_file'],
        
        onnx_models=config_data['onnx_models'],
        policy_transition_duration=config_data.get('policy_transition_duration', 0.5),
                
        num_motors=config_data['num_motors'],
        physics_timestep=config_data['physics_timestep'],
        control_freq=config_data['control_freq'],
        control_dt=1.0 / config_data['control_freq'],
        warmup_duration=config_data['warmup_duration'],
        command_scaling_factors=config_data['command_scaling_factors'],
        
        keyboard_velocity_adjust_step=config_data['keyboard_velocity_adjust_step'],
        gamepad_sensitivity=config_data['gamepad_sensitivity'],
        param_adjust_steps=config_data['param_adjust_steps'],
        
        initial_tuning_params=tuning_params,
        floating_controller=floating_config,
        # 【v4.10.0 新增】傳入地形設定
        terrain_generation=terrain_gen_config
    )
    
    print("✅ 設定檔載入成功 (包含懸浮控制器設定)。")
    return config_obj
