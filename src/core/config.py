# src/core/config.py

import yaml  # 引入 PyYAML 函式庫，用於解析 YAML 格式的設定檔
from dataclasses import dataclass, field  # 引入 dataclasses，方便快速建立資料類別
from typing import Dict, List, Any  # 引入類型提示，增強程式碼可讀性與健壯性


@dataclass
class TuningParamsConfig:
    """從設定檔載入的初始調校參數資料類別。"""

    # 此類別對應 config.yaml 中的 initial_tuning_params 區塊
    kp: float  # PD 控制器中的比例增益 (Proportional gain)，代表剛度
    kd: float  # PD 控制器中的微分增益 (Derivative gain)，代表阻尼
    action_scale: float  # AI 模型輸出的動作縮放比例
    bias: float  # 施加到馬達的額外力矩偏置


@dataclass
class FloatingControllerConfig:
    """懸浮控制器的設定。"""

    # 此類別對應 config.yaml 中的 floating_controller 區塊
    target_height: float  # 機器人懸浮時，身體距離地面的目標高度 (單位：公尺)
    kp_vertical: float  # 垂直方向的 PD 控制器 - P 增益
    kd_vertical: float  # 垂直方向的 PD 控制器 - D 增益
    kp_attitude: float  # 姿態（滾轉/俯仰）的 PD 控制器 - P 增益
    kd_attitude: float  # 姿態（滾轉/俯仰）的 PD 控制器 - D 增益


@dataclass
class AppConfig:
    """
    儲存所有應用程式設定的資料類別。
    這個類別作為一個集中的設定容器，將從 YAML 檔案讀取的設定值結構化。
    """

    # --- 開發與除錯設定 ---
    use_virtual_teensy: bool  # 是否啟用虛擬Teensy，用於無硬體測試

    # --- 檔案路徑 ---
    mujoco_model_file: str  # MuJoCo 主要場景的 XML 檔案路徑

    # 【修改】使用新的 onnx_models 結構，可以包含路徑和配方
    # 一個字典，儲存所有可用的 ONNX 模型資訊，鍵為模型暱稱
    onnx_models: Dict[str, Dict[str, Any]]

    # --- 模擬與控制參數 ---
    policy_transition_duration: float  # AI 策略模型之間平滑切換所需的秒數
    num_motors: int  # 機器人的馬達（致動器）數量
    physics_timestep: float  # MuJoCo 物理引擎的模擬時間步長 (dt)
    control_freq: float  # 控制迴圈的頻率 (Hz)，即每秒執行控制邏輯的次數
    control_dt: float  # 控制迴圈的時間間隔 (秒)，由 1.0 / control_freq 計算而來
    warmup_duration: float  # (已棄用) 預留的暖機時間
    command_scaling_factors: List[
        float
    ]  # 將使用者輸入指令 (如搖桿) 縮放到 ONNX 模型期望範圍的係數

    # --- 輸入處理參數 ---
    keyboard_velocity_adjust_step: float  # 使用鍵盤控制時，每按一下增加/減少的速度值
    gamepad_sensitivity: Dict[str, float]  # 遊戲搖桿各個軸的靈敏度設定
    param_adjust_steps: Dict[str, float]  # 在 UI 或透過搖桿調整參數時，每按一下的步進值

    # --- 結構化設定 ---
    initial_tuning_params: TuningParamsConfig  # 初始 PD 控制參數的實例
    floating_controller: FloatingControllerConfig  # 懸浮控制器設定的實例


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    從 YAML 檔案載入設定並回傳一個 AppConfig 物件。

    Args:
        path (str): YAML 設定檔的路徑。預設為 "config.yaml"。

    Returns:
        AppConfig: 一個包含所有應用程式設定的實例。

    Raises:
        FileNotFoundError: 如果設定檔不存在。
        IOError: 如果讀取或解析檔案時發生錯誤。
    """
    try:
        # 使用 'with' 陳述式確保檔案會被正確關閉
        with open(path, "r", encoding="utf-8") as f:
            # yaml.safe_load 可以安全地解析 YAML 內容，避免執行任意程式碼
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        # 如果檔案找不到，拋出一個更具描述性的錯誤
        raise FileNotFoundError(f"設定檔 '{path}' 不存在。請確保檔案路徑正確。")
    except Exception as e:
        # 捕捉其他可能的錯誤，如權限問題或 YAML 格式錯誤
        raise IOError(f"讀取或解析設定檔 '{path}' 時發生錯誤: {e}")

    # 將 YAML 中讀取的字典資料，實例化為對應的 dataclass 物件
    # 這提供了型別檢查和自動完成的好處
    tuning_params = TuningParamsConfig(**config_data["initial_tuning_params"])
    floating_config = FloatingControllerConfig(**config_data["floating_controller"])

    # 建立最終的 AppConfig 物件，將所有解析後的設定組合起來
    config_obj = AppConfig(
        # 使用 .get() 方法安全地讀取可選的設定值，如果不存在則使用預設值
        use_virtual_teensy=config_data.get("use_virtual_teensy", False),
        mujoco_model_file=config_data["mujoco_model_file"],
        onnx_models=config_data["onnx_models"],
        policy_transition_duration=config_data.get("policy_transition_duration", 0.5),
        num_motors=config_data["num_motors"],
        physics_timestep=config_data["physics_timestep"],
        control_freq=config_data["control_freq"],
        # 在載入時直接計算出 control_dt，方便後續使用
        control_dt=1.0 / config_data["control_freq"],
        warmup_duration=config_data["warmup_duration"],
        command_scaling_factors=config_data["command_scaling_factors"],
        keyboard_velocity_adjust_step=config_data["keyboard_velocity_adjust_step"],
        gamepad_sensitivity=config_data["gamepad_sensitivity"],
        param_adjust_steps=config_data["param_adjust_steps"],
        # 賦予先前建立的結構化設定物件
        initial_tuning_params=tuning_params,
        floating_controller=floating_config,
    )

    # 在控制台打印成功訊息，方便除錯
    print("✅ 設定檔載入成功 (包含懸浮控制器設定)。")
    # 回傳完整的設定物件
    return config_obj
