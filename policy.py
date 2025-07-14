import numpy as np
import onnxruntime as ort
import sys
from collections import deque
from config import AppConfig

class ONNXPolicy:
    """
    封裝 ONNX 模型的載入、觀察歷史管理和推論邏輯。
    """
    def __init__(self, config: AppConfig, base_obs_dim: int):
        """
        初始化 ONNXPolicy。

        Args:
            config (AppConfig): 應用程式的設定物件。
            base_obs_dim (int): 單幀基礎觀察的維度。
        """
        self.config = config
        self.base_obs_dim = base_obs_dim
        
        print(f"正在載入 ONNX 模型: {config.onnx_model_path}")
        try:
            self.sess = ort.InferenceSession(config.onnx_model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            sys.exit(f"❌ 錯誤: 無法載入 ONNX 模型 '{config.onnx_model_path}': {e}")

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.model_input_dim = self.sess.get_inputs()[0].shape[1]
        
        print(f"✅ 模型載入成功! 模型期望輸入維度: {self.model_input_dim}")

        self._determine_history_length()

        # 初始化觀察歷史佇列和上一個動作
        self.obs_history = deque(
            [np.zeros(self.base_obs_dim, dtype=np.float32)] * self.history_length, 
            maxlen=self.history_length
        )
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)

    def _determine_history_length(self):
        """根據模型輸入維度和基礎觀察維度，自動計算歷史長度。"""
        if self.model_input_dim % self.base_obs_dim != 0:
            sys.exit(
                f"❌ 致命錯誤: 基礎觀察維度 ({self.base_obs_dim}) 無法整除模型輸入維度 "
                f"({self.model_input_dim})。無法確定歷史長度。"
            )
        self.history_length = self.model_input_dim // self.base_obs_dim
        
        if self.history_length > 1:
            print(f"🤖 自動偵測到模型使用歷史堆疊，長度為: {self.history_length} 幀。")
        else:
            print("🤖 模型僅使用當前觀察 (歷史長度 = 1)。")

    def get_action(self, base_obs: np.ndarray) -> np.ndarray:
        """
        根據當前的基礎觀察，更新歷史並執行模型推論，回傳動作。

        Args:
            base_obs (np.ndarray): 由 ObservationBuilder 產生的當前幀基礎觀察。

        Returns:
            np.ndarray: ONNX 模型輸出的原始動作。
        """
        # 將最新的觀察加入歷史佇列
        self.obs_history.append(base_obs)
        
        # 將歷史佇列中的所有觀察拼接成單一向量，作為模型輸入
        onnx_input = np.concatenate(list(self.obs_history)).reshape(1, -1)
        
        # 執行推論
        action_raw = self.sess.run([self.output_name], {self.input_name: onnx_input})[0].flatten()
        
        # 更新 last_action，供下一個時間步的觀察使用
        self.last_action[:] = action_raw
        
        return onnx_input, action_raw

    def reset(self):
        """重置觀察歷史和上一個動作。"""
        self.obs_history.clear()
        for _ in range(self.history_length):
            self.obs_history.append(np.zeros(self.base_obs_dim, dtype=np.float32))
        self.last_action.fill(0.0)
        print("✅ ONNX 策略狀態已重置 (歷史觀察、上次動作)。")