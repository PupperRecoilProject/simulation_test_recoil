# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os
from collections import deque
from config import AppConfig

class ONNXPolicy:
    """
    封裝 ONNX 模型的載入、觀察歷史管理和推論邏輯。
    """
    def __init__(self, config: AppConfig, base_obs_dim: int):
        self.config = config
        self.base_obs_dim = base_obs_dim
        
        print(f"正在載入 ONNX 模型: {config.onnx_model_path}")
        sess_options = ort.SessionOptions()
        cache_path = os.path.splitext(config.onnx_model_path)[0] + ".optimized.ort"
        
        if os.path.exists(cache_path):
            print(f"⚡️ 發現優化模型快取，將從 '{cache_path}' 快速載入。")
        else:
            print(f"🐢 首次載入，將創建優化模型快取於 '{cache_path}' (可能需要一些時間)...")

        sess_options.optimized_model_filepath = cache_path
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.sess = ort.InferenceSession(
                config.onnx_model_path, 
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        except Exception as e:
            sys.exit(f"❌ 錯誤: 無法載入 ONNX 模型 '{config.onnx_model_path}': {e}")

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.model_input_dim = self.sess.get_inputs()[0].shape[1]
        
        print(f"✅ 模型載入成功! 模型期望輸入維度: {self.model_input_dim}")
        self._determine_history_length()

        self.obs_history = deque(
            [np.zeros(self.base_obs_dim, dtype=np.float32)] * self.history_length, 
            maxlen=self.history_length
        )
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)

    def _determine_history_length(self):
        """根據模型輸入維度和基礎觀察維度，自動計算歷史長度。"""
        if self.base_obs_dim == 0:
            print("⚠️ 警告: 基礎觀察維度為 0，無法計算歷史長度。")
            self.history_length = 0
            return
            
        if self.model_input_dim % self.base_obs_dim != 0:
            print(
                f"⚠️ 警告: 基礎觀察維度 ({self.base_obs_dim}) 無法整除模型輸入維度 "
                f"({self.model_input_dim})。歷史堆疊功能可能不準確。"
            )
            self.history_length = 1
        else:
            self.history_length = self.model_input_dim // self.base_obs_dim
        
        if self.history_length > 1:
            print(f"🤖 自動偵測到模型使用歷史堆疊，長度為: {self.history_length} 幀。")
        else:
            print("🤖 模型僅使用當前觀察 (歷史長度 = 1)。")

    def get_action(self, base_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """根據當前的基礎觀察，更新歷史並執行模型推論，回傳動作。"""
        if self.history_length == 0:
            return np.array([]), np.zeros(self.config.num_motors)

        self.obs_history.append(base_obs)
        onnx_input = np.concatenate(list(self.obs_history)).astype(np.float32).reshape(1, -1)
        
        if onnx_input.shape[1] != self.model_input_dim:
            return onnx_input, np.zeros(self.config.num_motors)
            
        action_raw = self.sess.run([self.output_name], {self.input_name: onnx_input})[0].flatten()
        self.last_action[:] = action_raw
        return onnx_input, action_raw

    def reset(self):
        """重置觀察歷史和上一個動作。"""
        self.obs_history.clear()
        if self.history_length > 0:
            for _ in range(self.history_length):
                self.obs_history.append(np.zeros(self.base_obs_dim, dtype=np.float32))
        self.last_action.fill(0.0)
        print("✅ ONNX 策略狀態已重置。")