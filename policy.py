# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os # <--- 導入 os 模組來處理路徑
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

        # --- 開始修改：啟用 ONNX 優化模型快取 ---
        sess_options = ort.SessionOptions()

        # 產生優化後模型的儲存路徑，例如： models/my_model.onnx -> models/my_model.optimized.ort
        cache_path = os.path.splitext(config.onnx_model_path)[0] + ".optimized.ort"
        
        # 檢查快取檔案是否存在
        if os.path.exists(cache_path):
            print(f"⚡️ 發現優化模型快取，將從 '{cache_path}' 快速載入。")
        else:
            print(f"🐢 首次載入，將創建優化模型快取於 '{cache_path}' (可能需要一些時間)...")

        sess_options.optimized_model_filepath = cache_path
        # 啟用所有可用的 CPU 優化
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # --- 結束修改 ---

        try:
            # 將 session options 傳入 InferenceSession
            self.sess = ort.InferenceSession(
                config.onnx_model_path, 
                sess_options=sess_options, # <--- 使用我們建立的選項
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

    def get_action(self, base_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        根據當前的基礎觀察，更新歷史並執行模型推論，回傳動作。

        Args:
            base_obs (np.ndarray): 由 ObservationBuilder 產生的當前幀基礎觀察。

        Returns:
            tuple[np.ndarray, np.ndarray]: (模型輸入向量, 模型輸出的原始動作)
        """
        self.obs_history.append(base_obs)
        
        onnx_input = np.concatenate(list(self.obs_history)).reshape(1, -1)
        
        action_raw = self.sess.run([self.output_name], {self.input_name: onnx_input})[0].flatten()
        
        self.last_action[:] = action_raw
        
        return onnx_input, action_raw

    def reset(self):
        """重置觀察歷史和上一個動作。"""
        self.obs_history.clear()
        for _ in range(self.history_length):
            self.obs_history.append(np.zeros(self.base_obs_dim, dtype=np.float32))
        self.last_action.fill(0.0)
        print("✅ ONNX 策略狀態已重置 (歷史觀察、上次動作)。")