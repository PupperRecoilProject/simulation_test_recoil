# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig

class PolicyManager:
    """
    封裝多個 ONNX 模型的載入、觀察歷史管理、平滑切換和推論邏輯。
    """
    def __init__(self, config: 'AppConfig', base_obs_dim: int):
        self.config = config
        self.base_obs_dim = base_obs_dim
        self.sessions = {} # 儲存所有已載入的 ONNX session
        self.model_names = [] # 儲存所有模型的名稱，用於循環
        
        print("--- 正在載入所有 ONNX 模型 ---")
        for name, path in config.onnx_models.items():
            print(f"  - 載入模型 '{name}' 從: {path}")
            try:
                # 載入 ONNX session，包含優化快取邏輯
                sess_options = ort.SessionOptions()
                cache_path = os.path.splitext(path)[0] + ".optimized.ort"
                sess_options.optimized_model_filepath = cache_path
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(path, sess_options=sess_options, providers=['CPUExecutionProvider'])
                
                # 驗證模型維度
                model_input_dim = session.get_inputs()[0].shape[1]
                if self.base_obs_dim > 0 and model_input_dim % self.base_obs_dim != 0:
                    print(f"    ⚠️ 警告: 模型 '{name}' 的輸入維度 ({model_input_dim}) 無法被基礎觀察維度 ({self.base_obs_dim}) 整除。")

                self.sessions[name] = session
                self.model_names.append(name)
            except Exception as e:
                print(f"    ❌ 錯誤: 無法載入模型 '{name}'。錯誤: {e}")

        if not self.sessions:
            sys.exit("❌ 致命錯誤: 未能成功載入任何 ONNX 模型。")

        # --- 狀態變數 ---
        self.active_policy_name = self.model_names[0]
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)
        self.obs_history = None # 將在 reset 時初始化

        # --- 平滑過渡相關變數 ---
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.old_policy_output = np.zeros(config.num_motors, dtype=np.float32)

        self.reset() # 初始設定
        print(f"✅ 策略管理器初始化完成，當前啟用模型: '{self.active_policy_name}'")

    def _get_session_info(self, name: str):
        """輔助函式，獲取指定模型的 session 和相關資訊"""
        session = self.sessions.get(name)
        if not session: return None, None, None, 0
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_input_dim = session.get_inputs()[0].shape[1]
        history_length = model_input_dim // self.base_obs_dim if self.base_obs_dim > 0 else 1
        return session, input_name, output_name, history_length

    def switch_policy(self, new_policy_name: str):
        """觸發向新模型的平滑過渡"""
        if new_policy_name not in self.sessions:
            print(f"⚠️ 警告: 無法切換，模型 '{new_policy_name}' 不存在。")
            return
        if new_policy_name == self.active_policy_name and not self.is_transitioning:
            return # 無需切換

        print(f"🚀 開始從 '{self.active_policy_name}' 平滑過渡到 '{new_policy_name}'...")
        self.is_transitioning = True
        self.transition_start_time = time.time()
        # 當前最後的動作輸出將作為過渡的起點
        self.old_policy_output = self.last_action.copy()
        self.active_policy_name = new_policy_name
        
        # 重置觀察歷史以適應新模型可能不同的歷史長度
        self.reset()

    def get_action(self, base_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """根據當前觀察和狀態（可能在過渡中）獲取動作"""
        self.obs_history.append(base_obs)
        onnx_input = np.concatenate(list(self.obs_history)).astype(np.float32).reshape(1, -1)

        # 獲取當前活動模型的輸出
        session, input_name, output_name, _ = self._get_session_info(self.active_policy_name)
        if not session or onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
            action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
        else:
            action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()

        # 如果正在進行平滑過渡，則進行插值
        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time
            duration = self.config.policy_transition_duration
            
            if duration <= 0: # 如果持續時間為0，立即切換
                self.is_transitioning = False
                final_action = action_raw
            elif elapsed >= duration:
                # 過渡結束
                self.is_transitioning = False
                final_action = action_raw
                print(f"✅ 已完成到 '{self.active_policy_name}' 的過渡。")
            else:
                # 線性插值 (alpha 從 0 變到 1)
                alpha = elapsed / duration
                final_action = (1.0 - alpha) * self.old_policy_output + alpha * action_raw
        else:
            final_action = action_raw

        self.last_action[:] = final_action # 儲存的是最終（可能被插值過的）動作
        return onnx_input, final_action

    def reset(self):
        """重置觀察歷史以適應當前活動模型。"""
        _, _, _, history_length = self._get_session_info(self.active_policy_name)
        
        self.obs_history = deque(
            [np.zeros(self.base_obs_dim, dtype=np.float32)] * history_length, 
            maxlen=history_length
        )
        print(f"✅ 策略狀態已為 '{self.active_policy_name}' 重置 (History Length: {history_length})。")