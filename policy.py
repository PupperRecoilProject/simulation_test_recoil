# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os
import time
from collections import deque
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from config import AppConfig
    from observation import ObservationBuilder
    from rendering import DebugOverlay # <-- 新增

class PolicyManager:
    def __init__(self, config: 'AppConfig', obs_builder: 'ObservationBuilder', overlay: 'DebugOverlay'): # <-- 接收 overlay
        self.config = config
        self.obs_builder = obs_builder
        self.overlay = overlay # <-- 儲存 overlay 的參考
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.model_recipes: Dict[str, List[str]] = {}
        self.model_history_lengths: Dict[str, int] = {}
        self.model_names: List[str] = []
        
        print("--- 正在載入所有 ONNX 模型及其配方 ---")
        for name, model_info in config.onnx_models.items():
            path = model_info.get('path')
            recipe = model_info.get('observation_recipe')

            if not path or not recipe:
                print(f"    ⚠️ 警告: 模型 '{name}' 缺少 'path' 或 'observation_recipe'，已跳過。")
                continue

            print(f"  - 載入模型 '{name}' 從: {path}")
            try:
                sess_options = ort.SessionOptions()
                cache_path = os.path.splitext(path)[0] + ".optimized.ort"
                sess_options.optimized_model_filepath = cache_path
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(path, sess_options=sess_options, providers=['CPUExecutionProvider'])

                self.obs_builder.set_recipe(recipe)
                base_obs_dim = len(self.obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors)))
                model_input_dim = session.get_inputs()[0].shape[1]
                history_len = 1
                if base_obs_dim > 0 and model_input_dim % base_obs_dim == 0:
                    history_len = model_input_dim // base_obs_dim
                
                self.sessions[name] = session
                self.model_recipes[name] = recipe
                self.model_history_lengths[name] = history_len
                self.model_names.append(name)
                print(f"    > 配方: {recipe}")
                print(f"    > 基礎維度: {base_obs_dim}, 模型輸入: {model_input_dim}, 推斷歷史長度: {history_len}")

            except Exception as e:
                print(f"    ❌ 錯誤: 無法載入模型 '{name}'。錯誤: {e}")

        if not self.sessions:
            sys.exit("❌ 致命錯誤: 未能成功載入任何 ONNX 模型。")

        self.active_policy_name = self.model_names[0]
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)
        self.obs_history = None
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.old_policy_output = np.zeros(config.num_motors, dtype=np.float32)

        self.reset()

        print("--- 正在預熱所有 ONNX 模型 (強制進行首次推論優化)... ---")
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            model_input_dim = session.get_inputs()[0].shape[1]
            dummy_input = np.zeros((1, model_input_dim), dtype=np.float32)
            try:
                session.run([output_name], {input_name: dummy_input})
                print(f"  - 模型 '{name}' 預熱成功。")
            except Exception as e:
                print(f"  - ⚠️ 模型 '{name}' 預熱失敗: {e}")

        print(f"✅ 策略管理器初始化完成，當前啟用模型: '{self.active_policy_name}'")

    def switch_policy(self, new_policy_name: str):
        if new_policy_name not in self.sessions:
            print(f"⚠️ 警告: 無法切換，模型 '{new_policy_name}' 不存在。")
            return
        if new_policy_name == self.active_policy_name and not self.is_transitioning:
            return

        print(f"🚀 開始從 '{self.active_policy_name}' 平滑過渡到 '{new_policy_name}'...")
        self.is_transitioning = True
        self.transition_start_time = time.time()
        self.old_policy_output = self.last_action.copy()
        self.active_policy_name = new_policy_name
        self.reset()

    def get_action(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_obs = self.obs_builder.get_observation(command, self.last_action)
        self.obs_history.append(base_obs)
        onnx_input = np.concatenate(list(self.obs_history)).astype(np.float32).reshape(1, -1)
        session = self.sessions[self.active_policy_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
            action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
        else:
            action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()

        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time
            duration = self.config.policy_transition_duration
            if duration <= 0 or elapsed >= duration:
                self.is_transitioning = False
                final_action = action_raw
                if duration > 0: print(f"✅ 已完成到 '{self.active_policy_name}' 的過渡。")
            else:
                alpha = elapsed / duration
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                final_action = (1.0 - smooth_alpha) * self.old_policy_output + smooth_alpha * action_raw
        else:
            final_action = action_raw

        self.last_action[:] = final_action
        return onnx_input, final_action

    def reset(self):
        """重置觀察歷史並同步切換觀察配方。"""
        active_recipe = self.model_recipes[self.active_policy_name]
        
        # 【修改】同時更新 obs_builder 和 overlay 的配方
        self.obs_builder.set_recipe(active_recipe)
        if self.overlay:
            self.overlay.set_recipe(active_recipe)

        history_length = self.model_history_lengths[self.active_policy_name]
        base_obs_dim = len(self.obs_builder.get_observation(np.zeros(3), np.zeros(self.config.num_motors)))

        self.obs_history = deque(
            [np.zeros(base_obs_dim, dtype=np.float32)] * history_length, 
            maxlen=history_length
        )
        print(f"✅ 策略狀態已為 '{self.active_policy_name}' 重置 (History: {history_length}, Obs Dim: {base_obs_dim})。")