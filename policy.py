# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os
import time
from collections import deque
from typing import TYPE_CHECKING, List, Dict, Tuple
from logger import log

# 為了型別提示，避免循環匯入
if TYPE_CHECKING:
    from config import AppConfig
    from observation import ObservationBuilder
    from rendering import DebugOverlay

class PolicyManager:
    """
    【版本 2.1 - 重構版】 - AI 策略大腦
    管理多個 ONNX 策略模型。它能夠：
    1. 在啟動時載入所有在 config.yaml 中定義的模型。
    2. 為每個模型維護獨立的觀察歷史，以支援需要多幀輸入的模型。
    3. 根據使用者指令，在兩個不同的策略模型之間進行平滑的線性融合（插值）。
    4. 【重構】合併了模擬和硬體的動作獲取邏輯，減少程式碼重複。
    """
    def __init__(self, config: 'AppConfig', obs_builder: 'ObservationBuilder', overlay: 'DebugOverlay'):
        self.config = config  # 儲存應用程式的全域設定
        self.obs_builder = obs_builder  # 儲存觀察向量產生器的參考
        self.overlay = overlay  # 除錯介面(DebugOverlay)的參考
        self.sessions: Dict[str, ort.InferenceSession] = {}  # 以模型名稱為鍵的 session 字典
        self.model_recipes: Dict[str, List[str]] = {}  # 每個模型的觀察配方
        self.model_history_lengths: Dict[str, int] = {}  # 每個模型需要的歷史幀數
        self.model_names: List[str] = []

        print("--- 正在載入所有 ONNX 模型及其配方 ---")
        for name, model_info in config.onnx_models.items():
            path = model_info.get('path')
            recipe = model_info.get('observation_recipe')
            if not path or not recipe:
                log.warning(f"模型 '{name}' 缺少 'path' 或 'observation_recipe'，已跳過。")
                continue

            log.info(f"  - 載入模型 '{name}' 從: {path}")
            try:
                sess_options = ort.SessionOptions()
                cache_path = os.path.splitext(path)[0] + ".optimized.ort"
                sess_options.optimized_model_filepath = cache_path
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(path, sess_options=sess_options, providers=['CPUExecutionProvider'])

                # 推斷觀察維度與歷史長度
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
                log.info(f"    > 配方: {recipe}")
                log.info(f"    > 基礎維度: {base_obs_dim}, 模型輸入: {model_input_dim}, 推斷歷史長度: {history_len}")
            except Exception as e:
                log.error(f"無法載入模型 '{name}'。錯誤: {e}")

        if not self.sessions:
            sys.exit("❌ 致命錯誤: 未能成功載入任何 ONNX 模型。")

        self.primary_policy_name = self.model_names[0]
        self.source_policy_name = self.model_names[0]
        self.target_policy_name = self.model_names[0]
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)
        self.obs_histories: Dict[str, deque] = {}
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.transition_alpha = 0.0

        self.reset()

        print("--- 正在預熱所有 ONNX 模型 (強制進行首次推論優化)... ---")
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            dummy_input = np.zeros((1, session.get_inputs()[0].shape[1]), dtype=np.float32)
            try:
                session.run([output_name], {input_name: dummy_input})
                print(f"  - 模型 '{name}' 預熱成功。")
            except Exception as e:
                print(f"  - ⚠️ 模型 '{name}' 預熱失敗: {e}")

        print(f"✅ 策略管理器初始化完成，主要模型: '{self.primary_policy_name}'")

    def get_active_recipe(self) -> List[str]:
        """返回目前主要模型所需的觀察配方。"""
        return self.model_recipes.get(self.primary_policy_name, [])

    def select_target_policy(self, target_name: str):
        """(由UI或鍵盤觸發) 選擇一個目標策略並開始平滑轉換。"""
        if target_name not in self.sessions:
            log.warning(f"無法切換，目標模型 '{target_name}' 不存在。")
            return
        if self.is_transitioning or target_name == self.primary_policy_name:
            return
        log.info(f"🚀 開始從 '{self.primary_policy_name}' 融合至 '{target_name}'...")
        self.is_transitioning = True
        self.transition_start_time = time.time()
        self.transition_alpha = 0.0
        self.source_policy_name = self.primary_policy_name
        self.target_policy_name = target_name

    def _get_action_internal(self, base_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        【重構】內部核心函式：處理觀察歷史並運行所有模型，回傳主要模型輸入與最終動作。
        """
        all_actions: Dict[str, np.ndarray] = {}
        primary_onnx_input = np.array([])

        for name, session in self.sessions.items():
            self.obs_histories[name].append(base_obs)
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()
            all_actions[name] = action_raw
            if name == self.primary_policy_name:
                primary_onnx_input = onnx_input

        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time
            duration = self.config.policy_transition_duration
            self.transition_alpha = min(elapsed / duration, 1.0) if duration > 0 else 1.0
            source_action = all_actions[self.source_policy_name]
            target_action = all_actions[self.target_policy_name]
            final_action = (1.0 - self.transition_alpha) * source_action + self.transition_alpha * target_action
            if self.transition_alpha >= 1.0:
                log.info(f"✅ 已完成向 '{self.target_policy_name}' 的融合。")
                self.is_transitioning = False
                self.primary_policy_name = self.target_policy_name
        else:
            final_action = all_actions[self.primary_policy_name]

        self.last_action[:] = final_action
        return primary_onnx_input, final_action

    def get_action(self, command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        【模擬專用】此函式會自行利用 ObservationBuilder 來生成單幀觀察向量。
        """
        self.obs_builder.set_recipe(self.get_active_recipe())
        base_obs = self.obs_builder.get_observation(command, self.last_action)
        return self._get_action_internal(base_obs)

    def get_action_for_hardware(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        【硬體專用】接收已由 HardwareController 建立好的觀察向量。
        注意：硬體端無法提供 `linear_velocity`，若模型需要此資訊仍會以零填充並發出警告。
        """
        active_recipe = self.get_active_recipe()
        if 'linear_velocity' in active_recipe:
            log.warning("警告：目前策略需要 'linear_velocity'，但硬體不支援此數據。模型表現可能不佳。")
        return self._get_action_internal(observation)

    def reset(self):
        """重置所有模型的觀察歷史與相關狀態。"""
        active_recipe = self.model_recipes[self.primary_policy_name]
        self.obs_builder.set_recipe(active_recipe)
        if self.overlay:
            self.overlay.set_recipe(active_recipe)
        for name in self.model_names:
            recipe = self.model_recipes[name]
            self.obs_builder.set_recipe(recipe)
            base_obs_dim = len(self.obs_builder.get_observation(np.zeros(3), np.zeros(self.config.num_motors)))
            history_length = self.model_history_lengths[name]
            self.obs_histories[name] = deque(
                [np.zeros(base_obs_dim, dtype=np.float32)] * history_length,
                maxlen=history_length,
            )
        self.obs_builder.set_recipe(active_recipe)
        self.is_transitioning = False
        log.info(f"所有策略狀態已重置。主要模型: '{self.primary_policy_name}'。")
