# src/hardware/policy.py
"""
【v4.4.5 修改】AI 策略大腦 (適應內部緩存供給者)

【v2.0 新增】AI 策略大腦

管理多個 ONNX 策略模型。它能夠：
1. 在啟動時載入所有在 config.yaml 中定義的模型。
2. 為每個模型維護獨立的觀察歷史。
3. 根據使用者指令，在兩個不同的策略模型之間進行平滑的線性融合（插值）。
4. 【v4.4.5 修改】不再觸發 ObservationManager 的計算，而是從中請求已標準化的數據分量。
"""
# ==================================
# === 標準庫導入 (Standard Libraries)
# ==================================
import sys
import os
import time
from collections import deque

# ==================================
# === 第三方庫導入 (Third-party Libraries)
# ==================================
import numpy as np
import onnxruntime as ort

# ==================================
# === 專案內部模組導入 (Internal Modules)
# ==================================
from src.core.logger import log

# ==================================
# === 類型提示專用區塊 (Type Hinting Only)
# ==================================
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.simulation.observation_manager import ObservationManager
    from src.simulation.rendering import DebugOverlay
    from src.core.state import SimulationState


class PolicyManager:
    """【v4.4.5 修改】AI 策略管理器 (帶內部緩存的數據供給者適配版)。"""

    # ============================ 初始化區塊 ============================
    # 【v4.4.5 修改】__init__ 函式，新增 state 參數
    def __init__(self, config: 'AppConfig', observation_manager: 'ObservationManager', overlay: 'DebugOverlay', state: 'SimulationState'):
        """
        【v4.3.2 修改】構造函式，接收 observation_manager。
        【v4.4.5 修改】新增 state 參數，以便直接訪問 SimulationState。
        """
        self.config = config
        self.observation_manager = observation_manager
        self.overlay = overlay
        # 【v4.4.5 新增】儲存對中央狀態的參考
        self.state = state
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.model_recipes: Dict[str, List[str]] = {}
        self.model_history_lengths: Dict[str, int] = {}
        self.model_names: List[str] = []
        
        # --- 模型載入與預熱 ---
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

                base_obs_dim = 0
                # 【v4.4.5 修改】從 ObservationManager 的 ALL_OBS_DIMS 獲取維度資訊
                for comp_name in recipe:
                    dim = self.observation_manager.ALL_OBS_DIMS.get(comp_name)
                    if dim is None:
                        print(f"    ⚠️ 警告: 配方中包含未知元件 '{comp_name}'。")
                        continue
                    base_obs_dim += dim

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

        self.primary_policy_name = self.model_names[0]
        self.source_policy_name = self.model_names[0]
        self.target_policy_name = self.model_names[0]
        
        self.last_action = np.zeros(config.num_motors, dtype=np.float32)
        
        self.obs_histories: Dict[str, deque] = {}
        
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.transition_alpha = 0.0

        self.reset() # 初始化所有模型的觀察歷史

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

        print(f"✅ 策略管理器初始化完成，主要模型: '{self.primary_policy_name}'")

    # ========================== 策略管理方法區塊 ==========================
    def get_active_recipe(self) -> List[str]:
        """【v4.3.2 修改】一個輔助函式，返回當前主要策略所使用的觀察配方。"""
        return self.model_recipes.get(self.primary_policy_name, [])

    def select_target_policy(self, target_name: str):
        """【v4.3.2 修改】(由鍵盤/UI觸發) 選擇一個目標策略並開始平滑轉換。"""
        if target_name not in self.sessions:
            print(f"⚠️ 警告: 無法切換，目標模型 '{target_name}' 不存在。")
            return
        if self.is_transitioning or target_name == self.primary_policy_name:
            return

        print(f"🚀 開始從 '{self.primary_policy_name}' 線性融合至 '{target_name}'...")
        self.is_transitioning = True
        self.transition_start_time = time.time()
        self.transition_alpha = 0.0
        self.source_policy_name = self.primary_policy_name
        self.target_policy_name = target_name

    # ========================== 動作生成與推論區塊 ==========================
    # 【v4.4.6 重構】get_action 函式
    def get_action(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        【v4.4.6 重構】從 state.std_obs 中拾取數據並拼接成模型輸入。
        
        說明：此函式不再觸發任何計算。它假定 state.std_obs 已經由
        主控制迴圈（SimulationController）用最新鮮的數據進行了更新。
        """
        # 【v4.4.6 刪除】不再需要由 PolicyManager 觸發任何 ObservationManager 的計算
        # self.observation_manager.new_frame()
        # self.observation_manager.update_all_observations()

        all_actions = {}
        primary_onnx_input = np.array([])
        
        # 【v4.4.6 新增】在一個鎖定的區塊內，從 state 複製一份標準化觀測數據字典
        with self.state.lock:
            # 增加 hasattr 檢查以提高魯棒性，防止在初始化早期出錯
            if hasattr(self.state, 'std_obs'):
                current_std_obs = self.state.std_obs.copy()
            else:
                # 如果 std_obs 不存在，則創建一個空的，以避免後續崩潰
                current_std_obs = {}

        # --- 遍歷所有模型，生成動作 ---
        for name, session in self.sessions.items():
            recipe = self.model_recipes[name]
            
            # 【v4.4.6 修改】從複製的字典中拾取數據並拼接
            try:
                obs_list = [current_std_obs[comp_name] for comp_name in recipe]
                base_obs = np.concatenate(obs_list)
            except KeyError as e:
                log.error(f"模型 '{name}' 的配方中包含的觀測組件 '{e}' 在 state.std_obs 中不存在。")
                # 提供一個符合維度的零向量以避免崩潰
                base_obs_dim = sum(self.observation_manager.ALL_OBS_DIMS.get(comp_name, 0) for comp_name in recipe)
                base_obs = np.zeros(base_obs_dim)

            # --- 更新觀測歷史並拼接成 ONNX 輸入 ---
            self.obs_histories[name].append(base_obs)
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            
            # --- 執行 ONNX 推論 ---
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                log.warning(f"模型 '{name}' 輸入維度不匹配，預期 {session.get_inputs()[0].shape[1]} 但得到 {onnx_input.shape[1]}。將返回零動作。")
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()
            
            all_actions[name] = action_raw

            if name == self.primary_policy_name:
                primary_onnx_input = onnx_input
        
        # --- 根據狀態決定最終動作 (融合邏輯) ---
        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time
            duration = self.config.policy_transition_duration
            
            if duration > 0: self.transition_alpha = min(elapsed / duration, 1.0)
            else: self.transition_alpha = 1.0

            source_action = all_actions[self.source_policy_name]
            target_action = all_actions[self.target_policy_name]
            final_action = (1.0 - self.transition_alpha) * source_action + self.transition_alpha * target_action

            if self.transition_alpha >= 1.0:
                print(f"✅ 已完成向 '{self.target_policy_name}' 的融合。")
                self.is_transitioning = False
                self.primary_policy_name = self.target_policy_name
        else:
            final_action = all_actions[self.primary_policy_name]

        # 【v4.4.6 修改】回寫 last_action 的職責現在更清晰：
        # PolicyManager 負責產生動作，並將其記錄為下一幀的輸入。
        self.last_action[:] = final_action
        with self.state.lock:
            self.state.raw_last_action = final_action

        return primary_onnx_input, final_action

    # 【v4.4.6 重構】get_action_for_hardware 函式
    def get_action_for_hardware(self) -> tuple[np.ndarray, np.ndarray]:
        """
        【v4.4.6 重構】硬體模式下的數據拾取與推論邏輯，與 get_action 完全同步。
        """
        # (此函式的內部修改邏輯與 get_action() 完全相同)
        # 【v4.4.6 刪除】不再觸發計算
        # self.observation_manager.new_frame()
        
        all_actions = {}
        primary_onnx_input = np.array([])
        
        with self.state.lock:
            if hasattr(self.state, 'std_obs'):
                current_std_obs = self.state.std_obs.copy()
            else:
                current_std_obs = {}

        for name, session in self.sessions.items():
            recipe = self.model_recipes[name]
            
            try:
                obs_list = [current_std_obs[comp_name] for comp_name in recipe]
                base_obs = np.concatenate(obs_list)
            except KeyError as e:
                log.error(f"模型 '{name}' 的配方中包含的觀測組件 '{e}' 在 state.std_obs 中不存在。")
                base_obs_dim = sum(self.observation_manager.ALL_OBS_DIMS.get(comp_name, 0) for comp_name in recipe)
                base_obs = np.zeros(base_obs_dim)

            # --- 更新觀測歷史並拼接成 ONNX 輸入 ---
            self.obs_histories[name].append(base_obs)
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            
            # --- 執行 ONNX 推論 ---
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                log.warning(f"模型 '{name}' 輸入維度不匹配，預期 {session.get_inputs()[0].shape[1]} 但得到 {onnx_input.shape[1]}。將返回零動作。")
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()
            
            all_actions[name] = action_raw

            if name == self.primary_policy_name:
                primary_onnx_input = onnx_input

        # --- 根據狀態決定最終動作 (融合邏輯) ---
        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time
            duration = self.config.policy_transition_duration
            if duration > 0: self.transition_alpha = min(elapsed / duration, 1.0)
            else: self.transition_alpha = 1.0
            source_action = all_actions[self.source_policy_name]
            target_action = all_actions[self.target_policy_name]
            final_action = (1.0 - self.transition_alpha) * source_action + self.transition_alpha * target_action
            if self.transition_alpha >= 1.0:
                self.is_transitioning = False
                self.primary_policy_name = self.target_policy_name
        else:
            final_action = all_actions[self.primary_policy_name]

        self.last_action[:] = final_action
        # 【v4.4.5 修改】確保在硬體模式下也將最終動作寫回 state.raw_last_action。
        with self.state.lock:
            self.state.raw_last_action = final_action

        return primary_onnx_input, final_action

    # ========================== 重置與清理區塊 ==========================
    # 【v4.4.5 重構】reset 函式
    def reset(self):
        """
        【v4.4.5 修改】簡化 reset 邏輯，不再需要處理 recipe 相關的設置。
        【v4.3.2 修改】重置所有模型的觀察歷史。
        """
        # 【v4.4.5 刪除】不再需要與 overlay 交互 recipe
        # if self.overlay:
        #     self.overlay.set_recipe(...)
        # active_recipe = self.model_recipes[self.primary_policy_name]
        # self.observation_manager.set_recipe(active_recipe)

        for name in self.model_names:
            recipe = self.model_recipes[name]
            # 計算基礎維度
            # 【v4.4.5 修改】從 ObservationManager 的 ALL_OBS_DIMS 獲取維度資訊
            base_obs_dim = sum(self.observation_manager.ALL_OBS_DIMS[comp_name] for comp_name in recipe if comp_name in self.observation_manager.ALL_OBS_DIMS)
            
            history_length = self.model_history_lengths[name]
            
            self.obs_histories[name] = deque(
                [np.zeros(base_obs_dim, dtype=np.float32)] * history_length,
                maxlen=history_length
            )
        
        self.is_transitioning = False
        print(f"✅ 所有策略狀態已重置。主要模型: '{self.primary_policy_name}'。")