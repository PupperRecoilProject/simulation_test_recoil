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
    from rendering import DebugOverlay

class PolicyManager:
    """
    【版本 2.0】
    管理多個 ONNX 策略模型，能夠同時運行所有模型，
    並在使用者選擇的兩個策略之間進行線性融合。
    """
    def __init__(self, config: 'AppConfig', obs_builder: 'ObservationBuilder', overlay: 'DebugOverlay'): # <-- 接收 overlay
        self.config = config # 儲存應用程式設定
        self.obs_builder = obs_builder # 儲存觀察向量產生器的參考
        self.overlay = overlay # <-- 儲存 overlay 的參考
        self.sessions: Dict[str, ort.InferenceSession] = {} # 字典，用來儲存已載入的 ONNX 推論 session
        self.model_recipes: Dict[str, List[str]] = {} # 字典，儲存每個模型對應的觀察配方
        self.model_history_lengths: Dict[str, int] = {} # 字典，儲存每個模型需要的歷史觀察長度
        self.model_names: List[str] = [] # 列表，儲存所有成功載入模型的名稱
        
        print("--- 正在載入所有 ONNX 模型及其配方 ---")
        # 遍歷設定檔中定義的所有模型
        for name, model_info in config.onnx_models.items():
            path = model_info.get('path') # 獲取模型檔案路徑
            recipe = model_info.get('observation_recipe') # 獲取模型對應的觀察配方

            if not path or not recipe:
                print(f"    ⚠️ 警告: 模型 '{name}' 缺少 'path' 或 'observation_recipe'，已跳過。")
                continue # 如果路徑或配方不存在，則跳過

            print(f"  - 載入模型 '{name}' 從: {path}")
            try:
                # --- ONNX Runtime 優化與載入 ---
                sess_options = ort.SessionOptions() # 建立 ONNX Runtime 的 session 設定
                cache_path = os.path.splitext(path)[0] + ".optimized.ort" # 定義優化後模型的快取檔案路徑
                sess_options.optimized_model_filepath = cache_path # 將快取路徑設定給 session
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 啟用所有圖優化
                # 載入 session。此時若 .ort 快取檔案不存在，ONNX Runtime 會進行優化並生成它
                session = ort.InferenceSession(path, sess_options=sess_options, providers=['CPUExecutionProvider'])

                # --- 推斷模型輸入維度和歷史長度 ---
                self.obs_builder.set_recipe(recipe) # 暫時設定配方以計算基礎維度
                base_obs_dim = len(self.obs_builder.get_observation(np.zeros(3), np.zeros(config.num_motors))) # 取得單幀觀察的維度
                model_input_dim = session.get_inputs()[0].shape[1] # 從模型本身獲取其輸入層的總維度
                history_len = 1 # 預設歷史長度為 1
                if base_obs_dim > 0 and model_input_dim % base_obs_dim == 0:
                    history_len = model_input_dim // base_obs_dim # 計算出模型需要的歷史幀數
                
                # 儲存模型相關資訊
                self.sessions[name] = session
                self.model_recipes[name] = recipe
                self.model_history_lengths[name] = history_len
                self.model_names.append(name)
                print(f"    > 配方: {recipe}")
                print(f"    > 基礎維度: {base_obs_dim}, 模型輸入: {model_input_dim}, 推斷歷史長度: {history_len}")

            except Exception as e:
                print(f"    ❌ 錯誤: 無法載入模型 '{name}'。錯誤: {e}")

        if not self.sessions:
            sys.exit("❌ 致命錯誤: 未能成功載入任何 ONNX 模型。") # 如果沒有任何模型成功載入，則終止程式

        # --- 【新】狀態變數，用於管理多模型融合 ---
        self.primary_policy_name = self.model_names[0] # 當前穩定的主要策略
        self.source_policy_name = self.model_names[0]  # 開始轉換時的來源策略
        self.target_policy_name = self.model_names[0]  # 正在轉換去的目標策略
        
        self.last_action = np.zeros(config.num_motors, dtype=np.float32) # 初始化上一次的動作向量
        
        # 為每個模型維護一個獨立的觀察歷史
        self.obs_histories: Dict[str, deque] = {} # 字典，key為模型名稱，value為對應的觀察歷史佇列
        
        self.is_transitioning = False # 是否正在進行模型融合的旗標
        self.transition_start_time = 0.0 # 融合開始的時間戳
        self.transition_alpha = 0.0 # 線性融合的權重 (0.0 代表完全是 source, 1.0 代表完全是 target)

        self.reset() # 初始化所有模型的觀察歷史

        print("--- 正在預熱所有 ONNX 模型 (強制進行首次推論優化)... ---")
        # 遍歷所有載入的 session
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name # 獲取輸入層名稱
            output_name = session.get_outputs()[0].name # 獲取輸出層名稱
            model_input_dim = session.get_inputs()[0].shape[1] # 獲取輸入維度
            dummy_input = np.zeros((1, model_input_dim), dtype=np.float32) # 建立一個符合維度的假輸入
            try:
                session.run([output_name], {input_name: dummy_input}) # 執行一次推論以觸發可能的 JIT 編譯或優化
                print(f"  - 模型 '{name}' 預熱成功。")
            except Exception as e:
                print(f"  - ⚠️ 模型 '{name}' 預熱失敗: {e}")

        print(f"✅ 策略管理器初始化完成，主要模型: '{self.primary_policy_name}'")

    def select_target_policy(self, target_name: str):
        """【新】選擇一個目標策略並開始轉換。"""
        if target_name not in self.sessions:
            print(f"⚠️ 警告: 無法切換，目標模型 '{target_name}' 不存在。") # 檢查目標是否存在
            return
        # 如果正在轉換中，或目標就是當前的主要模型，則不執行任何操作
        if self.is_transitioning or target_name == self.primary_policy_name:
            return

        print(f"🚀 開始從 '{self.primary_policy_name}' 線性融合至 '{target_name}'...")
        self.is_transitioning = True # 設定轉換旗標
        self.transition_start_time = time.time() # 記錄起始時間
        self.transition_alpha = 0.0 # 重置融合權重
        self.source_policy_name = self.primary_policy_name # 當前的主要模型成為來源
        self.target_policy_name = target_name # 設定目標模型

    def get_action(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        【重構】獲取最終動作。此版本會運行所有模型，並根據狀態進行融合。
        """
        all_actions = {} # 建立一個字典來儲存本幀所有模型的輸出
        primary_onnx_input = np.array([]) # 用於除錯顯示的輸入

        # --- 步驟 1: 運行所有模型，獲取各自的輸出 ---
        for name, session in self.sessions.items():
            recipe = self.model_recipes[name] # 獲取該模型對應的配方
            self.obs_builder.set_recipe(recipe) # 動態設定觀察產生器的配方
            
            # 產生觀察並更新對應模型的歷史
            base_obs = self.obs_builder.get_observation(command, self.last_action)
            self.obs_histories[name].append(base_obs)
            
            # 拼接成 ONNX 輸入
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            
            # 檢查維度是否匹配
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32) # 維度不符時返回零向量
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten() # 執行推論
            
            all_actions[name] = action_raw # 將模型的輸出存入字典

            # 如果是當前主要模型，儲存其輸入以供除錯介面顯示
            if name == self.primary_policy_name:
                primary_onnx_input = onnx_input

        # --- 步驟 2: 根據狀態決定最終動作 ---
        if self.is_transitioning:
            elapsed = time.time() - self.transition_start_time # 計算經過時間
            duration = self.config.policy_transition_duration # 讀取設定的總時長
            
            if duration > 0:
                self.transition_alpha = min(elapsed / duration, 1.0) # 線性計算 alpha，並限制在 [0, 1] 範圍
            else:
                self.transition_alpha = 1.0 # 如果持續時間為0，直接完成

            # 根據 alpha 在來源和目標策略的輸出之間進行線性插值 (Lerp)
            source_action = all_actions[self.source_policy_name]
            target_action = all_actions[self.target_policy_name]
            final_action = (1.0 - self.transition_alpha) * source_action + self.transition_alpha * target_action

            # 如果融合完成
            if self.transition_alpha >= 1.0:
                print(f"✅ 已完成向 '{self.target_policy_name}' 的融合。")
                self.is_transitioning = False # 結束轉換狀態
                self.primary_policy_name = self.target_policy_name # 目標模型成為新的主要模型
        else:
            # 如果不在轉換中，直接使用主要模型的輸出
            final_action = all_actions[self.primary_policy_name]

        self.last_action[:] = final_action # 更新 last_action 供下一幀使用
        return primary_onnx_input, final_action # 返回主要模型的輸入和最終融合後的動作

    def reset(self):
        """重置所有模型的觀察歷史，並設定 Overlay。"""
        # 重置主模型的觀察配方，用於顯示
        active_recipe = self.model_recipes[self.primary_policy_name]
        self.obs_builder.set_recipe(active_recipe) # 設定觀察產生器使用主要模型的配方
        if self.overlay:
            self.overlay.set_recipe(active_recipe) # 設定除錯介面也使用主要模型的配方

        # 為每個模型初始化獨立的觀察歷史佇列
        for name in self.model_names:
            recipe = self.model_recipes[name]
            self.obs_builder.set_recipe(recipe) # 臨時設定以計算維度
            base_obs_dim = len(self.obs_builder.get_observation(np.zeros(3), np.zeros(self.config.num_motors)))
            history_length = self.model_history_lengths[name]
            
            # 為每個模型建立一個固定長度的雙向佇列 (deque)
            self.obs_histories[name] = deque(
                [np.zeros(base_obs_dim, dtype=np.float32)] * history_length,
                maxlen=history_length
            )
        
        # 恢復 obs_builder 為主要模型的配方
        self.obs_builder.set_recipe(active_recipe)
        self.is_transitioning = False # 強制停止任何正在進行的轉換
        print(f"✅ 所有策略狀態已重置。主要模型: '{self.primary_policy_name}'。")