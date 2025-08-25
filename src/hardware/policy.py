# policy.py
import numpy as np
import onnxruntime as ort
import sys
import os
import time
from collections import deque
from typing import TYPE_CHECKING, List, Dict
# 【v4.5.3 最終權威修正】 導入 log 物件以進行時序檢驗
from src.core.logger import log

# 為了型別提示，避免循環匯入
if TYPE_CHECKING:
    from src.core.config import AppConfig
    # 【v4.3.2 刪除】 移除舊的 ObservationBuilder 類型提示
    # from src.simulation.observation import ObservationBuilder
    # 【v4.3.2 新增】 導入新的 ObservationManager 類型提示
    from src.simulation.observation_manager import ObservationManager
    from src.simulation.rendering import DebugOverlay

class PolicyManager:
    """
    【版本 2.0】 - AI 策略大腦
    管理多個 ONNX 策略模型。它能夠：
    1. 在啟動時載入所有在 config.yaml 中定義的模型。
    2. 為每個模型維護獨立的觀察歷史，以支援需要多幀輸入的模型。
    3. 根據使用者指令，在兩個不同的策略模型之間進行平滑的線性融合（插值）。
    4. 提供獨立的介面供模擬 (`get_action`) 和實體硬體 (`get_action_for_hardware`) 使用。
    """
    # 【v4.3.2 修改】 構造函式簽名，將 obs_builder 替換為 observation_manager
    def __init__(self, config: 'AppConfig', observation_manager: 'ObservationManager', overlay: 'DebugOverlay'):
        self.config = config # 儲存應用程式的全域設定
        # 【v4.3.2 修改】 更新對觀測管理器的參考
        self.observation_manager = observation_manager # 儲存觀測管理器的新參考
        self.overlay = overlay # 儲存除錯介面(DebugOverlay)的參考
        self.sessions: Dict[str, ort.InferenceSession] = {} # 字典，儲存已載入的 ONNX 推論 session，鍵為模型名稱
        self.model_recipes: Dict[str, List[str]] = {} # 字典，儲存每個模型對應的觀察配方
        self.model_history_lengths: Dict[str, int] = {} # 字典，儲存每個模型需要的歷史觀察幀數
        self.model_names: List[str] = [] # 列表，儲存所有成功載入模型的名稱
        
        print("--- 正在載入所有 ONNX 模型及其配方 ---")
        # 遍歷設定檔中定義的所有模型
        for name, model_info in config.onnx_models.items():
            path = model_info.get('path') # 獲取模型檔案路徑
            recipe = model_info.get('observation_recipe') # 獲取模型對應的觀察配方

            # 如果模型資訊不完整，則跳過
            if not path or not recipe:
                print(f"    ⚠️ 警告: 模型 '{name}' 缺少 'path' 或 'observation_recipe'，已跳過。")
                continue

            print(f"  - 載入模型 '{name}' 從: {path}")
            try:
                # --- ONNX Runtime 優化與載入 ---
                sess_options = ort.SessionOptions() # 建立 ONNX Runtime 的 session 設定
                # 定義優化後模型的快取檔案路徑，例如 "model.onnx" -> "model.optimized.ort"
                cache_path = os.path.splitext(path)[0] + ".optimized.ort"
                sess_options.optimized_model_filepath = cache_path # 設定快取路徑
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 啟用所有圖優化
                
                # 載入 session。如果 .ort 快取檔案已存在且最新，會直接載入；否則會進行優化並生成快取檔
                session = ort.InferenceSession(path, sess_options=sess_options, providers=['CPUExecutionProvider'])

                # --- 推斷模型輸入維度和歷史長度 ---
                # 【v4.3.2 修改】 根據新的 ObservationManager 來計算基礎觀測維度
                # 不再需要調用 set_recipe 或 get_observation
                base_obs_dim = 0
                for comp_name in recipe:
                    # 從 ObservationManager 的 ALL_OBS_DIMS 獲取維度
                    dim = self.observation_manager.ALL_OBS_DIMS.get(comp_name)
                    if dim is None:
                        print(f"    ⚠️ 警告: 配方中包含未知元件 '{comp_name}'。")
                        continue
                    base_obs_dim += dim

                # 從模型本身獲取其輸入層的總維度
                model_input_dim = session.get_inputs()[0].shape[1]
                
                # 計算模型需要的歷史幀數 (history length)
                history_len = 1 # 預設為1
                if base_obs_dim > 0 and model_input_dim % base_obs_dim == 0:
                    history_len = model_input_dim // base_obs_dim
                
                # 儲存該模型的所有相關資訊
                self.sessions[name] = session
                self.model_recipes[name] = recipe
                self.model_history_lengths[name] = history_len
                self.model_names.append(name)
                print(f"    > 配方: {recipe}")
                print(f"    > 基礎維度: {base_obs_dim}, 模型輸入: {model_input_dim}, 推斷歷史長度: {history_len}")

            except Exception as e:
                print(f"    ❌ 錯誤: 無法載入模型 '{name}'。錯誤: {e}")

        # 如果沒有任何模型成功載入，則終止程式
        if not self.sessions:
            sys.exit("❌ 致命錯誤: 未能成功載入任何 ONNX 模型。")

        # --- 狀態變數，用於管理多模型融合 ---
        self.primary_policy_name = self.model_names[0] # 當前穩定的主要策略，預設為第一個載入的模型
        self.source_policy_name = self.model_names[0]  # 開始轉換時的來源策略
        self.target_policy_name = self.model_names[0]  # 正在轉換去的目標策略
        
        self.last_action = np.zeros(config.num_motors, dtype=np.float32) # 初始化上一次的動作向量
        
        # 為每個模型維護一個獨立的觀察歷史佇列
        self.obs_histories: Dict[str, deque] = {}
        
        self.is_transitioning = False # 是否正在進行模型融合的旗標
        self.transition_start_time = 0.0 # 融合開始的時間戳
        self.transition_alpha = 0.0 # 線性融合的權重 (0.0=source, 1.0=target)

        self.reset() # 初始化所有模型的觀察歷史

        print("--- 正在預熱所有 ONNX 模型 (強制進行首次推論優化)... ---")
        # 遍歷所有載入的 session
        for name, session in self.sessions.items():
            input_name = session.get_inputs()[0].name # 獲取輸入層名稱
            output_name = session.get_outputs()[0].name # 獲取輸出層名稱
            model_input_dim = session.get_inputs()[0].shape[1] # 獲取輸入維度
            dummy_input = np.zeros((1, model_input_dim), dtype=np.float32) # 建立一個符合維度的假輸入
            try:
                # 執行一次推論以觸發可能的 JIT 編譯或優化
                session.run([output_name], {input_name: dummy_input})
                print(f"  - 模型 '{name}' 預熱成功。")
            except Exception as e:
                print(f"  - ⚠️ 模型 '{name}' 預熱失敗: {e}")

        print(f"✅ 策略管理器初始化完成，主要模型: '{self.primary_policy_name}'")

    # 【v4.3.2 修改】 get_active_recipe 方法 (僅修改註解)
    def get_active_recipe(self) -> List[str]:
        """一個輔助函式，返回當前主要策略所使用的觀察配方，主要供 DebugOverlay 使用。"""
        # 【v4.3.2 修改】 統一註解
        return self.model_recipes.get(self.primary_policy_name, [])

    # 【v4.3.2 修改】 select_target_policy 方法 (僅修改註解)
    def select_target_policy(self, target_name: str):
        """(由鍵盤/UI觸發) 選擇一個目標策略並開始平滑轉換。"""
        # 【v4.3.2 修改】 統一註解
        # 檢查目標模型是否存在
        if target_name not in self.sessions:
            print(f"⚠️ 警告: 無法切換，目標模型 '{target_name}' 不存在。")
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

    # 【v4.3.2 修改】 get_action 方法
    def get_action(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        【v4.3.2 修改】
        【模擬專用】獲取最終動作。此版本透過 ObservationManager 獲取觀測數據，運行所有模型，並根據狀態進行融合。
        此函式不再負責建構觀測數據，而是依賴 PolicyManager 初始化時傳入的 ObservationManager 實例。
        """
        all_actions = {} # 建立一個字典來儲存本幀所有模型的輸出
        primary_onnx_input = np.array([]) # 用於除錯顯示的輸入

        # --- 步驟 1: 運行所有模型，獲取各自的輸出 ---
        for name, session in self.sessions.items():
            recipe = self.model_recipes[name]
            
            # 【v4.3.2 修改】 設定 ObservationManager 的配方並獲取觀測數據
            # 在獲取數據前，PolicyManager 負責通知 ObservationManager 當前策略需要哪些數據格式。
            self.observation_manager.set_recipe(recipe)
            base_obs = self.observation_manager.get_observation()
            
            # 【時序檢驗探針 2】 記錄 AI 決策時的時間戳
            decision_time = time.perf_counter()
            log.info(f"[PolicyManager] AI 決策 @ {decision_time:.6f}")
            
            self.obs_histories[name].append(base_obs)
            
            # 將歷史觀察拼接成 ONNX 的最終輸入
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            
            # 檢查維度是否匹配，以防萬一
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                # 【v4.3.2 修改】 統一註解
                print(f"⚠️ 警告: 模型 '{name}' 輸入維度不匹配，預期 {session.get_inputs()[0].shape[1]} 但得到 {onnx_input.shape[1]}。將返回零動作。")
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
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
            
            # 線性計算 alpha，並限制在 [0, 1] 範圍
            if duration > 0: self.transition_alpha = min(elapsed / duration, 1.0)
            else: self.transition_alpha = 1.0

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

    # 【v4.3.2 修改】 get_action_for_hardware 方法
    def get_action_for_hardware(self) -> tuple[np.ndarray, np.ndarray]:
        """
        【v4.3.2 修改】
        【硬體專用】獲取最終動作。此版本不再接收外部觀測，而是從 ObservationManager 獲取。
        HardwareController 會將原始數據寫入 SimulationState，ObservationManager 從那裡讀取。
        """
        all_actions = {}
        primary_onnx_input = np.array([])

        for name, session in self.sessions.items():
            recipe = self.model_recipes[name]
            
            # 【v4.3.2 修改】 設定 ObservationManager 的配方並獲取觀測數據
            # PolicyManager 通知 ObservationManager 當前策略需要哪些數據格式。
            self.observation_manager.set_recipe(recipe)
            base_obs = self.observation_manager.get_observation()
            
            # 硬體模式下， ObservationManager 已經從 state.raw_... 建立了數據，我們將其添加到歷史中
            self.obs_histories[name].append(base_obs)
            
            onnx_input = np.concatenate(list(self.obs_histories[name])).astype(np.float32).reshape(1, -1)
            
            # 【v4.3.2 修改】 統一註解
            if onnx_input.shape[1] != session.get_inputs()[0].shape[1]:
                print(f"⚠️ 警告: 模型 '{name}' 輸入維度不匹配，預期 {session.get_inputs()[0].shape[1]} 但得到 {onnx_input.shape[1]}。將返回零動作。")
                action_raw = np.zeros(self.config.num_motors, dtype=np.float32)
            else:
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                action_raw = session.run([output_name], {input_name: onnx_input})[0].flatten()
            
            all_actions[name] = action_raw

            if name == self.primary_policy_name:
                primary_onnx_input = onnx_input

        # 融合邏輯與 get_action 完全相同
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

        self.last_action[:] = final_action # last_action 也需要為硬體模式更新
        return primary_onnx_input, final_action

    # 【v4.3.2 修改】 reset 方法
    def reset(self):
        """
        【v4.3.2 修改】
        重置所有模型的觀察歷史，並同步更新除錯介面和 ObservationManager。
        """
        # 重置主模型的觀察配方，PolicyManager 通知 ObservationManager 當前策略需要哪些數據格式。
        active_recipe = self.model_recipes[self.primary_policy_name]
        self.observation_manager.set_recipe(active_recipe)
        
        if self.overlay:
            self.overlay.set_recipe(active_recipe)

        # 為每個模型初始化獨立的、填滿零的觀察歷史佇列
        for name in self.model_names:
            recipe = self.model_recipes[name]
            # 【v4.3.2 修改】 根據新的 ObservationManager 來計算基礎觀測維度
            # 不再需要調用 set_recipe
            base_obs_dim = 0
            for comp_name in recipe:
                # 從 ObservationManager 的 ALL_OBS_DIMS 獲取維度
                dim = self.observation_manager.ALL_OBS_DIMS.get(comp_name)
                if dim is None:
                    # 這裡可以選擇打印警告或拋出錯誤，取決於嚴格程度
                    continue
                base_obs_dim += dim
            
            history_length = self.model_history_lengths[name]
            
            # 建立一個固定長度的雙向佇列 (deque)，當新元素加入時，舊元素會自動被擠出
            self.obs_histories[name] = deque(
                [np.zeros(base_obs_dim, dtype=np.float32)] * history_length,
                maxlen=history_length
            )
        
        # 恢復 PolicyManager 的 observation_manager 為主要模型的配方，以供後續使用
        self.observation_manager.set_recipe(active_recipe)
        
        self.is_transitioning = False # 強制停止任何正在進行的轉換
        print(f"✅ 所有策略狀態已重置。主要模型: '{self.primary_policy_name}'。")
