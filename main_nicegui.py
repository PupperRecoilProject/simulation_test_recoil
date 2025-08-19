# main_nicegui.py
"""
【全功能主應用程式】

此腳本是專案的主要入口點，提供了一個基於 NiceGUI 的全功能、多執行緒控制台。

主要用途:
- 日常的機器人模擬與控制。
- 透過圖形化介面進行精細的參數調校。
- 控制實體硬體，並支持無模擬的 `--no-sim` (無頭) 模式。
- 查看詳細的即時日誌和狀態監控。

架構特性:
- UI、模擬、硬體控制在各自獨立的執行緒中運行，確保了介面的高響應性。
- 使用事件驅動和中央狀態管理，實現了模組間的解耦和線程安全。

建議：
- 除非有特定的底層除錯需求，否則應始終使用此腳本作為啟動器。
"""
import sys
import argparse
from nicegui import ui, app

# --- 我們的模組導入 ---
from src.core.config import load_config
from src.core.state import SimulationState
from src.hardware.policy import PolicyManager
from src.controllers.hardware_controller import HardwareController
from src.hardware.serial_communicator import SerialCommunicator
from src.input_handlers.xbox_input_handler import XboxInputHandler
from src.controllers.ui_controller import UIController
from src.controllers.simulation_controller import SimulationController
from src.input_handlers.keyboard_input_handler import KeyboardInputHandler
from src.core.logger import log
# [合併] 同時導入 ObservationManager 和 gamepad_presence_guard
from src.simulation.observation_manager import ObservationManager
from src.utils.gamepad_presence_guard import start_gamepad_presence_guard


def create_simulation_components(use_sim: bool, config, state: 'SimulationState'): # 建立模擬元件
    """
    [合併後的版本] 根據是否使用模擬，建立對應的模組實例。
    保留了 ObservationManager 架構和 MuJoCo 不存在時的回退機制。
    """
    if use_sim: # 如果使用模擬
        log.info("✅ Simulation mode enabled.") # 記錄日誌
        try:
            # [合併] 導入最新的模擬元件
            from src.simulation.simulation import Simulation
            from src.simulation.terrain_manager import TerrainManager
            from src.simulation.floating_controller import FloatingController
            # 注意：這裡不再導入舊的 ObservationBuilder
        except ModuleNotFoundError as exc:
            # [合併] 保留 fake-Teensy 的 fallback 機制
            if exc.name == "mujoco":
                log.warning("⚠️ 找不到 MuJoCo 模組，自動切換為 no-sim 模式。")
                # 遞迴呼叫，但強制 use_sim=False
                return create_simulation_components(False, config, state)
            raise

        sim = Simulation(config) # 建立 Simulation 物件
        terrain = TerrainManager(sim.model, sim.data) # 建立 TerrainManager 物件
        floating = FloatingController(config, sim.model, sim.data, terrain) # 建立 FloatingController 物件
        obs_manager = ObservationManager(state) # [保留 dev4.3 架構] 實例化 ObservationManager
        return sim, obs_manager, terrain, floating # 回傳所有元件
    else: # 如果不使用模擬
        log.info("🚫 Simulation disabled, using mock components.") # 記錄日誌
        from src.mock.mock_simulation import (
            MockSimulation,
            MockObservationManager,
            MockTerrainManager,
            MockFloatingController,
        )

        sim = MockSimulation(config) # 建立 MockSimulation 物件
        terrain = MockTerrainManager() # 建立 MockTerrainManager 物件
        floating = MockFloatingController() # 建立 MockFloatingController 物件
        obs_manager = MockObservationManager() # [保留 dev4.3 架構] 實例化 MockObservationManager
        return sim, obs_manager, terrain, floating # 回傳所有 mock 元件


def main() -> None: # 主函式
    """Initialise all components and start UI and simulation threads."""

    parser = argparse.ArgumentParser(description="Pupper Robot Controller") # 建立命令列參數解析器
    parser.add_argument("--no-sim", action="store_true", help="run without MuJoCo simulation") # 加入 --no-sim 參數
    args = parser.parse_args() # 解析參數

    use_sim = not args.no_sim # 判斷是否使用模擬

    print("\n--- Robot Simulation Controller (NiceGUI edition) ---") # 顯示標題
    if not use_sim:
        print("========= RUNNING IN NO-SIM MODE =========") # 如果不使用模擬，顯示提示訊息

    try:
        config = load_config() # 載入設定
        state = SimulationState(config) # 建立 SimulationState 物件
        # [合併] 保留 fake-Teensy 的虛擬裝置檢查邏輯
        if config.use_virtual_teensy:
            state.serial_is_connected = True # 如果使用虛擬 Teensy，則設定序列埠已連接
            log.info("虛擬Teensy模式啟用，跳過序列埠連線檢查。") # 記錄日誌
    except Exception as exc:
        sys.exit(f"failed to initialise: {exc}") # 如果初始化失敗，則退出程式

    # --- 核心組件裝配 ---
    # [保留 dev4.3 架構] 確保使用 observation_manager
    sim, observation_manager, terrain_manager, floating_controller = create_simulation_components(use_sim, config, state)

    # 將核心物件的參考存入 state
    state.sim = sim # 將 sim 存入 state
    state.terrain_manager_ref = terrain_manager # 將 terrain_manager 存入 state
    state.floating_controller_ref = floating_controller # 將 floating_controller 存入 state
    state.observation_manager_ref = observation_manager # 將 observation_manager 存入 state

    # 按照依賴順序初始化所有管理器
    serial_comm = SerialCommunicator() # 建立 SerialCommunicator 物件
    state.serial_communicator_ref = serial_comm # 將 serial_comm 存入 state

    xbox_handler = XboxInputHandler(state) # 建立 XboxInputHandler 物件
    state.xbox_handler_ref = xbox_handler # 將 xbox_handler 存入 state
    # [合併] 保留 fake-Teensy 的搖桿狀態守衛
    start_gamepad_presence_guard(state) # 啟動搖桿存在守衛

    # [保留 dev4.3 架構] 將 observation_manager 傳入 PolicyManager
    policy_manager = PolicyManager(config, observation_manager, None)
    state.policy_manager_ref = policy_manager # 將 policy_manager 存入 state
    state.available_policies = policy_manager.model_names # 將可用的策略存入 state

    # 初始化 HardwareController
    hw_controller = HardwareController(config, policy_manager, state, serial_comm) # 建立 HardwareController 物件
    state.hardware_controller_ref = hw_controller # 將 hw_controller 存入 state

    # 初始化 KeyboardInputHandler
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager) # 建立 KeyboardInputHandler 物件
    sim.register_callbacks(keyboard_handler) # 註冊回呼函式

    # 初始化中央調度器與 UI 控制器
    simulation_controller = SimulationController(state) # 建立 SimulationController 物件
    ui_controller = UIController(state) # 建立 UIController 物件

    # --- 背景執行緒與資源清理設定 ---
    def start_background_threads() -> None: # 啟動背景執行緒
        log.info("NiceGUI 已啟動，啟動背景執行緒...") # 記錄日誌
        simulation_controller.start() # 啟動模擬控制器
        xbox_handler.start() # 啟動 xbox 處理器

    def cleanup_resources() -> None: # 清理資源
        log.info("NiceGUI 正在關閉，釋放資源...") # 記錄日誌
        simulation_controller.stop() # 停止模擬控制器
        hw_controller.shutdown() # 關閉硬體控制器
        serial_comm.close() # 關閉序列埠通訊器
        xbox_handler.close() # 關閉 xbox 處理器
        sim.close() # 關閉模擬器
        log.info("✅ All resources released.") # 記錄日誌

    app.on_startup(start_background_threads) # 設定啟動時執行的函式
    app.on_shutdown(cleanup_resources) # 設定關閉時執行的函式

    # --- 啟動 UI ---
    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。") # 顯示提示訊息
    ui.run(title="Pupper Robot Console", port=8080) # 啟動 UI


if __name__ in {"__main__", "__mp_main__"}: # 如果是主程式
    main() # 執行主函式