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
import subprocess
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
# 【v4.3.2 新增】 導入新的 ObservationManager
from src.simulation.observation_manager import ObservationManager


# 【v4.3.2 修改】 create_simulation_components 函式
def create_simulation_components(use_sim: bool, config, state: 'SimulationState'): # 【v4.3.2 新增】 傳入 state
    """根據是否使用模擬，建立對應的模組實例。"""
    if use_sim:
        log.info("✅ Simulation mode enabled.")
        from src.simulation.simulation import Simulation
        # 【v4.3.2 刪除】 移除舊的 ObservationBuilder
        # from src.simulation.observation import ObservationBuilder
        from src.simulation.terrain_manager import TerrainManager
        from src.simulation.floating_controller import FloatingController

        sim = Simulation(config)
        # 【v4.10.0 修改】將完整的 config 物件傳入 TerrainManager
        terrain = TerrainManager(config, sim.model, sim.data)
        floating = FloatingController(config, sim.model, sim.data, terrain)
        # 【v4.3.2 修改】 實例化 ObservationManager
        obs_manager = ObservationManager(state)
        return sim, obs_manager, terrain, floating
    else:
        log.info("🚫 Simulation disabled, using mock components.")
        from src.mock.mock_simulation import (
            MockSimulation,
            # 【v4.3.2 修改】 導入 MockObservationManager
            MockObservationManager,
            MockTerrainManager,
            MockFloatingController,
        )

        sim = MockSimulation(config)
        terrain = MockTerrainManager()
        floating = MockFloatingController()
        # 【v4.3.2 修改】 實例化 MockObservationManager
        obs_manager = MockObservationManager()
        return sim, obs_manager, terrain, floating

# 【整合 NanoOwl】定義全域變數來持有子程序的參考
nanoowl_process = None

# 【整合 NanoOwl】清理子程序的函式 (從您提供的檔案中複製)
def nanoowlcleanup_subprocess():
    """一個專門用來在程式退出時清理子程序的函式。"""
    global nanoowl_process
    if nanoowl_process and nanoowl_process.poll() is None:
        log.info(f"Cleanup: 正在終止 NanoOwl 伺服器子程序 (PID: {nanoowl_process.pid})...")
        nanoowl_process.terminate()
        try:
            nanoowl_process.wait(timeout=2)
            log.info("Cleanup: ✅ NanoOwl 子程序已終止。")
        except subprocess.TimeoutExpired:
            log.warning("Cleanup: NanoOwl 子程序在2秒內未終止，將強制 kill。")
            nanoowl_process.kill()

# 【v4.3.2 修改】 main 函式
def main() -> None:
    global nanoowl_process # <--- 宣告我們要修改全域變數
    
    """Initialise all components and start UI and simulation threads."""

    parser = argparse.ArgumentParser(description="Pupper Robot Controller")
    parser.add_argument("--no-sim", action="store_true", help="run without MuJoCo simulation")
    args = parser.parse_args()

    use_sim = not args.no_sim

    print("\n--- Robot Simulation Controller (NiceGUI edition) ---")
    if not use_sim:
        print("========= RUNNING IN NO-SIM MODE =========")

    try:
        config = load_config()
        state = SimulationState(config)
    except Exception as exc:
        sys.exit(f"failed to initialise: {exc}")
    
    # ==========================================================
    # 【整合 NanoOwl】在啟動 NiceGUI 之前，先啟動影像伺服器子程序
    # ==========================================================
    try:
        log.info("正在背景啟動 NanoOwl 影像伺服器...")
        # 使用 sys.executable 確保用的是同一個 Python 解譯器
        command = [sys.executable, "tree_demo_server.py"]
        nanoowl_process = subprocess.Popen(command)
        log.info(f"✅ NanoOwl 伺服器子程序已啟動 (PID: {nanoowl_process.pid})。")
    except FileNotFoundError:
        log.error("❌ 錯誤: 找不到 tree_demo_server.py。請確保它在專案根目錄下。")
    except Exception as e:
        log.error(f"❌ 啟動 NanoOwl 伺服器失敗: {e}")
    # ==========================================================

    # --- 核心組件裝配 ---
    # 【v4.3.2 修改】 更新變數名，並傳入 state
    sim, observation_manager, terrain_manager, floating_controller = create_simulation_components(use_sim, config, state)

    # 將核心物件的參考存入 state，使其成為全域上下文
    state.sim = sim
    state.terrain_manager_ref = terrain_manager
    state.floating_controller_ref = floating_controller
    # 【v4.3.2 新增】 將 observation_manager 存入 state
    state.observation_manager_ref = observation_manager

    # 按照依賴順序初始化所有管理器
    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm

    xbox_handler = XboxInputHandler(state)
    state.xbox_handler_ref = xbox_handler

    # 【v4.3.2 修改】 將 observation_manager 傳入 PolicyManager
    # 【v4.4.7 修改】 新增 state 參數
    policy_manager = PolicyManager(config, observation_manager, None, state) # 在 NiceGUI 模式下，overlay 設為 None
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names

    # 初始化 HardwareController
    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    state.hardware_controller_ref = hw_controller

    # 初始化 KeyboardInputHandler
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    sim.register_callbacks(keyboard_handler)

    # 初始化中央調度器與 UI 控制器
    simulation_controller = SimulationController(state)
    ui_controller = UIController(state)

    # --- 背景執行緒與資源清理設定 ---
    def start_background_threads() -> None:
        log.info("NiceGUI 已啟動，啟動背景執行緒...")
        simulation_controller.start()
        xbox_handler.start()
        
        # 【整合 NanoOwl】在客戶端連接時才注入 JavaScript
        # 我們將 ui_controller 的新方法註冊到 on_connect 事件上
        #app.on_connect(ui_controller.inject_websocket_script)

    def cleanup_resources() -> None:
        log.info("NiceGUI 正在關閉，釋放資源...")
        
        # 【整合 NanoOwl】終止影像伺服器子程序
        # 【修正】直接調用清理函式
        nanoowlcleanup_subprocess()
        
        # 停止所有背景執行緒
        if simulation_controller:
            simulation_controller.stop()
        #if rendering_thread:
        #    rendering_thread.stop()
        #    rendering_thread.join(timeout=2) # 等待渲染執行緒結束

        # 關閉其他資源
        if xbox_handler:
            xbox_handler.close()
        if hw_controller:
            hw_controller.shutdown()
        if serial_comm:
            serial_comm.close()

        # 確保 sim.close() 已經被移除！
        # sim.close() # <--- 刪除或註解掉這一行

        log.info("✅ 所有資源已釋放。")

    app.on_startup(start_background_threads)
    app.on_shutdown(cleanup_resources)

    # --- 啟動 UI ---
    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。")
    try:
        ui.run(title="Pupper Robot Console", port=8080, reload=False)
    except KeyboardInterrupt:
        # 當用戶按下 Ctrl+C 或關閉視窗時，Uvicorn 會拋出 KeyboardInterrupt。
        # 我們在這裡捕獲它，以防止它在終端顯示為一個 "錯誤"。
        # app.on_shutdown 註冊的 cleanup_resources 會被自動調用，
        # 我們只需要在這裡提供一個乾淨的退出點即可。
        print("\n👋 收到終止信號，正在優雅退出...")
    except Exception as e:
        # 捕獲其他可能的異常
        log.error(f"UI 運行時發生未知錯誤: {e}", exc_info=True)
        # 即使 UI 出錯，我們也嘗試執行清理
        cleanup_resources()


if __name__ == "__main__":
    main()