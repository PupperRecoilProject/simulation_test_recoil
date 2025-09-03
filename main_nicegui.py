# main_nicegui.py
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
from src.simulation.observation_manager import ObservationManager

def create_simulation_components(use_sim: bool, config, state: 'SimulationState'):
    if use_sim:
        log.info("✅ Simulation mode enabled.")
        from src.simulation.simulation import Simulation
        from src.simulation.terrain_manager import TerrainManager
        from src.simulation.floating_controller import FloatingController
        sim = Simulation(config)
        terrain = TerrainManager(config, sim.model, sim.data)
        floating = FloatingController(config, sim.model, sim.data, terrain)
        obs_manager = ObservationManager(state)
        return sim, obs_manager, terrain, floating
    else:
        log.info("🚫 Simulation disabled, using mock components.")
        from src.mock.mock_simulation import (
            MockSimulation, MockObservationManager,
            MockTerrainManager, MockFloatingController,
        )
        sim = MockSimulation(config)
        terrain = MockTerrainManager()
        floating = MockFloatingController()
        obs_manager = MockObservationManager()
        return sim, obs_manager, terrain, floating

# 【整合】定義全域變數來持有子程序的參考
nanoowl_process = None

def main() -> None:
    global nanoowl_process
    
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
    
    # 【整合】啟動 NanoOwl 影像伺服器子程序
    try:
        log.info("正在背景啟動 NanoOwl 影像伺服器...")
        command = [sys.executable, "tree_demo_server.py"]
        nanoowl_process = subprocess.Popen(command)
        log.info(f"✅ NanoOwl 伺服器子程序已啟動 (PID: {nanoowl_process.pid})。")
    except FileNotFoundError:
        log.error("❌ 錯誤: 找不到 tree_demo_server.py。請確保它在專案根目錄下。")
    except Exception as e:
        log.error(f"❌ 啟動 NanoOwl 伺服器失敗: {e}")

    # --- 核心組件裝配 (與之前版本相同) ---
    sim, observation_manager, terrain_manager, floating_controller = create_simulation_components(use_sim, config, state)
    state.sim = sim
    state.terrain_manager_ref = terrain_manager
    state.floating_controller_ref = floating_controller
    state.observation_manager_ref = observation_manager
    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm
    xbox_handler = XboxInputHandler(state)
    state.xbox_handler_ref = xbox_handler
    policy_manager = PolicyManager(config, observation_manager, None, state)
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names
    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    state.hardware_controller_ref = hw_controller
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    sim.register_callbacks(keyboard_handler)
    simulation_controller = SimulationController(state)
    ui_controller = UIController(state)

    # --- 背景執行緒與資源清理設定 ---
    def start_background_threads() -> None:
        log.info("NiceGUI 已啟動，啟動背景執行緒...")
        simulation_controller.start()
        xbox_handler.start()
        # 【整合】在客戶端連接時注入 JavaScript
        app.on_connect(ui_controller.inject_websocket_script)

    def cleanup_resources() -> None:
        log.info("NiceGUI 正在關閉，釋放資源...")
        # 【整合】終止影像伺服器子程序
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
        
        # ... (其他資源的清理邏輯保持不變) ...
        if simulation_controller: simulation_controller.stop()
        if xbox_handler: xbox_handler.close()
        if hw_controller: hw_controller.shutdown()
        if serial_comm: serial_comm.close()
        log.info("✅ 所有資源已釋放。")

    app.on_startup(start_background_threads)
    app.on_shutdown(cleanup_resources)

    # --- 啟動 UI ---
    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。")
    try:
        ui.run(title="Pupper Robot Console", port=8080, reload=False)
    except KeyboardInterrupt:
        print("\n👋 收到終止信號，正在優雅退出...")
    except Exception as e:
        log.error(f"UI 運行時發生未知錯誤: {e}", exc_info=True)
        cleanup_resources()

if __name__ == "__main__":
    main()