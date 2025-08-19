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
from src.utils.gamepad_presence_guard import start_gamepad_presence_guard
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
        terrain = TerrainManager(sim.model, sim.data)
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


# 【v4.3.2 修改】 main 函式
def main() -> None:
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
        if config.use_virtual_teensy:
            # 🔌 虛擬Teensy模式：不需要實體序列埠也能進入硬體模式
            state.serial_is_connected = True
            log.info("虛擬Teensy模式啟用，跳過序列埠連線檢查。")
    except Exception as exc:
        sys.exit(f"failed to initialise: {exc}")

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
    # 啟動搖桿存在守門員，只用於 UI 顯示，不影響 handler
    start_gamepad_presence_guard(state)

    # 【v4.3.2 修改】 將 observation_manager 傳入 PolicyManager
    policy_manager = PolicyManager(config, observation_manager, None) # 在 NiceGUI 模式下，overlay 設為 None
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

    def cleanup_resources() -> None:
        log.info("NiceGUI 正在關閉，釋放資源...")
        simulation_controller.stop()
        hw_controller.shutdown()
        serial_comm.close()
        xbox_handler.close()
        sim.close()
        log.info("✅ All resources released.")

    app.on_startup(start_background_threads)
    app.on_shutdown(cleanup_resources)

    # --- 啟動 UI ---
    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。")
    ui.run(title="Pupper Robot Console", port=8080)


if __name__ in {"__main__", "__mp_main__"}:
    main()
