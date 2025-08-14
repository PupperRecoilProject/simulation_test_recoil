"""Entry point using NiceGUI for the control interface."""

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


def create_simulation_components(use_sim: bool, config):
    """根據是否使用模擬，建立對應的模組實例。"""
    if use_sim:
        log.info("✅ Simulation mode enabled.")
        try:
            # 匯入需要的 MuJoCo 模擬元件
            from src.simulation.simulation import Simulation
            from src.simulation.observation import ObservationBuilder
            from src.simulation.terrain_manager import TerrainManager
            from src.simulation.floating_controller import FloatingController
        except ModuleNotFoundError as exc:
            # 若系統未安裝 MuJoCo，提示並退回至 no-sim 模式
            if exc.name == "mujoco":
                log.warning("⚠️ 找不到 MuJoCo 模組，自動切換為 no-sim 模式。")
                return create_simulation_components(False, config)
            raise

        sim = Simulation(config)
        terrain = TerrainManager(sim.model, sim.data)
        floating = FloatingController(config, sim.model, sim.data, terrain)
        obs = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)
        return sim, obs, terrain, floating
    else:
        log.info("🚫 Simulation disabled, using mock components.")
        from src.mock.mock_simulation import (
            MockSimulation,
            MockObservationBuilder,
            MockTerrainManager,
            MockFloatingController,
        )

        sim = MockSimulation(config)
        terrain = MockTerrainManager()
        floating = MockFloatingController()
        obs = MockObservationBuilder()
        return sim, obs, terrain, floating


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
    sim, obs_builder, terrain_manager, floating_controller = create_simulation_components(use_sim, config)

    # 將核心物件的參考存入 state，使其成為全域上下文
    state.sim = sim
    state.terrain_manager_ref = terrain_manager
    state.floating_controller_ref = floating_controller

    # 按照依賴順序初始化所有管理器
    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm

    xbox_handler = XboxInputHandler(state)
    state.xbox_handler_ref = xbox_handler

    policy_manager = PolicyManager(config, obs_builder, None) # 在 NiceGUI 模式下，overlay 設為 None
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names

    # 初始化 HardwareController，它不再依賴 state
    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    state.hardware_controller_ref = hw_controller

    # 初始化 KeyboardInputHandler
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    sim.register_callbacks(keyboard_handler)

    # 初始化中央調度器與 UI 控制器
    simulation_controller = SimulationController(state)

    # 根據 UIController 最新的 __init__(self, state) 定義，
    # 移除多餘的 hw_controller 參數。
    # UIController 現在只依賴 state，所有它需要知道的硬體狀態
    # (如 is_running) 都應該由 SimulationController 更新到 state 中。
    ui_controller = UIController(state)

    # --- 背景執行緒與資源清理設定 ---
    def start_background_threads() -> None:
        log.info("NiceGUI 已啟動，啟動背景執行緒...")
        simulation_controller.start()
        xbox_handler.start()

    def cleanup_resources() -> None:
        log.info("NiceGUI 正在關閉，釋放資源...")
        # 步驟 1: 停止 simulation_controller 的主迴圈
        simulation_controller.stop()

        # 步驟 2: 【v4.0.2 修正】呼叫新的、為關閉而設計的 shutdown 方法
        # 舊: hw_controller.stop_controller_threads()
        hw_controller.shutdown()

        # 步驟 3: 依次關閉其他資源
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
