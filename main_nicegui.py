"""Entry point using NiceGUI for the control interface."""

import sys
import argparse
from nicegui import ui, app

from utils.config import load_config
from state import SimulationState
from core.policy import PolicyManager
from controllers.hardware_controller import HardwareController
from serial_communicator import SerialCommunicator
from inputs.xbox_input_handler import XboxInputHandler
from ui_controller import UIController
from controllers.simulation_controller import SimulationController
from inputs.keyboard_input_handler import KeyboardInputHandler
from utils.logger import log


def create_simulation_components(use_sim: bool, config):
    """根據是否使用模擬，建立對應的模組實例。"""
    if use_sim:
        log.info("✅ Simulation mode enabled.")
        from core.simulation import Simulation
        from core.observation import ObservationBuilder
        from core.terrain_manager import TerrainManager
        from controllers.floating_controller import FloatingController

        sim = Simulation(config)
        terrain = TerrainManager(sim.model, sim.data)
        floating = FloatingController(config, sim.model, sim.data, terrain)
        obs = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)
        return sim, obs, terrain, floating
    else:
        log.info("🚫 Simulation disabled, using mock components.")
        from mock_simulation import (
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
    except Exception as exc:
        sys.exit(f"failed to initialise: {exc}")

    sim, obs_builder, terrain_manager, floating_controller = create_simulation_components(use_sim, config)

    # 使用 lock 確保設定參考時的執行緒安全
    with state.lock:
        state.sim = sim
        state.terrain_manager_ref = terrain_manager
        state.floating_controller_ref = floating_controller

    serial_comm = SerialCommunicator()
    with state.lock:
        state.serial_communicator_ref = serial_comm

    xbox_handler = XboxInputHandler(state)
    with state.lock:
        state.xbox_handler_ref = xbox_handler

    policy_manager = PolicyManager(config, obs_builder, None)
    with state.lock:
        state.policy_manager_ref = policy_manager
        state.available_policies = policy_manager.model_names

    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    with state.lock:
        state.hardware_controller_ref = hw_controller

    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    sim.register_callbacks(keyboard_handler)

    simulation_controller = SimulationController(state)
    ui_controller = UIController(state)

    def start_background_threads() -> None:
        log.info("NiceGUI 已啟動，啟動背景執行緒...")
        simulation_controller.start()
        xbox_handler.start()

    def cleanup_resources() -> None:
        log.info("NiceGUI 正在關閉，釋放資源...")
        simulation_controller.stop()
        hw_controller.stop()
        serial_comm.close()
        xbox_handler.close()
        sim.close()
        log.info("✅ All resources released.")

    app.on_startup(start_background_threads)
    app.on_shutdown(cleanup_resources)

    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。")
    ui.run(title="Pupper Robot Console", port=8080)


if __name__ in {"__main__", "__mp_main__"}:
    main()
