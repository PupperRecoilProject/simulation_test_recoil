"""Entry point using NiceGUI for the control interface."""

import sys
import threading  # 新增：用於啟動背景執行緒
from nicegui import ui, app  # 從 nicegui 匯入 ui 物件與 app 實例

from config import load_config
from state import SimulationState
from simulation import Simulation
from simulation_controller import SimulationController
from ui_controller import UIController
from policy import PolicyManager
from observation import ObservationBuilder
from floating_controller import FloatingController
from serial_communicator import SerialCommunicator
from terrain_manager import TerrainManager
from hardware_controller import HardwareController
from keyboard_input_handler import KeyboardInputHandler
from xbox_input_handler import XboxInputHandler


def main() -> None:
    """Initialise all components and start UI and simulation threads."""

    print("\n--- Robot Simulation Controller (NiceGUI edition) ---")

    try:
        config = load_config()
        state = SimulationState(config)
        sim = Simulation(config)
    except Exception as exc:  # pragma: no cover - startup errors
        sys.exit(f"failed to initialise: {exc}")

    state.sim = sim

    terrain_manager = TerrainManager(sim.model, sim.data)
    state.terrain_manager_ref = terrain_manager

    floating_controller = FloatingController(config, sim.model, sim.data, terrain_manager)
    state.floating_controller_ref = floating_controller

    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm

    xbox_handler = XboxInputHandler(state)
    state.xbox_handler_ref = xbox_handler

    obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)

    policy_manager = PolicyManager(config, obs_builder, None)
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names

    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    state.hardware_controller_ref = hw_controller

    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    sim.register_callbacks(keyboard_handler)

    simulation_controller = SimulationController(state)
    ui_controller = UIController(state)

    def start_background_threads() -> None:
        """UI 啟動後啟動所有背景執行緒。"""
        print("NiceGUI 已啟動，現在安全地啟動背景執行緒...")
        simulation_controller.start()
        xbox_handler.start()

    def cleanup_resources() -> None:
        """UI 關閉時釋放所有背景資源。"""
        print("NiceGUI 正在關閉，正在清理資源...")
        simulation_controller.stop()
        # 關閉硬體控制執行緒
        hw_controller.stop_controller_threads()
        serial_comm.close()
        xbox_handler.close()
        sim.close()
        print("✅ 所有資源已成功釋放。")

    app.on_startup(start_background_threads)
    app.on_shutdown(cleanup_resources)

    print("🚀 正在啟動 NiceGUI 控制台... 請打開您的瀏覽器。")
    ui.run(
        title="Pupper Robot Console",
        port=8080,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()

