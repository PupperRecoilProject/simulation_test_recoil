"""Entry point using NiceGUI for the control interface."""

import sys
import threading

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

    simulation_thread = threading.Thread(target=simulation_controller.run, daemon=True)
    simulation_thread.start()

    print("simulation thread started, launching UI ...")
    ui_controller.run()

    print("closing application ...")
    simulation_controller.stop()
    hw_controller.stop()
    serial_comm.close()
    xbox_handler.close()
    sim.close()


if __name__ == "__main__":
    main()

