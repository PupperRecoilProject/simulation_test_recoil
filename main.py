import sys
import time
import numpy as np
import mujoco
import glfw
import OpenGL.GL as gl

from config import load_config
from state import SimulationState
from robot_platform import SimulationPlatform
from policy import PolicyManager
from observation import ObservationBuilder
from keyboard_input_handler import KeyboardInputHandler
from xbox_input_handler import XboxInputHandler
from floating_controller import FloatingController
from serial_communicator import SerialCommunicator
from terrain_manager import TerrainManager
from hardware_controller import HardwareController
from gui_manager import GuiManager


def main():
    """Program entry point."""
    print("\n--- 機器人控制器 v2.0 (抽象平台與GUI) ---")

    # Initialize core components
    config = load_config()
    state = SimulationState(config)

    platform = SimulationPlatform(config)
    platform.setup()
    sim = platform.sim
    state.sim = sim

    # Initialize managers
    terrain_manager = TerrainManager(sim.model, sim.data)
    state.terrain_manager_ref = terrain_manager

    floating_controller = FloatingController(config, sim.model, sim.data, terrain_manager)
    state.floating_controller_ref = floating_controller

    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm

    xbox_handler = XboxInputHandler(state)

    obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, platform.default_pose, config)
    policy_manager = PolicyManager(config, obs_builder, None)
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names

    hw_controller = HardwareController(config, policy_manager, state, serial_comm)
    state.hardware_controller_ref = hw_controller

    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    keyboard_handler.register_callbacks(platform.window)

    gui = GuiManager(platform.window)

    # Reset helpers
    def hard_reset():
        print("\n--- 正在執行機器人硬重置 (R Key) ---")
        if state.control_mode == "HARDWARE_MODE":
            return
        mujoco.mj_resetData(sim.model, sim.data)
        sim.data.qpos[0], sim.data.qpos[1] = 0, 0
        start_ground_z = terrain_manager.get_height_at(0, 0)
        sim.data.qpos[2] = start_ground_z + 0.3
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        sim.data.qpos[7:] = platform.default_pose
        sim.data.qvel[:] = 0
        sim.data.ctrl[:] = platform.default_pose
        for _ in range(10):
            mujoco.mj_step(sim.model, sim.data)
        policy_manager.reset()
        if state.control_mode == "FLOATING":
            state.set_control_mode("WALKING")
        state.reset_control_state(sim.data.time)
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        state.hard_reset_requested = False
        mujoco.mj_forward(sim.model, sim.data)

    def soft_reset():
        print("\n--- 正在執行空中姿態重置 (X Key) ---")
        if state.control_mode == "HARDWARE_MODE":
            return
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        sim.data.qpos[7:] = platform.default_pose
        sim.data.qvel[:] = 0
        policy_manager.reset()
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        mujoco.mj_forward(sim.model, sim.data)
        state.soft_reset_requested = False

    # Startup
    if terrain_manager.is_functional:
        terrain_manager.initial_generate()
    hard_reset()

    print("\n--- Simulation Started ---")

    while not platform.should_close():
        time_start = time.time()

        if state.input_mode == "GAMEPAD":
            xbox_handler.update_state()
        if state.hard_reset_requested:
            hard_reset()
        if state.soft_reset_requested:
            soft_reset()

        robot_state = platform.get_robot_state()
        state.latest_pos = robot_state.get('pos', state.latest_pos)
        state.latest_quat = robot_state.get('quat', state.latest_quat)

        if terrain_manager.is_functional:
            terrain_manager.update(state.latest_pos, state.terrain_mode)

        if state.control_mode == "HARDWARE_MODE":
            if hw_controller.is_running:
                with hw_controller.lock:
                    t_since_update = time.time() - hw_controller.hw_state.last_update_time
                    conn_status = (
                        f"Data Delay: {t_since_update:.2f}s" if t_since_update < 1.0 else "Data Timeout!"
                    )
                    state.hardware_status_text = f"Connection Status: {conn_status}\n"
                    state.hardware_status_text += f"LinVel: {np.array2string(hw_controller.hw_state.lin_vel_local, precision=2)}\n"
                    state.hardware_status_text += f"Gyro: {np.array2string(hw_controller.hw_state.imu_gyro_radps, precision=2)}"
            else:
                state.hardware_status_text = "Hardware controller not running."
        elif state.control_mode == "SERIAL_MODE":
            if state.serial_is_connected:
                state.serial_latest_messages = serial_comm.get_latest_messages()
            if state.serial_command_to_send:
                serial_comm.send_command(state.serial_command_to_send)
                state.serial_command_to_send = ""
        else:
            onnx_input, action_final = policy_manager.get_action(state.command)
            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_final

            if state.control_mode == "MANUAL_CTRL":
                final_ctrl = state.manual_final_ctrl.copy()
                platform.apply_action(final_ctrl - platform.default_pose, state.tuning_params)
            elif state.control_mode == "JOINT_TEST":
                final_ctrl = platform.default_pose + state.joint_test_offsets
                platform.apply_action(state.joint_test_offsets, state.tuning_params)
            else:
                platform.apply_action(action_final, state.tuning_params)

            target_time = sim.data.time + config.control_dt
            while sim.data.time < target_time:
                platform.step()

        # Render simulation to FBO
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, gui.fbo)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        viewport = mujoco.MjrRect(0, 0, *gui.sim_panel_size)
        platform.sim.render_scene(viewport)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        gui.start_frame()
        gui.render_gui(state)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gui.render_frame()

        glfw.swap_buffers(platform.window)
        glfw.poll_events()

        loop_dur = time.time() - time_start
        if loop_dur < config.control_dt:
            time.sleep(config.control_dt - loop_dur)

    hw_controller.stop()
    gui.shutdown()
    platform.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n程式已安全退出。")


if __name__ == "__main__":
    main()
