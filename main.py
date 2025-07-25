# main.py
import sys
import numpy as np
import mujoco
import time

from config import load_config
from state import SimulationState
from platform import SimulationPlatform
from policy import PolicyManager
from observation import ObservationBuilder
from rendering import DebugOverlay
from keyboard_input_handler import KeyboardInputHandler
from xbox_input_handler import XboxInputHandler
from floating_controller import FloatingController
from serial_communicator import SerialCommunicator
from terrain_manager import TerrainManager
from hardware_controller import HardwareController
from gui_manager import GuiManager

def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    from xbox_controller import XboxController 
    print("\n--- 機器人模擬控制器 (含硬體與多模型模式) ---")
    
    # --- 1. 初始化核心組件 ---
    config = load_config()
    state = SimulationState(config)
    platform = SimulationPlatform(config)
    sim = platform.sim

    # --- 2. 【核心修改】將核心物件的參考存入 state，使其成為全域上下文 ---
    state.sim = sim
    
    # --- 3. 按照依賴順序初始化所有管理器 ---
    terrain_manager = TerrainManager(sim.model, sim.data)
    state.terrain_manager_ref = terrain_manager
    
    floating_controller = FloatingController(config, sim.model, sim.data, terrain_manager)
    state.floating_controller_ref = floating_controller
    
    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm # 將 serial_comm 存入 state
    
    xbox_handler = XboxInputHandler(state)

    obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    overlay = DebugOverlay()
    
    policy_manager = PolicyManager(config, obs_builder, overlay)
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names
    
    # 將 serial_comm 傳入 HardwareController 的建構函式
    hw_controller = HardwareController(config, policy_manager, state, serial_comm) 
    state.hardware_controller_ref = hw_controller

    # KeyboardInputHandler 不再需要直接傳入 serial_comm
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    keyboard_handler.register_callbacks(sim.window)

    gui = GuiManager(sim.window)

    # --- 4. 定義重置函式 ---
    def hard_reset():
        print("\n--- 正在執行機器人硬重置 (R Key) ---")
        if state.control_mode == "HARDWARE_MODE": return
        mujoco.mj_resetData(sim.model, sim.data)
        sim.data.qpos[0], sim.data.qpos[1] = 0, 0
        start_ground_z = terrain_manager.get_height_at(0, 0)
        robot_height_offset = 0.3
        sim.data.qpos[2] = start_ground_z + robot_height_offset
        print(f"機器人重置至原點：地形高度({start_ground_z:.2f}m) + 偏移({robot_height_offset:.2f}m) = 世界Z({sim.data.qpos[2]:.2f}m)")
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        sim.data.qpos[7:] = sim.default_pose
        sim.data.qvel[:] = 0
        sim.data.ctrl[:] = sim.default_pose
        for _ in range(10): mujoco.mj_step(sim.model, sim.data)
        policy_manager.reset()
        if state.control_mode == "FLOATING": state.set_control_mode("WALKING")
        state.reset_control_state(sim.data.time)
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        state.hard_reset_requested = False
        mujoco.mj_forward(sim.model, sim.data)

    def soft_reset():
        print("\n--- 正在執行空中姿態重置 (X Key) ---")
        if state.control_mode == "HARDWARE_MODE": return
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0])
        sim.data.qpos[7:] = sim.default_pose
        sim.data.qvel[:] = 0
        policy_manager.reset()
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        mujoco.mj_forward(sim.model, sim.data)
        state.soft_reset_requested = False

    # --- 5. 啟動程序 ---
    if terrain_manager.is_functional:
        terrain_manager.initial_generate()
    hard_reset()
    
    # 【快捷鍵變更】更新啟動時的提示文字
    print("\n--- Simulation Started (SPACE: Pause, N: Step) ---")
    print("    (F: Float, G: Joint Test, B: Manual Ctrl, H: Hardware Mode)")
    print("    (M: Input Mode, R: Hard Reset, X: Soft Reset)")
    print("    (Y: Regen Terrain, P: Save Terrain PNG, 1..: Select Policy)")
    print("    (V: Cycle Terrain, K: Toggle HW AI)")
    print("    ( ~ : Toggle Serial Console )") # 新增此行，替換舊的 T 鍵提示

    state.execute_one_step = False

    # --- 6. 主模擬迴圈 ---
    while not sim.should_close():
        gui.start_frame()
        if state.single_step_mode and not state.execute_one_step:
            sim.render(state, overlay)
            gui.render_gui(state)
            gui.render_frame()
            continue
        if state.execute_one_step: state.execute_one_step = False

        if state.input_mode == "GAMEPAD": xbox_handler.update_state()
        if state.hard_reset_requested: hard_reset()
        if state.soft_reset_requested: soft_reset()

        state.latest_pos = sim.data.body('torso').xpos.copy()
        state.latest_quat = sim.data.body('torso').xquat.copy()
        
        if terrain_manager.is_functional:
            terrain_manager.update(state.latest_pos, state.terrain_mode)

        if state.control_mode == "HARDWARE_MODE":
            if hw_controller.is_running:
                with hw_controller.lock:
                    t_since_update = time.time() - hw_controller.hw_state.last_update_time
                    conn_status = f"Data Delay: {t_since_update:.2f}s" if t_since_update < 1.0 else "Data Timeout!"
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
        else: # 模擬模式 (WALKING, FLOATING, etc.)
            if state.single_step_mode: print("\n" + "="*20 + f" STEP AT TIME {sim.data.time:.4f} " + "="*20)

            onnx_input, action_final = policy_manager.get_action(state.command)
            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_final

            if state.control_mode == "MANUAL_CTRL":
                final_ctrl = state.manual_final_ctrl.copy()
                sim.apply_position_control(final_ctrl, state.tuning_params)
            elif state.control_mode == "JOINT_TEST":
                final_ctrl = sim.default_pose + state.joint_test_offsets
                sim.apply_position_control(final_ctrl, state.tuning_params)
            else:
                final_ctrl = sim.default_pose + action_final * state.tuning_params.action_scale
                sim.apply_position_control(final_ctrl, state.tuning_params)
            
            state.latest_final_ctrl = final_ctrl
            
            target_time = sim.data.time + config.control_dt
            while sim.data.time < target_time:
                mujoco.mj_step(sim.model, sim.data)

        sim.render(state, overlay)
        gui.render_gui(state)
        gui.render_frame()

    # --- 7. 程式結束，清理資源 ---
    hw_controller.stop()
    gui.shutdown()
    platform.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n程式已安全退出。")

if __name__ == "__main__":
    main()