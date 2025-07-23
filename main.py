# main.py
import sys
import numpy as np
import mujoco
import time

from config import load_config
from state import SimulationState
from simulation import Simulation
from policy import PolicyManager
from observation import ObservationBuilder
from rendering import DebugOverlay
from keyboard_input_handler import KeyboardInputHandler
from xbox_input_handler import XboxInputHandler
from floating_controller import FloatingController
from serial_communicator import SerialCommunicator
from terrain_manager import TerrainManager
from hardware_controller import HardwareController

def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    from xbox_controller import XboxController 
    print("\n--- 機器人模擬控制器 (含硬體與多模型模式) ---")
    
    config = load_config()
    state = SimulationState(config)
    sim = Simulation(config)
    
    terrain_manager = TerrainManager(sim.model, sim.data)
    state.terrain_manager_ref = terrain_manager
    
    floating_controller = FloatingController(config, sim.model, sim.data)
    state.floating_controller_ref = floating_controller
    
    serial_comm = SerialCommunicator()
    xbox_handler = XboxInputHandler(state)

    # --- 初始化流程 ---
    obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    overlay = DebugOverlay() # 先建立 overlay
    
    # 將 obs_builder 和 overlay 傳入 PolicyManager
    policy_manager = PolicyManager(config, obs_builder, overlay)
    
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names # 讓 state 知道有哪些可用的模型名稱
    
    hw_controller = HardwareController(config, policy_manager, state)
    state.hardware_controller_ref = hw_controller

    keyboard_handler = KeyboardInputHandler(state, serial_comm, xbox_handler, terrain_manager)
    keyboard_handler.register_callbacks(sim.window)

    def hard_reset():
        print("\n--- 正在執行完全重置 (Hard Reset) ---")
        if state.control_mode == "HARDWARE_MODE": return
        sim.reset()
        policy_manager.reset() # 重置策略管理器狀態
        if state.control_mode == "FLOATING": state.set_control_mode("WALKING")
        state.reset_control_state(sim.data.time)
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        state.hard_reset_requested = False

    def soft_reset():
        print("\n--- 正在執行空中姿態重置 (Soft Reset) ---")
        if state.control_mode == "HARDWARE_MODE": return
        sim.data.qpos[7:] = sim.default_pose
        sim.data.qvel[6:] = 0
        policy_manager.reset() # 重置策略管理器狀態
        state.clear_command()
        state.joint_test_offsets.fill(0.0)
        state.manual_final_ctrl.fill(0.0)
        state.manual_mode_is_floating = False
        mujoco.mj_forward(sim.model, sim.data)
        state.soft_reset_requested = False

    hard_reset()
    # --- 【新】更新啟動提示訊息 ---
    print("\n--- 模擬開始 (SPACE: 暫停, N:下一步) ---")
    print("    (F: 懸浮, G: 關節測試, B: 手動控制, T: 序列埠, H: 硬體模式)")
    print("    (M: 輸入模式, R: 硬重置, X: 軟重置, U: 掃描序列埠, J: 掃描搖桿)")
    print("    (1,2,3..: 選擇目標策略 | 在硬體模式下，按 K 啟用/禁用 AI)")

    state.execute_one_step = False

    while not sim.should_close():
        if state.single_step_mode and not state.execute_one_step:
            sim.render(state, overlay)
            continue
        if state.execute_one_step: state.execute_one_step = False

        if state.input_mode == "GAMEPAD": xbox_handler.update_state()
        if state.hard_reset_requested: hard_reset()
        if state.soft_reset_requested: soft_reset()

        state.latest_pos = sim.data.body('torso').xpos.copy()
        state.latest_quat = sim.data.body('torso').xquat.copy()

        if state.control_mode == "HARDWARE_MODE":
            if hw_controller.is_running:
                with hw_controller.lock:
                    t_since_update = time.time() - hw_controller.hw_state.last_update_time
                    conn_status = f"數據延遲: {t_since_update:.2f}s" if t_since_update < 1.0 else "數據超時!"
                    state.hardware_status_text = f"連接狀態: {conn_status}\n"
                    state.hardware_status_text += f"LinVel: {np.array2string(hw_controller.hw_state.lin_vel_local, precision=2)}\n"
                    state.hardware_status_text += f"Gyro: {np.array2string(hw_controller.hw_state.imu_gyro_radps, precision=2)}"
            else:
                state.hardware_status_text = "硬體控制器未運行。"
        
        elif state.control_mode == "SERIAL_MODE":
            if state.serial_is_connected: state.serial_latest_messages = serial_comm.get_latest_messages()
            if state.serial_command_to_send:
                serial_comm.send_command(state.serial_command_to_send)
                state.serial_command_to_send = ""
        else:
            if state.single_step_mode: print("\n" + "="*20 + f" STEP AT TIME {sim.data.time:.4f} " + "="*20)

            # 主迴圈的這一部分不需要改變，因為所有複雜性都已封裝在 PolicyManager 中
            onnx_input, action_final = policy_manager.get_action(state.command)
            state.latest_onnx_input = onnx_input.flatten()
            state.latest_action_raw = action_final

            if state.control_mode == "MANUAL_CTRL":
                final_ctrl = state.manual_final_ctrl.copy()
                sim.apply_position_control(final_ctrl, state.tuning_params)
            elif state.control_mode == "JOINT_TEST":
                final_ctrl = sim.default_pose + state.joint_test_offsets
                sim.apply_position_control(final_ctrl, state.tuning_params)
            else: # WALKING or FLOATING
                final_ctrl = sim.default_pose + action_final * state.tuning_params.action_scale
                sim.apply_position_control(final_ctrl, state.tuning_params)
            
            state.latest_final_ctrl = final_ctrl
            
            target_time = sim.data.time + config.control_dt
            while sim.data.time < target_time:
                mujoco.mj_step(sim.model, sim.data)

        sim.render(state, overlay) # 渲染場景和除錯資訊

    hw_controller.stop()
    sim.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n程式已安全退出。")

if __name__ == "__main__":
    main()