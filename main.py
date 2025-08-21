# main.py
"""
【輕量級開發者模式】

此腳本提供了一個基於原生 GLFW 視窗的輕量級、單執行緒模擬環境。

主要用途:
- 快速啟動和測試核心物理模擬。
- 作為底層演算法（如 AI 策略、控制器邏輯）的開發和除錯平台。
- 在不依賴 NiceGUI Web UI 的情況下運行模擬。

與 main_nicegui.py 的區別:
- 此腳本為單執行緒，所有操作（模擬、渲染、輸入）在一個主迴圈中同步執行。
- UI 極簡，僅透過文字疊加層顯示狀態。
- 不具備 main_nicegui.py 的非阻塞特性和豐富的圖形化介面。

建議：
- 日常使用和參數調校，請使用 `main_nicegui.py`。
- 進行核心演算法開發或需要一個簡單、高效的測試環境時，請使用此腳本。
"""
import sys
import numpy as np
import mujoco
import time

from src.core.config import load_config
from src.core.state import SimulationState
from src.simulation.simulation import Simulation
from src.hardware.policy import PolicyManager
# 【v4.3.2 刪除】 移除舊的 ObservationBuilder
# from src.simulation.observation import ObservationBuilder
# 【v4.3.2 新增】 導入新的 ObservationManager
from src.simulation.observation_manager import ObservationManager
from src.simulation.rendering import DebugOverlay
from src.input_handlers.keyboard_input_handler import KeyboardInputHandler
from src.input_handlers.xbox_input_handler import XboxInputHandler
from src.simulation.floating_controller import FloatingController
from src.hardware.serial_communicator import SerialCommunicator
from src.simulation.terrain_manager import TerrainManager
from src.controllers.hardware_controller import HardwareController

# 【v4.3.2 修改】 main 函式
def main():
    """主程式入口：初始化所有組件並運行模擬迴圈。"""
    from src.hardware.xbox_controller import XboxController 
    print("\n--- 機器人模擬控制器 (含硬體與多模型模式) ---")
    
    # --- 1. 初始化核心組件 ---
    config = load_config()
    state = SimulationState(config)
    sim = Simulation(config)

    # --- 2. 將核心物件的參考存入 state，使其成為全域上下文 ---
    state.sim = sim
    
    # --- 3. 按照依賴順序初始化所有管理器 ---
    terrain_manager = TerrainManager(sim.model, sim.data)
    state.terrain_manager_ref = terrain_manager
    
    floating_controller = FloatingController(config, sim.model, sim.data, terrain_manager)
    state.floating_controller_ref = floating_controller
    
    serial_comm = SerialCommunicator()
    state.serial_communicator_ref = serial_comm
    
    xbox_handler = XboxInputHandler(state)

    # 【v4.3.2 刪除】 刪除舊的 ObservationBuilder 實例化
    # obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config)
    
    # 【v4.3.2 新增】 實例化新的 ObservationManager
    observation_manager = ObservationManager(state)
    # 【v4.3.2 新增】 將其參考存入 state 以便全局訪問
    state.observation_manager_ref = observation_manager

    # 在無 GUI 版本中仍建立 DebugOverlay 以顯示文字資訊
    overlay = DebugOverlay()

    # 【v4.3.2 修改】 將 observation_manager 傳入 PolicyManager
    # 【v4.4.5 修改】 在 PolicyManager 實例化時傳入 state 參數
    policy_manager = PolicyManager(config, observation_manager, overlay, state)
    state.policy_manager_ref = policy_manager
    state.available_policies = policy_manager.model_names
    
    # 將 serial_comm 傳入 HardwareController 的建構函式
    hw_controller = HardwareController(config, policy_manager, state, serial_comm) 
    state.hardware_controller_ref = hw_controller

    # KeyboardInputHandler 不再需要直接傳入 serial_comm
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager)
    # 先註冊回呼，待初始化視窗後會正式生效
    sim.register_callbacks(keyboard_handler)
    # 初始化 GLFW 視窗與渲染上下文
    sim.initialize_window_and_context()

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
    # 【v4.3.3 修改】 移除歷史註解，使輸出更整潔
    print("\n--- Simulation Started (SPACE: Pause, N: Step) ---")
    print("    (F: Float, G: Joint Test, B: Manual Ctrl, H: Hardware Mode)")
    print("    (M: Input Mode, R: Hard Reset, X: Soft Reset)")
    print("    (Y: Regen Terrain, P: Save Terrain PNG, 1..: Select Policy)")
    print("    (V: Cycle Terrain, K: Toggle HW AI)")
    print("    ( ~ : Toggle Serial Console )")

    state.execute_one_step = False

    # --- 6. 主模擬迴圈 ---
    while not sim.should_close():
        if state.single_step_mode and not state.execute_one_step:
            sim.render(state)
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
                    t_since_update = time.time() - hw_controller.hw_state_data.last_update_time
                    conn_status = f"Data Delay: {t_since_update:.2f}s" if t_since_update < 1.0 else "Data Timeout!"
                    state.hardware_status_text = f"Connection Status: {conn_status}\n"
                    state.hardware_status_text += f"LinVel: {np.array2string(hw_controller.hw_state_data.lin_vel_local, precision=2)}\n"
                    state.hardware_status_text += f"Gyro: {np.array2string(hw_controller.hw_state_data.imu_gyro_radps, precision=2)}"
            else:
                state.hardware_status_text = "Hardware controller not running."
        
        elif state.control_mode == "SERIAL_MODE":
            pass
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

        sim.render(state)

    # --- 7. 程式結束，清理資源 ---
    hw_controller.shutdown() # 【v4.3.2 修正】 使用 shutdown 代替 stop
    sim.close()
    xbox_handler.close()
    serial_comm.close()
    print("\n程式已安全退出。")

if __name__ == "__main__":
    main()