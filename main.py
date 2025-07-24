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
    
    # --- 1. 初始化核心組件 ---
    config = load_config() # 載入設定檔
    state = SimulationState(config) # 建立全域狀態物件
    sim = Simulation(config) # 建立模擬環境物件
    
    # --- 2. 【核心修改】將核心物件的參考存入 state，使其成為全域上下文 ---
    state.sim = sim # 將模擬物件存入 state
    
    # --- 3. 按照依賴順序初始化所有管理器 ---
    terrain_manager = TerrainManager(sim.model, sim.data) # 初始化地形管理器
    state.terrain_manager_ref = terrain_manager # 將地形管理器存入 state
    
    floating_controller = FloatingController(config, sim.model, sim.data, terrain_manager) # 初始化懸浮控制器
    state.floating_controller_ref = floating_controller # 將懸浮控制器存入 state
    
    serial_comm = SerialCommunicator() # 初始化序列埠通訊器
    state.serial_communicator_ref = serial_comm # 將 serial_comm 存入 state
    
    xbox_handler = XboxInputHandler(state) # 初始化 Xbox 搖桿處理器

    obs_builder = ObservationBuilder(sim.data, sim.model, sim.torso_id, sim.default_pose, config) # 初始化觀察向量產生器
    overlay = DebugOverlay() # 初始化除錯介面渲染器
    
    policy_manager = PolicyManager(config, obs_builder, overlay) # 初始化策略管理器
    state.policy_manager_ref = policy_manager # 將策略管理器存入 state
    state.available_policies = policy_manager.model_names # 獲取所有可用的策略名稱
    
    # 將 serial_comm 傳入 HardwareController 的建構函式
    hw_controller = HardwareController(config, policy_manager, state, serial_comm) # 初始化硬體控制器
    state.hardware_controller_ref = hw_controller # 將硬體控制器存入 state

    # KeyboardInputHandler 不再需要直接傳入 serial_comm
    keyboard_handler = KeyboardInputHandler(state, xbox_handler, terrain_manager) # 初始化鍵盤處理器
    keyboard_handler.register_callbacks(sim.window) # 註冊鍵盤回呼函式

    # --- 4. 定義重置函式 ---
    def hard_reset():
        """硬重置函式，將機器人恢復到初始狀態。"""
        print("\n--- 正在執行機器人硬重置 (R Key) ---")
        if state.control_mode == "HARDWARE_MODE": return # 硬體模式下不執行
        mujoco.mj_resetData(sim.model, sim.data) # 重置 MuJoCo 數據
        sim.data.qpos[0], sim.data.qpos[1] = 0, 0 # 重置 X, Y 位置
        start_ground_z = terrain_manager.get_height_at(0, 0) # 獲取地面高度
        robot_height_offset = 0.3 # 設定機器人離地高度
        sim.data.qpos[2] = start_ground_z + robot_height_offset # 設定 Z 位置
        print(f"機器人重置至原點：地形高度({start_ground_z:.2f}m) + 偏移({robot_height_offset:.2f}m) = 世界Z({sim.data.qpos[2]:.2f}m)")
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0]) # 重置姿態四元數
        sim.data.qpos[7:] = sim.default_pose # 重置關節角度到預設姿態
        sim.data.qvel[:] = 0 # 重置速度
        sim.data.ctrl[:] = sim.default_pose # 重置控制指令
        for _ in range(10): mujoco.mj_step(sim.model, sim.data) # 執行幾個模擬步驟穩定狀態
        policy_manager.reset() # 重置策略管理器
        if state.control_mode == "FLOATING": state.set_control_mode("WALKING") # 如果在懸浮模式，切換回走路模式
        state.reset_control_state(sim.data.time) # 重置控制狀態
        state.clear_command() # 清除使用者指令
        state.joint_test_offsets.fill(0.0) # 清除關節測試偏移
        state.manual_final_ctrl.fill(0.0) # 清除手動控制指令
        state.manual_mode_is_floating = False # 關閉手動懸浮
        state.hard_reset_requested = False # 重置請求旗標
        # 【錯誤修正】將 self.data 改為 sim.data，因為此函式不是類別方法，沒有 self。
        mujoco.mj_forward(sim.model, sim.data) # 更新模擬狀態

    def soft_reset():
        """軟重置函式，僅重置機器人姿態和速度。"""
        print("\n--- 正在執行空中姿態重置 (X Key) ---")
        if state.control_mode == "HARDWARE_MODE": return # 硬體模式下不執行
        sim.data.qpos[3:7] = np.array([1., 0, 0, 0]) # 重置姿態四元數
        sim.data.qpos[7:] = sim.default_pose # 重置關節角度
        sim.data.qvel[:] = 0 # 重置速度
        policy_manager.reset() # 重置策略
        state.clear_command() # 清除指令
        state.joint_test_offsets.fill(0.0) # 清除關節測試偏移
        state.manual_final_ctrl.fill(0.0) # 清除手動控制指令
        state.manual_mode_is_floating = False # 關閉手動懸浮
        # 【錯誤修正】將 self.data 改為 sim.data，原因同上。
        mujoco.mj_forward(sim.model, sim.data) # 更新模擬狀態
        state.soft_reset_requested = False # 重置請求旗標

    # --- 5. 啟動程序 ---
    if terrain_manager.is_functional: # 如果地形管理器可用
        terrain_manager.initial_generate() # 生成初始地形
    hard_reset() # 執行一次硬重置
    
    print("\n--- 模擬開始 (SPACE: 暫停, N:下一步) ---")
    print("    (F: 懸浮, G: 關節測試, B: 手動控制, T: 序列埠, H: 硬體模式)")
    print("    (M: 輸入模式, R: 重置機器人, X: 軟重置)")
    print("    (Y: 重生地形, P: 儲存地形PNG, 1..: 選策略 | K: 硬體AI開關)")
    print("    (V: 切換地形模式)")

    state.execute_one_step = False # 初始化單步執行旗標

    # --- 6. 主模擬迴圈 ---
    while not sim.should_close(): # 當視窗未關閉時循環
        if state.single_step_mode and not state.execute_one_step: # 如果是單步模式且未請求下一步
            sim.render(state, overlay) # 僅渲染，不推進模擬
            continue # 繼續下一輪迴圈
        if state.execute_one_step: state.execute_one_step = False # 如果執行了單步，重置旗標

        if state.input_mode == "GAMEPAD": xbox_handler.update_state() # 更新搖桿狀態
        if state.hard_reset_requested: hard_reset() # 處理硬重置請求
        if state.soft_reset_requested: soft_reset() # 處理軟重置請求

        state.latest_pos = sim.data.body('torso').xpos.copy() # 更新最新位置
        state.latest_quat = sim.data.body('torso').xquat.copy() # 更新最新姿態
        
        if terrain_manager.is_functional: # 如果地形管理器可用
            terrain_manager.update(state.latest_pos, state.terrain_mode) # 更新地形

        if state.control_mode == "HARDWARE_MODE": # 如果是硬體模式
            if hw_controller.is_running: # 如果硬體控制器在運行
                with hw_controller.lock: # 鎖定硬體狀態
                    t_since_update = time.time() - hw_controller.hw_state.last_update_time # 計算距離上次數據更新的時間
                    # 將狀態文字從中文改為英文
                    conn_status = f"Data Delay: {t_since_update:.2f}s" if t_since_update < 1.0 else "Data Timeout!"
                    state.hardware_status_text = f"Connection Status: {conn_status}\n"
                    state.hardware_status_text += f"LinVel: {np.array2string(hw_controller.hw_state.lin_vel_local, precision=2)}\n"
                    state.hardware_status_text += f"Gyro: {np.array2string(hw_controller.hw_state.imu_gyro_radps, precision=2)}"
            else:
                # 將狀態文字從中文改為英文
                state.hardware_status_text = "Hardware controller not running."
        
        elif state.control_mode == "SERIAL_MODE": # 如果是序列埠模式
            if state.serial_is_connected: state.serial_latest_messages = serial_comm.get_latest_messages() # 獲取最新日誌
            if state.serial_command_to_send: # 如果有待發送的指令
                serial_comm.send_command(state.serial_command_to_send) # 發送指令
                state.serial_command_to_send = "" # 清空待發送指令
        else: # 模擬模式 (WALKING, FLOATING, etc.)
            if state.single_step_mode: print("\n" + "="*20 + f" STEP AT TIME {sim.data.time:.4f} " + "="*20) # 單步模式下打印時間

            onnx_input, action_final = policy_manager.get_action(state.command) # 從策略管理器獲取動作
            state.latest_onnx_input = onnx_input.flatten() # 儲存 ONNX 輸入
            state.latest_action_raw = action_final # 儲存原始動作輸出

            if state.control_mode == "MANUAL_CTRL": # 如果是手動控制模式
                final_ctrl = state.manual_final_ctrl.copy() # 使用手動設定的控制量
                sim.apply_position_control(final_ctrl, state.tuning_params) # 應用位置控制
            elif state.control_mode == "JOINT_TEST": # 如果是關節測試模式
                final_ctrl = sim.default_pose + state.joint_test_offsets # 使用預設姿態+偏移量
                sim.apply_position_control(final_ctrl, state.tuning_params) # 應用位置控制
            else: # AI 控制模式
                final_ctrl = sim.default_pose + action_final * state.tuning_params.action_scale # 計算最終控制量
                sim.apply_position_control(final_ctrl, state.tuning_params) # 應用位置控制
            
            state.latest_final_ctrl = final_ctrl # 儲存最終控制量
            
            target_time = sim.data.time + config.control_dt # 計算下一個控制週期的目標時間
            while sim.data.time < target_time: # 推進物理模擬直到目標時間
                mujoco.mj_step(sim.model, sim.data)

        sim.render(state, overlay) # 渲染場景和除錯介面

    # --- 7. 程式結束，清理資源 ---
    hw_controller.stop() # 停止硬體控制器
    sim.close() # 關閉模擬視窗
    xbox_handler.close() # 關閉搖桿
    serial_comm.close() # 關閉序列埠
    print("\n程式已安全退出。")

if __name__ == "__main__":
    main()