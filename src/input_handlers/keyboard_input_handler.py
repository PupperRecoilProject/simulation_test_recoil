# -*- coding: utf-8 -*-
"""
KeyboardInputHandler
- 指令向量佈局: [vy, vx, wz, pitch]
- 每次按鍵 (keydown/PRESS) 就對對應分量加/減固定步長；放開不變
- 鍵位: W/S→vx、A/D→vy、Q/E→wz、I/K→pitch、C→清零
"""

try:
    import glfw
except ImportError:
    glfw = None

import numpy as np

from src.core.state import SimulationState
from src.core.logger import log
from src.core.event_system import (
    event_bus,
    EVENT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_SHUTDOWN_REQUESTED,
    EVENT_TUNING_PARAM_ADJUSTED,
    EVENT_TUNING_PARAM_SELECT_REQUESTED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_UI_PAGE_CHANGE_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_JOINT_SELECT_REQUESTED,
    EVENT_JOINT_VALUE_ADJUSTED,
    EVENT_COMMAND_UPDATED,
    EVENT_POLICY_CHANGE_REQUESTED,
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_SERIAL_COMMAND_SEND,
)


class KeyboardInputHandler:
    """處理所有鍵盤輸入事件，並根據當前模式進行分派。
    指令向量佈局: [vy, vx, wz, pitch]
    """
    def __init__(self, state: SimulationState, xbox_handler, terrain_manager):
        """初始化函式，儲存必要的物件參考。"""
        self.state = state  # 全域狀態的參考
        self.config = state.config  # 設定檔的參考
        self.terrain_manager = terrain_manager  # 地形管理器的參考
        self.param_keys = ['kp', 'kd', 'action_scale', 'bias']  # 可調參數的鍵名列表
        self.num_params = len(self.param_keys)  # 可調參數的數量

        # 若未在 config.yaml 定義 keyboard_pitch_step，提供安全預設值
        if not hasattr(self.config, 'keyboard_pitch_step'):
            try:
                base = float(getattr(self.config, 'keyboard_velocity_adjust_step', 0.3))
            except Exception:
                base = 0.3
            self.config.keyboard_pitch_step = max(0.02, base * 0.5)  # I/K 每次俯仰調整幅度

    
    def register_callbacks(self, window):
        """向 GLFW 註冊鍵盤事件回呼，同時保留既有（例如 MuJoCo viewer）回呼。"""
        if glfw is None:
            log.warning("glfw 模組不存在，無法註冊鍵盤事件")
            return

        # 先「取回舊回呼」：PyGLFW 的 set_*_callback 會回傳之前的 callback
        prev_key_cb  = glfw.set_key_callback(window, None)
        prev_char_cb = glfw.set_char_callback(window, None)

        def combo_key_callback(win, key, scancode, action, mods):            # 先執行我們的鍵盤邏輯
            try:
                self.key_callback(win, key, scancode, action, mods)
            except Exception as ex:
                log.error(f'key_callback error: {ex}')
            # 一律再轉交給「舊回呼」（例如 MuJoCo viewer）
            if prev_key_cb is not None:                
                try:
                    prev_key_cb(win, key, scancode, action, mods)
                except Exception as ex:
                    log.error(f'prev_key_cb error: {ex}')

        def combo_char_callback(win, codepoint):
            try:
                self.char_callback(win, codepoint)
            except Exception as ex:
                log.error(f'char_callback error: {ex}')

            if prev_char_cb is not None:
                try:
                    prev_char_cb(win, codepoint)
                except Exception as ex:
                    log.error(f'prev_char_cb error: {ex}')

        # 重新掛上我們的「組合」回呼
        glfw.set_key_callback(window, combo_key_callback)
        glfw.set_char_callback(window, combo_char_callback)

        # 可選：避免失焦漏掉 RELEASE（視需求）
        # glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)


    def char_callback(self, window, codepoint):
        """處理可列印字元的輸入，專門用於序列埠模式。"""
        if self.state.control_mode == "SERIAL_MODE":
            self.state.serial_command_buffer += chr(codepoint)

    def key_callback(self, window, key, scancode, action, mods):
        """處理所有按鍵事件，為所有專用模式建立壁壘。"""
        # 根據當前模式，分派給不同的處理函式
        if self.state.control_mode == "SERIAL_MODE":
            self.handle_serial_mode_keys(key, action)
            return

        if self.state.control_mode == "JOINT_TEST":
            self.handle_joint_test_mode_keys(key, action)
            return

        if self.state.control_mode == "MANUAL_CTRL":
            self.handle_manual_ctrl_mode_keys(key, action)
            return

        # 如果不在任何專用模式中，則執行通用和預設模式的按鍵處理
        self.handle_global_and_default_keys(window, key, action)

    # ---------------- 專用模式：Serial ----------------

    def handle_serial_mode_keys(self, key, action):
        """專門處理序列埠模式下的按鍵。"""
        if action == glfw.PRESS:
            if key == glfw.KEY_GRAVE_ACCENT:
                # 返回上一個模式（或預設 WALKING）
                event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode=self.state.previous_mode or "WALKING")
                return
            if key == glfw.KEY_ENTER:
                # 送出整個緩衝字串並清空
                command = self.state.serial_command_buffer.strip()
                if command:
                    event_bus.publish(EVENT_SERIAL_COMMAND_SEND, command=command)
                    self.state.serial_command_buffer = ""
                return
            if key == glfw.KEY_BACKSPACE and len(self.state.serial_command_buffer) > 0:
                # 刪除最後一個字元
                self.state.serial_command_buffer = self.state.serial_command_buffer[:-1]

    # ---------------- 專用模式：Joint Test ----------------

    def handle_joint_test_mode_keys(self, key, action):
        """專門處理關節測試模式下的按鍵，只更新狀態，不發送運動指令。"""
        if action == glfw.PRESS and key == glfw.KEY_G:
            # 如果硬體控制器正在運行，則返回 HARDWARE_MODE，否則返回 WALKING
            if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                self.state.set_control_mode("HARDWARE_MODE")
            else:
                self.state.set_control_mode("WALKING")
            return

        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
                self.state.joint_test_index = (self.state.joint_test_index - 1) % 12
            elif key == glfw.KEY_RIGHT_BRACKET and action == glfw.PRESS:
                self.state.joint_test_index = (self.state.joint_test_index + 1) % 12
            elif key == glfw.KEY_UP:
                self.state.joint_test_offsets[self.state.joint_test_index] += 0.1
            elif key == glfw.KEY_DOWN:
                self.state.joint_test_offsets[self.state.joint_test_index] -= 0.1
            elif key == glfw.KEY_C and action == glfw.PRESS:
                self.state.joint_test_offsets.fill(0.0)
            # 運動指令已統一由其他路徑處理

    # ---------------- 專用模式：Manual Ctrl ----------------

    def handle_manual_ctrl_mode_keys(self, key, action):
        """專門處理手動控制模式下的按鍵（離散步進：只在 PRESS 累加一步）。"""
        # 模式退出
        if action == glfw.PRESS and key == glfw.KEY_G:
            self.state.set_control_mode("WALKING")
            return

        # ---- 單發/切換類（只在 PRESS 處理） ----
        if action == glfw.PRESS:
            if key == glfw.KEY_F:
                self.state.manual_mode_is_floating = not self.state.manual_mode_is_floating
                event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=self.state.manual_mode_is_floating)
                return

            if key == glfw.KEY_GRAVE_ACCENT:
                event_bus.publish(EVENT_MODE_CHANGE_REQUESTED, mode="SERIAL_MODE")
                return

            if key == glfw.KEY_ESCAPE:
                event_bus.publish(EVENT_SHUTDOWN_REQUESTED)
                return
            if key == glfw.KEY_R:
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="hard")
                return
            if key == glfw.KEY_X:
                event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type="soft")
                return
            if key == glfw.KEY_Y:
                if self.state.terrain_mode == "INFINITE":
                    self.terrain_manager.regenerate_terrain_and_adjust_robot(self.state.latest_pos)
                else:
                    print("⚠️ 'Y'鍵 (重生地形) 只在無限地形模式下有效。")
                return
            if key == glfw.KEY_P:
                self.terrain_manager.save_hfield_to_png()
                return

            if key == glfw.KEY_TAB:
                event_bus.publish(EVENT_UI_PAGE_CHANGE_REQUESTED)
                return

        # ---- 參數選擇 / 調整（保留 REPEAT）----
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_LEFT_BRACKET:
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=-1)
            elif key == glfw.KEY_RIGHT_BRACKET:
                event_bus.publish(EVENT_TUNING_PARAM_SELECT_REQUESTED, direction=1)
            elif key == glfw.KEY_UP:
                event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=1)
            elif key == glfw.KEY_DOWN:
                event_bus.publish(EVENT_TUNING_PARAM_ADJUSTED, direction=-1)

        # ---- 離散步進（移動指令：僅 PRESS 時生效）----
        if action == glfw.PRESS:
            step = float(self.config.keyboard_velocity_adjust_step)
            pstep = float(getattr(self.config, 'keyboard_pitch_step', step * 0.5))
            cmd = self.state.command.copy()  # 4D: [vy, vx, wz, pitch]

            # W/S → vx
            if key == glfw.KEY_W:
                cmd[1] += step
            elif key == glfw.KEY_S:
                cmd[1] -= step

            # A/D → vy
            elif key == glfw.KEY_A:
                cmd[0] += step
            elif key == glfw.KEY_D:
                cmd[0] -= step

            # Q/E → wz
            elif key == glfw.KEY_Q:
                cmd[2] += step
            elif key == glfw.KEY_E:
                cmd[2] -= step

            # I/K → pitch
            elif key == glfw.KEY_I:
                cmd[3] += pstep
            elif key == glfw.KEY_K:
                cmd[3] -= pstep

            # C → 清零
            elif key == glfw.KEY_C:
                cmd.fill(0.0)

            # 夾取 pitch（若有設定）
            try:
                if hasattr(self.config, 'pitch_limit'):
                    cmd[3] = float(np.clip(cmd[3], -float(self.config.pitch_limit), float(self.config.pitch_limit)))
            except Exception:
                pass

            # 發布：宣告輸入來源（可選）+ 指令更新
            event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode='KEYBOARD')
            event_bus.publish(EVENT_COMMAND_UPDATED, command=cmd)

    # ---------------- 通用 / 預設模式 ----------------

    def handle_global_and_default_keys(self, window, key, action):
        """處理通用鍵與預設模式下的按鍵。"""
        if action == glfw.PRESS:
            # 模式切換
            if key == glfw.KEY_G:
                self.state.set_control_mode("JOINT_TEST")
                return
            if key == glfw.KEY_H:
                if self.state.hardware_controller_ref and self.state.hardware_controller_ref.is_running:
                    self.state.set_control_mode("HARDWARE_MODE")
                else:
                    print("⚠️ 硬體控制器未啟動，無法切換到 HARDWARE_MODE。")
                return
            if key == glfw.KEY_T:
                self.state.manual_mode_is_floating = not self.state.manual_mode_is_floating
                event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=self.state.manual_mode_is_floating)
                return
            if key == glfw.KEY_M:
                event_bus.publish(EVENT_UI_PAGE_CHANGE_REQUESTED)
                return

            # 其他全域操作
            if key == glfw.KEY_L:
                event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device="xbox")
                return
            if key == glfw.KEY_O:
                event_bus.publish(EVENT_TERRAIN_CHANGE_REQUESTED, name="TOGGLE")
                return
            if key == glfw.KEY_N:
                event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name="NEXT")
                return
