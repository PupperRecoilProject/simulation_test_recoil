# simulation.py
import mujoco
import glfw
import sys
import numpy as np
from typing import TYPE_CHECKING
import os

from config import AppConfig
from state import SimulationState, TuningParams

if TYPE_CHECKING:
    from rendering import DebugOverlay
    from keyboard_input_handler import KeyboardInputHandler

class Simulation:
    """
    封裝 MuJoCo 模擬、GLFW 視窗和渲染邏輯。
    """
    def __init__(self, config: AppConfig):
        """初始化 MuJoCo 模型、資料、GLFW 視窗以及滑鼠控制相關狀態。"""
        self.config = config
        
        try:
            with open(config.mujoco_model_file, 'r', encoding='utf-8') as f:
                xml_string = f.read()
            corrected_xml_string = xml_string.replace('meshdir="assets"', 'meshdir="mesh"')
            
            self.model = mujoco.MjModel.from_xml_string(corrected_xml_string)
            print(f"✅ XML '{config.mujoco_model_file}' 已載入，並在執行時將 meshdir 從 'assets' 修正為 'mesh'。")
            
            for i in range(self.model.nu):
                self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
            print("✅ 所有致動器的模式已在執行時被強制設為 AFFINE，以啟用 Python 端的 PD 控制。")
            
        except Exception as e:
            sys.exit(f"❌ 錯誤: 無法載入或處理 XML 檔案 '{config.mujoco_model_file}': {e}")
            
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.physics_timestep

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if self.torso_id == -1:
            sys.exit("❌ 錯誤: 在 XML 中找不到名為 'torso' 的 body。")
        
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id != -1:
            self.default_pose = self.model.key_qpos[home_key_id][7:].copy()
        else:
            self.default_pose = np.zeros(config.num_motors)
            print("⚠️ 警告: 在 XML 中未找到名為 'home' 的 keyframe，將使用零作為預設姿態。")

        if not glfw.init(): sys.exit("❌ 錯誤: GLFW 初始化失敗。")
        self.window = glfw.create_window(1200, 900, "MuJoCo 模擬器 (含滑鼠控制)", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit("❌ 錯誤: GLFW 視窗建立失敗。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.mouse_button_left = False
        self.mouse_button_right = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.5, -20, 90
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        print("✅ MuJoCo 模擬環境與視窗初始化完成 (含滑鼠控制)。")

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_button_left = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_button_right = (action == glfw.PRESS)
        if action == glfw.PRESS:
            self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        if not (self.mouse_button_left or self.mouse_button_right):
            return
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
        width, height = glfw.get_window_size(window)
        
        action_type = mujoco.mjtMouse.mjMOUSE_NONE
        if self.mouse_button_right:
            action_type = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif self.mouse_button_left:
            action_type = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        
        if action_type != mujoco.mjtMouse.mjMOUSE_NONE:
            mujoco.mjv_moveCamera(self.model, action_type, dx / height, dy / height, self.scene, self.cam)

    def _scroll_callback(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scene, self.cam)

    def register_callbacks(self, keyboard_handler: "KeyboardInputHandler"):
        keyboard_handler.register_callbacks(self.window)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print("✅ MuJoCo 模擬已重置。")

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)
        
    def apply_position_control(self, target_pos: np.ndarray, params: TuningParams):
        self.model.actuator_gainprm[:, 0] = params.kp
        self.model.actuator_biasprm[:, 1] = -params.kp
        self.model.actuator_biasprm[:, 2] = -params.kd
        self.data.ctrl[:] = target_pos
        force_bias = np.full(self.config.num_motors, params.bias)
        self.data.qfrc_applied[6:] = force_bias

    def step(self, state: SimulationState):
        while self.data.time < state.control_timer:
            mujoco.mj_step(self.model, self.data)

    def render(self, state: SimulationState, overlay: "DebugOverlay"):
        """【修改】簡化 render 函式，將所有渲染邏輯集中到 DebugOverlay 中。"""
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        
        # 呼叫 DebugOverlay 的主渲染函式
        overlay.render(viewport, self.context, state, self)
        
        # 交換緩衝區並處理事件
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        glfw.terminate()
