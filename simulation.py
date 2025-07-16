# simulation.py
import mujoco
import glfw
import sys
import numpy as np
from typing import TYPE_CHECKING
from config import AppConfig
from state import SimulationState, TuningParams

if TYPE_CHECKING:
    from rendering import DebugOverlay

class Simulation:
    def __init__(self, config: AppConfig):
        self.config = config
        
        try:
            self.model = mujoco.MjModel.from_xml_path(config.mujoco_model_file)
        except Exception as e:
            sys.exit(f"❌ 錯誤: 無法載入 XML 檔案 '{config.mujoco_model_file}': {e}")
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.physics_timestep

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if self.torso_id == -1: sys.exit(f"❌ 錯誤: 在 XML 中找不到名為 'torso' 的 body。")
        
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        self.default_pose = (self.model.key_qpos[home_key_id][7:].copy() if home_key_id != -1 else np.zeros(config.num_motors))

        if not glfw.init(): sys.exit("❌ 錯誤: GLFW 初始化失敗。")
        self.window = glfw.create_window(1200, 900, "Refactored MuJoCo Runner", None, None)
        if not self.window: glfw.terminate(); sys.exit("❌ 錯誤: GLFW 視窗建立失敗。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.5, -20, 90
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        print("✅ MuJoCo 模擬環境與視窗初始化完成。")

    def register_callbacks(self, keyboard_handler):
        glfw.set_key_callback(self.window, keyboard_handler.key_callback)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print("✅ MuJoCo 模擬已重置。")

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def apply_control(self, ctrl_cmd: np.ndarray, params: TuningParams):
        self.model.actuator_gainprm[:, 0] = params.kp
        self.model.actuator_biasprm[:, 1] = params.bias
        self.data.ctrl[:] = ctrl_cmd

    def step(self, state: SimulationState):
        while self.data.time < state.control_timer:
            mujoco.mj_step(self.model, self.data)

    def render(self, state: SimulationState, overlay: "DebugOverlay"):
        self.cam.lookat = self.data.body('torso').xpos
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        overlay.render(viewport, self.context, state, self)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        glfw.terminate()