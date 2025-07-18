# simulation.py
import mujoco
import glfw
import sys
import numpy as np
from typing import TYPE_CHECKING

from config import AppConfig
from state import SimulationState, TuningParams

# 使用 TYPE_CHECKING 來避免循環導入，同時又能獲得 Pylance/MyPy 的型別提示
if TYPE_CHECKING:
    from rendering import DebugOverlay
    from keyboard_input_handler import KeyboardInputHandler

class Simulation:
    """
    封裝 MuJoCo 模擬、GLFW 視窗和渲染邏輯。
    新增了完整的滑鼠視角控制功能。
    """
    def __init__(self, config: AppConfig):
        """初始化 MuJoCo 模型、資料、GLFW 視窗以及滑鼠控制相關狀態。"""
        self.config = config
        
        try:
            self.model = mujoco.MjModel.from_xml_path(config.mujoco_model_file)
        except Exception as e:
            sys.exit(f"❌ 錯誤: 無法載入 XML 檔案 '{config.mujoco_model_file}': {e}")
            
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

        # --- 新增：初始化滑鼠控制所需變數 ---
        self.mouse_button_left = False
        self.mouse_button_right = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        # --- 新增結束 ---

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.5, -20, 90
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        
        # --- 根據您的要求，將字體大小設定為 100% ---
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        
        # --- 新增：註冊滑鼠相關的回調函式 ---
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        # --- 新增結束 ---

        print("✅ MuJoCo 模擬環境與視窗初始化完成 (含滑鼠控制)。")

    # --- 新增：滑鼠事件回調函式 ---
    def _mouse_button_callback(self, window, button, action, mods):
        """處理滑鼠按鍵按下和釋放事件。"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_button_left = True
                self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.mouse_button_left = False
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.mouse_button_right = True
                self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.mouse_button_right = False

    def _mouse_move_callback(self, window, xpos, ypos):
        """處理滑鼠移動事件，根據按下的按鍵來旋轉或平移視角。"""
        if not (self.mouse_button_left or self.mouse_button_right):
            return

        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos

        width, height = glfw.get_window_size(window)

        # 根據按下的按鍵決定操作類型
        action_type = None
        if self.mouse_button_right:
            action_type = mujoco.mjtMouse.mjMOUSE_MOVE_H # 平移
        elif self.mouse_button_left:
            action_type = mujoco.mjtMouse.mjMOUSE_ROTATE_H # 旋轉

        if action_type:
            mujoco.mjv_moveCamera(self.model, action_type, dx / height, dy / height, self.scene, self.cam)

    def _scroll_callback(self, window, xoffset, yoffset):
        """處理滑鼠滾輪事件，用於縮放視角。"""
        # yoffset > 0 表示向上滾動 (放大), yoffset < 0 表示向下滾動 (縮小)
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scene, self.cam)
    # --- 新增結束 ---

    def register_callbacks(self, keyboard_handler: "KeyboardInputHandler"):
        """註冊所有來自鍵盤處理器的回調函式。"""
        keyboard_handler.register_callbacks(self.window)

    def reset(self):
        """重置 MuJoCo 模擬狀態到初始狀態。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print("✅ MuJoCo 模擬已重置。")

    def should_close(self) -> bool:
        """檢查視窗是否應該關閉。"""
        return glfw.window_should_close(self.window)
        
    def apply_position_control(self, target_pos: np.ndarray, params: TuningParams):
        """使用MuJoCo內建的PD控制器"""
        # 設定致動器增益 a = Kp
        self.model.actuator_gainprm[:, 0] = params.kp
        # 設定偏置 b1 = -Kp
        self.model.actuator_biasprm[:, 1] = -params.kp
        # 設定偏置 b2 = -Kd
        self.model.actuator_biasprm[:, 2] = -params.kd
        # 注意：現在控制輸入是目標角度！
        self.data.ctrl[:] = target_pos
        # =========================================================================
        # === 【核心修復】將 tuning_params.bias 應用為一個額外的力矩偏置         ===
        # =========================================================================
        # 創建一個長度為12的向量，每個元素的值都是當前的 bias
        force_bias = np.full(self.config.num_motors, params.bias)
        
        # 將這個偏置力矩向量應用到12個關節的自由度上。
        # data.qfrc_applied 的前6個元素對應浮動基座，後12個對應關節。
        # 因為此函式在主迴圈中每一步都會被呼叫，所以會不斷刷新這個值，無需手動清零。
        self.data.qfrc_applied[6:] = force_bias
        # =========================================================================

    def step(self, state: SimulationState):
        """執行物理模擬，直到模擬時間趕上控制計時器。"""
        while self.data.time < state.control_timer:
            mujoco.mj_step(self.model, self.data)

    def render(self, state: SimulationState, overlay: "DebugOverlay"):
        """渲染當前場景和除錯資訊。"""
        # 僅在非序列埠模式下自動追蹤軀幹
        if state.control_mode != "SERIAL_MODE":
            # 如果滑鼠正在操作，則暫時停止自動追蹤，以提供更好的手動控制體驗
            if not (self.mouse_button_left or self.mouse_button_right):
                 self.cam.lookat = self.data.body('torso').xpos
        
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        
        if state.control_mode != "SERIAL_MODE":
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)
        
        overlay.render(viewport, self.context, state, self)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        """關閉視窗並終止 GLFW。"""
        glfw.terminate()