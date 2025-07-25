import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import OpenGL.GL as gl
import numpy as np

from state import SimulationState


class GuiManager:
    """Render GUI panels using Dear ImGui."""

    def __init__(self, window):
        if not window:
            raise ValueError("GUI Manager requires a valid GLFW window")
        self.window = window
        imgui.create_context()
        self.impl = GlfwRenderer(window)

        self.fbo = gl.glGenFramebuffers(1)
        self.texture = gl.glGenTextures(1)
        self.render_buffer = gl.glGenRenderbuffers(1)
        self.sim_panel_size = (1, 1)
        self.update_fbo(800, 600)

    def update_fbo(self, width, height):
        if width <= 0 or height <= 0:
            return
        self.sim_panel_size = (int(width), int(height))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            width,
            height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0
        )

        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.render_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.render_buffer
        )

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("❌ Framebuffer initialization failed")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def start_frame(self):
        self.impl.process_inputs()
        imgui.new_frame()

    def render_gui(self, state: SimulationState):
        main_viewport = imgui.get_main_viewport()
        imgui.set_next_window_position(main_viewport.pos.x, main_viewport.pos.y)
        imgui.set_next_window_size(main_viewport.size.x, main_viewport.size.y)
        imgui.begin(
            "MainLayout",
            flags=imgui.WINDOW_NO_DECORATION | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS,
        )

        imgui.begin_child("SimulationView", width=imgui.get_window_width() - 360, height=-1, border=True)
        size = imgui.get_content_region_available()
        if (int(size.x), int(size.y)) != self.sim_panel_size:
            self.update_fbo(int(size.x), int(size.y))
        imgui.image(self.texture, size.x, size.y, uv0=(0, 1), uv1=(1, 0))
        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("Controls", width=0, height=-1, border=True)
        self.draw_control_panel(state)
        imgui.end_child()

        imgui.end()

    def draw_control_panel(self, state: SimulationState):
        imgui.text("Pupper 控制台")
        imgui.separator()

        if imgui.collapsing_header("模式控制", default_open=True)[0]:
            if imgui.radio_button("走路", state.control_mode == "WALKING"):
                state.set_control_mode("WALKING")
            imgui.same_line()
            if imgui.radio_button("懸浮", state.control_mode == "FLOATING"):
                state.set_control_mode("FLOATING")
            if imgui.radio_button("硬體", state.control_mode == "HARDWARE_MODE"):
                state.set_control_mode("HARDWARE_MODE")

        if imgui.collapsing_header("AI 策略", default_open=True)[0]:
            pm = state.policy_manager_ref
            if pm:
                current_idx = pm.model_names.index(pm.primary_policy_name) if pm.primary_policy_name in pm.model_names else 0
                clicked, sel = imgui.combo("當前策略", current_idx, pm.model_names)
                if clicked:
                    pm.select_target_policy(pm.model_names[sel])
                if pm.is_transitioning:
                    imgui.progress_bar(pm.transition_alpha, (0, 0), f"融合中... {int(pm.transition_alpha*100)}%")

        if imgui.collapsing_header("調校參數", default_open=True)[0]:
            p = state.tuning_params
            changed, kp = imgui.slider_float("Kp", p.kp, 0.0, 50.0, "%.2f")
            if changed:
                p.kp = kp
            changed, kd = imgui.slider_float("Kd", p.kd, 0.0, 5.0, "%.2f")
            if changed:
                p.kd = kd
            changed, ascale = imgui.slider_float("動作縮放", p.action_scale, 0.0, 1.0, "%.2f")
            if changed:
                p.action_scale = ascale

        if imgui.collapsing_header("狀態資訊", default_open=True)[0]:
            imgui.text(f"輸入模式: {state.input_mode}")
            if state.control_mode == "HARDWARE_MODE":
                imgui.text_unformatted(state.hardware_status_text)
            else:
                imgui.text(f"模擬時間: {state.sim.data.time:.2f} s")
                imgui.text(f"軀幹 Z 高度: {state.latest_pos[2]:.3f} m")

    def render_frame(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def shutdown(self):
        self.impl.shutdown()
