"""Minimal GUI manager built on optional imgui."""

from __future__ import annotations

from typing import Tuple

try:
    import imgui  # type: ignore
    from imgui.integrations.glfw import GlfwRenderer  # type: ignore
    HAS_IMGUI = True
except Exception:  # pragma: no cover - optional dependency
    HAS_IMGUI = False


class GuiManager:
    def __init__(self, window):
        self.window = window
        if HAS_IMGUI:
            imgui.create_context()
            self.impl = GlfwRenderer(window)
        else:
            self.impl = None
        self.sim_panel_size: Tuple[int, int] = (0, 0)

    def start_frame(self) -> None:
        if not HAS_IMGUI:
            return
        self.impl.process_inputs()
        imgui.new_frame()

    def render_gui(self, state) -> None:
        if not HAS_IMGUI:
            return
        imgui.begin("Placeholder")
        imgui.text("imgui not fully integrated")
        imgui.end()

    def render_frame(self) -> None:
        if not HAS_IMGUI:
            return
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def shutdown(self) -> None:
        if self.impl is not None:
            self.impl.shutdown()
