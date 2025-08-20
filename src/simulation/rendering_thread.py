# 【v4.5.0 新增】 src/simulation/rendering_thread.py

import threading
import time
import glfw
import mujoco
from typing import TYPE_CHECKING

from src.core.logger import log

if TYPE_CHECKING:
    from src.core.state import SimulationState
    from src.simulation.simulation import Simulation

class RenderingThread(threading.Thread):
    """
    【v4.5.0 新增】 一個專門用於處理 MuJoCo/GLFW 渲染的獨立執行緒。

    職責:
    1.  擁有並管理 GLFW 視窗和 MuJoCo 渲染上下文的唯一所有權。
    2.  在其自己的主迴圈中，以盡力而為 (best-effort) 的方式持續渲染場景。
    3.  從 SimulationState 的一個安全緩衝區中讀取最新的物理狀態進行渲染。
    4.  處理所有與視窗直接相關的事件 (滑鼠、鍵盤、視窗關閉)。
    """

    def __init__(self, state: "SimulationState", sim: "Simulation"):
        """
        初始化渲染執行緒。

        Args:
            state (SimulationState): 對中央狀態管理器的引用。
            sim (Simulation): 對底層 MuJoCo 模擬接口的引用。
        """
        super().__init__(name="RenderingThread", daemon=True)
        self.state = state
        self.sim = sim
        self._stop_event = threading.Event()
        log.info("✅ 渲染執行緒已初始化。")

    def run(self):
        """
        渲染執行緒的主迴圈。
        此方法在執行緒啟動時被呼叫，並負責初始化 GLFW 和渲染上下文。
        """
        try:
            # --- 步驟 1: 在此執行緒中初始化所有與 OpenGL/GLFW 相關的資源 ---
            self.sim.initialize_window_and_context()
            if not self.sim.window:
                log.error("❌ 渲染執行緒：GLFW 視窗初始化失敗，執行緒即將退出。")
                return

            log.info("✅ 渲染執行緒：GLFW 視窗與渲染上下文已成功初始化。")

            # --- 步驟 2: 渲染主迴圈 ---
            while not self._stop_event.is_set() and not glfw.window_should_close(self.sim.window):
                # a. 從 SimulationState 的渲染緩衝區獲取最新數據
                with self.state.render_data_lock:
                    render_data = self.state.render_data_buffer
                    # 將緩衝區中的數據同步到本地的 sim.data
                    # 這是確保渲染數據與物理計算數據分離的關鍵
                    if render_data:
                        self.sim.data.qpos[:] = render_data.get('qpos', self.sim.data.qpos)
                        # 注意：直接修改 xpos, xquat 等衍生量可能不安全或無效
                        # 最穩健的方式是僅同步 qpos/qvel/ctrl，然後呼叫 mj_forward
                        mujoco.mj_forward(self.sim.model, self.sim.data)

                # b. 執行完整的渲染流程
                viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.sim.window))
                
                # 更新配方以確保 DebugOverlay 正確顯示
                if self.state.policy_manager_ref and self.state.policy_manager_ref.observation_manager:
                    self.sim.overlay.set_recipe(self.state.policy_manager_ref.get_active_recipe())
                
                self.sim.overlay.render(viewport, self.sim.context, self.state, self.sim)
                
                # c. 交換緩衝區 (會受 VSync 阻塞)
                glfw.swap_buffers(self.sim.window)

                # d. 處理視窗事件
                glfw.poll_events()

                # e. 短暫休眠，避免在 VSync 禁用時空轉耗盡 CPU
                time.sleep(0.001)

        except Exception as e:
            log.error(f"❌ 渲染執行緒發生未處理的異常: {e}", exc_info=True)
        finally:
            log.info("渲染執行緒正在清理資源並退出...")
            if self.sim.window:
                glfw.terminate()
                self.sim.window = None

    def stop(self):
        """
        向渲染執行緒發送停止信號。
        """
        self._stop_event.set()
        log.info("渲染執行緒已請求停止。")