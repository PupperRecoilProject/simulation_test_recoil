# 【v4.5.0 新增】 src/simulation/rendering_thread.py

import threading
import time
import glfw
import mujoco
import numpy as np # 【v4.5.1 修正】 導入 numpy
from typing import TYPE_CHECKING

from src.core.logger import log

if TYPE_CHECKING:
    from src.core.state import SimulationState
    from src.simulation.simulation import Simulation

class RenderingThread(threading.Thread):
    """
    【v4.5.0】
    一個專門的執行緒，負責擁有 GLFW 視窗、OpenGL 上下文以及所有與渲染相關的 MuJoCo 物件。
    它以固定的頻率從 SimulationState 的緩衝區讀取最新的物理狀態 (qpos)，並將其渲染到螢幕上。
    """

    def __init__(self, state: "SimulationState", sim: "Simulation"):
        """
        初始化渲染執行緒。

        Args:
            state: 對全局 SimulationState 的參考。
            sim: 對 Simulation 物件的參考，用於獲取模型和視窗相關屬性。
        """
        super().__init__(name="RenderingThread", daemon=True)
        self.state = state
        self.sim = sim
        self._stop_event = threading.Event()
        log.info("✅ 渲染執行緒已初始化。")

    def run(self):
        """渲染執行緒的主迴圈。"""
        try:
            # 步驟 1: 在此執行緒中初始化 GLFW 視窗和 MuJoCo 渲染上下文
            self.sim.initialize_window_and_context()
            if not self.sim.window:
                log.error("❌ 渲染執行緒：GLFW 視窗初始化失敗，執行緒即將退出。")
                return

            # 步驟 2: 為此執行緒創建一個獨立的、專用於渲染的 MjData 實例
            self.render_data = mujoco.MjData(self.sim.model)
            log.info("✅ 渲染執行緒：已創建獨立的 MjData 實例。")

            # 步驟 3: 進入主渲染迴圈
            while not self._stop_event.is_set() and not glfw.window_should_close(self.sim.window):
                # 從共享緩衝區安全地讀取最新的數據包
                with self.state.render_data_lock:
                    data_packet = self.state.render_data_buffer

                if data_packet:
                    # 更新本地 MjData 的時間和 qpos
                    self.render_data.time = data_packet.get('time', self.render_data.time)
                    qpos_data = data_packet.get('qpos')
                    if qpos_data is not None:
                        self.render_data.qpos[:] = qpos_data

                    # 【v4.5.1 修正】
                    # 在調用任何可能因無效數據而掛起的 MuJoCo 函式之前，
                    # 增加一個防禦性檢查，以驗證物理狀態數據是否有效。
                    if not np.isfinite(self.render_data.qpos).all():
                        error_msg = f"偵測到無效的物理狀態 (NaN/inf)，渲染已中止。qpos: {self.render_data.qpos}"
                        log.error(f"❌ {error_msg}")
                        raise RuntimeError(error_msg) # 主動拋出異常，將凍結轉為崩潰

                    # 使用更新後的數據計算正向運動學，為渲染做準備
                    mujoco.mj_forward(self.sim.model, self.render_data)

                # --- 更新與渲染場景 ---
                terrain_manager = self.state.terrain_manager_ref
                if terrain_manager:
                    if terrain_manager.needs_physics_and_scene_update:
                        mujoco.mjr_uploadHField(self.sim.model, self.sim.context, terrain_manager.hfield_id) # 上傳高度場數據到 GPU
                        terrain_manager.needs_physics_and_scene_update = False

                viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.sim.window))
                
                # 更新配方以確保 DebugOverlay 正確顯示
                if self.state.policy_manager_ref and self.state.policy_manager_ref.observation_manager:
                    self.sim.overlay.set_recipe(self.state.policy_manager_ref.get_active_recipe()) # 設定除錯疊層的配方
                
                # 【v4.5.1 最終權威修正】
                # 修正對攝影機 (cam) 的存取路徑。
                # 攝影機物件由 self.sim (Simulation 實例) 所擁有，而非 self (RenderingThread 實例)。
                if not (self.sim.mouse_button_left or self.sim.mouse_button_right):
                    self.sim.cam.lookat = self.render_data.body('torso').xpos # 讓攝影機追蹤軀幹

                # 【v4.5.1 最終權威修正】 修正所有渲染相關物件的存取路徑
                mujoco.mjv_updateScene(self.sim.model, self.render_data, self.sim.opt, None, self.sim.cam, mujoco.mjtCatBit.mjCAT_ALL, self.sim.scene)
                mujoco.mjr_render(viewport, self.sim.scene, self.sim.context)
                self.sim.overlay.render(viewport, self.sim.context, self.state, self.sim, self.render_data)
                
                glfw.swap_buffers(self.sim.window) # 交換緩衝區以顯示
                glfw.poll_events() # 處理視窗事件
                time.sleep(0.001) # 短暫休眠以釋放 CPU

        except Exception as e:
            log.error(f"❌ 渲染執行緒發生未處理的異常: {e}", exc_info=True)
        finally:
            log.info("渲染執行緒正在清理資源並退出...")
            if self.sim.window:
                glfw.terminate() # 確保 GLFW 被終止
                self.sim.window = None

    def stop(self):
        """向執行緒發送停止信號。"""
        self._stop_event.set()
        log.info("渲染執行緒已請求停止。")