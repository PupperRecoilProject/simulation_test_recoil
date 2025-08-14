# virtual_teensy.py
import numpy as np
import time
from typing import TYPE_CHECKING
from src.core.logger import log  # 導入專案內的日誌記錄器

if TYPE_CHECKING:
    from simulation import Simulation
    from state import SimulationState


class VirtualTeensy:
    """
    一個真實Teensy硬體的軟體模擬器 (數位雙生)。
    它模仿了`pyserial`的`Serial`物件的關鍵介面 (如 read, write, readline)，
    使得HardwareController可以在不知道對方是真是假的情況下與之互動。
    """

    def __init__(self, state: "SimulationState"):
        """
        初始化虛擬Teensy。

        Args:
            state (SimulationState): 對全域狀態物件的參考，以便存取模擬器(sim)。
        """
        self.state = state
        self.sim = state.sim
        self.is_open = True
        self.in_waiting = 0  # 模擬的輸入緩衝區中等待讀取的位元組數
        self._buffer = b""  # 內部緩衝區，用於存放待回傳的感測器數據
        self._last_read_time = time.time()
        self.mode = "HUMAN"  # 模擬Teensy的監控模式，預設為人類可讀

        # 為了模仿 ObservationBuilder 的行為，我們需要一個 torso_id
        try:
            import mujoco

            self.torso_id = mujoco.mj_name2id(
                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
        except Exception:
            self.torso_id = -1

        log.info("✅ 虛擬Teensy 已初始化，並連結到 MuJoCo 模擬實例。")

    def write(self, data: bytes):
        """
        模擬向Teensy寫入指令。
        攔截來自HardwareController的指令，並在模擬器中執行對應操作。
        """
        command = data.decode("utf-8").strip()
        log.debug(f"[VirtualTeensy] 收到指令: {command}")

        if command == "monitor p":
            self.mode = "POLICY_STREAM"
            log.info("[VirtualTeensy] 模式切換 -> POLICY_STREAM")
        elif command == "monitor h":
            self.mode = "HUMAN"
            log.info("[VirtualTeensy] 模式切換 -> HUMAN")
        elif command == "stop":
            # 在模擬中，「停止」意味著將控制訊號設為當前的關節位置，以產生制動力
            self.sim.data.ctrl[:] = self.sim.data.qpos[7:]
            log.info("[VirtualTeensy] 執行 Stop 指令 (ctrl set to current qpos)")
        elif command.startswith("move all"):
            try:
                parts = command.split(" ")[2:]
                target_angles = np.array(parts, dtype=np.float32)
                # 核心：將move指令直接應用於MuJoCo的控制器
                self.sim.data.ctrl[:] = (
                    self.sim.default_pose
                    + target_angles * self.state.tuning_params.action_scale
                )
            except (IndexError, ValueError) as e:
                log.error(f"[VirtualTeensy] 解析 move all 指令失敗: {e}")
        return len(data)  # 模仿pyserial，回傳寫入的位元組數

    def readline(self) -> bytes:
        """
        模擬從Teensy讀取一行數據。
        從MuJoCo模擬器中提取物理狀態，並將其格式化成與真實Teensy完全一樣的字串。
        """
        if self.mode != "POLICY_STREAM":
            return b""

        # 模擬通訊頻率，避免無限快地產生數據
        current_time = time.time()
        if current_time - self._last_read_time < (
            1.0 / 100
        ):  # 假設Teensy以100Hz回傳數據
            return b""
        self._last_read_time = current_time

        # --- 從MuJoCo竊取數據 ---
        # 這些計算邏輯直接從 observation.py 借用，以確保一致性
        q_inv = self._get_torso_inverse_rotation()

        # 1. 角速度 (3維)
        angular_velocity = self._rotate_vec_by_quat_inv(
            self.sim.data.cvel[self.torso_id, :3], q_inv
        )
        # 2. 重力向量 (3維)
        gravity_vector = self._rotate_vec_by_quat_inv(np.array([0, 0, -1]), q_inv)
        # 3. 加速度計 (3維)
        accelerometer = self.sim.data.sensor("accelerometer").data.copy()
        # 4. 俯仰角 (1維)
        pitch = np.arcsin(-2.0 * (q_inv[1] * q_inv[3] - q_inv[0] * q_inv[2]))
        # 5. 關節角度 (12維)
        joint_positions = self.sim.data.qpos[7:]
        # 6. 關節速度 (12維)
        joint_velocities = self.sim.data.qvel[6:]

        # --- 將數據格式化為字串 ---
        data_vec = np.concatenate(
            [
                angular_velocity,
                gravity_vector,
                accelerometer,
                [pitch],
                joint_positions,
                joint_velocities,
            ]
        )

        # 轉換成逗號分隔的字串，與真實Teensy的'monitor p'輸出完全一致
        formatted_string = ",".join(f"{x:.6f}" for x in data_vec) + "\n"

        return formatted_string.encode("utf-8")

    def reset_input_buffer(self):
        """模仿pyserial的介面。"""
        log.debug("[VirtualTeensy] reset_input_buffer() called.")
        self._buffer = b""
        self.in_waiting = 0

    def close(self):
        """模仿pyserial的介面。"""
        log.info("[VirtualTeensy] Connection closed.")
        self.is_open = False

    # --- 以下是從 observation.py 複製過來的輔助函式 ---
    def _get_torso_inverse_rotation(self):
        torso_quat = self.sim.data.xquat[self.torso_id]
        norm = np.sum(np.square(torso_quat))
        if norm < 1e-8:
            torso_quat = np.array([1.0, 0, 0, 0])
        torso_quat /= np.sqrt(np.sum(np.square(torso_quat)))
        return np.array(
            [torso_quat[0], -torso_quat[1], -torso_quat[2], -torso_quat[3]]
        ) / np.sum(np.square(torso_quat))

    def _rotate_vec_by_quat_inv(self, v, q_inv):
        u, s = q_inv[1:], q_inv[0]
        return (
            2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)
        )
