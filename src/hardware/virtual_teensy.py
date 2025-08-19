import time
import numpy as np


class VirtualTeensy:
    """
    模擬 Teensy 的序列埠介面 (serial-like interface)：
    - write(): 處理 monitor/move 等命令
    - readline(): 以固定頻率輸出一行 34 維資料
    """

    def __init__(self, state, rate_hz: float = 50.0):
        self.state = state
        self.is_open = True
        self.mode = "HUMAN"  # or "POLICY_STREAM"
        self._last_read_time = 0.0
        # 以 50Hz 為基準的最小間隔
        self._min_interval = 1.0 / float(rate_hz)

    # ---- 指令處理 ----
    def write(self, data: bytes) -> int:
        try:
            cmd = data.decode("utf-8").strip()
        except Exception:
            return len(data)

        if cmd.startswith("monitor"):
            # monitor p -> 開始串流；monitor h -> 停止串流
            self.mode = "POLICY_STREAM" if "p" in cmd else "HUMAN"

        elif cmd == "stop":
            sim = getattr(self.state, "sim", None)
            if sim is not None and hasattr(sim, "data"):
                # 將控制目標鎖在目前關節角度
                sim.data.ctrl[:] = sim.data.qpos[7:]

        elif cmd.startswith("move all"):
            # move all <12個角度>
            parts = cmd.split()
            vals = parts[2:] if len(parts) >= 14 else []
            if len(vals) == 12:
                tgt = np.array([float(v) for v in vals], dtype=np.float32)
                scale = float(getattr(self.state, "action_scale", 1.0))
                sim = getattr(self.state, "sim", None)
                if sim is not None and hasattr(sim, "data"):
                    base = getattr(sim, "default_pose", np.zeros(12, dtype=np.float32))
                    sim.data.ctrl[:] = base + tgt * scale

        return len(data)

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    # ---- 串流輸出（50 Hz）----
    def readline(self) -> bytes:
        if not self.is_open or self.mode != "POLICY_STREAM":
            return b""

        now = time.perf_counter()
        if now - self._last_read_time < self._min_interval:
            return b""
        self._last_read_time = now

        sim = getattr(self.state, "sim", None)

        # 角速度(3)、重力向量(3)、加速度計(3)、俯仰(1)、關節角(12)、關節角速(12) = 34
        ang_vel = self._safe_call(sim, "imu_gyro_local", 3)          # rad/s
        g_vec = self._safe_call(sim, "gravity_local", 3, default=[0, 0, 1.0])  # 單位向量
        acc = self._safe_call(sim, "imu_accel_local", 3)             # m/s^2
        pitch = float(getattr(sim, "pitch_rad", 0.0))
        qpos = self._safe_call(sim, "joint_positions", 12)           # rad
        qvel = self._safe_call(sim, "joint_velocities", 12)          # rad/s

        vec = np.concatenate([
            ang_vel,
            g_vec,
            acc,
            np.array([pitch], dtype=np.float32),
            qpos,
            qvel,
        ]).astype(np.float32)

        if vec.shape[0] != 34:
            return b""

        line = ",".join(f"{x:.6f}" for x in vec) + "\n"
        return line.encode("utf-8")

    @staticmethod
    def _safe_call(sim, name: str, n: int, default=None):
        """安全地呼叫模擬器方法，若失敗則回傳預設值。"""
        if default is None:
            default = [0.0] * n
        if sim is None:
            return np.array(default, dtype=np.float32)
        fn = getattr(sim, name, None)
        if callable(fn):
            out = np.asarray(fn(), dtype=np.float32)
            if out.shape[0] == n:
                return out
        return np.array(default, dtype=np.float32)
