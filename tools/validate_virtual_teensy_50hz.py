import os
import sys
import time
import numpy as np

# 為了能從工具資料夾直接執行，將專案根目錄加入模組搜尋路徑
# Append project root to module search path for direct execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.hardware.virtual_teensy import VirtualTeensy
from src.core.state import SimulationState
from src.controllers.hardware_controller import construct_observation_51


class _DummySim:
    """極簡模擬器，用於驗證虛擬Teensy輸出格式。"""

    def __init__(self):
        self._t = 0.0

    def step(self):
        self._t += 0.02

    def imu_gyro_local(self):
        return np.array([0.0, 0.0, np.sin(self._t)], dtype=np.float32)

    def gravity_local(self):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def imu_accel_local(self):
        return np.array([0.0, 0.0, -9.81], dtype=np.float32)

    @property
    def pitch_rad(self):
        return float(0.1 * np.sin(self._t))

    def joint_positions(self):
        return np.linspace(-0.5, 0.5, 12).astype(np.float32)

    def joint_velocities(self):
        return np.zeros(12, dtype=np.float32)


class _DummyConfig:
    """簡化版設定，提供最少必要欄位給 SimulationState 使用。"""
    use_virtual_teensy = True
    num_motors = 12  # 模擬用的馬達數
    # 最小化的初始調校參數，避免初始化時找不到欄位
    class _DummyTuning:
        kp = 0.0
        kd = 0.0
        action_scale = 1.0
        bias = 0.0

    initial_tuning_params = _DummyTuning()


def main():
    state = SimulationState(_DummyConfig())
    state.sim = _DummySim()

    vt = VirtualTeensy(state, rate_hz=50.0)
    vt.write(b"monitor p\n")

    hw = type("HW", (), {})()
    last_print = time.time()
    got = 0

    while got < 250:  # 約 5 秒
        b = vt.readline()
        if not b:
            time.sleep(0.001)
            continue
        parts = b.decode("utf-8").strip().split(",")
        if len(parts) != 34:
            continue
        v = np.asarray([float(x) for x in parts], dtype=np.float32)

        hw.angular_velocity_radps = v[0:3]
        hw.gravity_vector_norm = v[3:6]
        hw.accelerometer_ms2 = v[6:9]
        hw.pitch_rad = float(v[9])
        hw.joint_positions_rad = v[10:22]
        hw.joint_velocities_radps = v[22:34]

        obs = construct_observation_51(state, hw)
        got += 1

        now = time.time()
        if now - last_print > 1.0:
            print(
                f"[sample] 34D gyro={hw.angular_velocity_radps.round(3)}, "
                f"q0={hw.joint_positions_rad[0]:.3f}; 51D shape={obs.shape}"
            )
            last_print = now

    vt.close()
    print("OK.")


if __name__ == "__main__":
    main()
