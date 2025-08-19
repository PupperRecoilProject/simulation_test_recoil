import os
import sys
import time
import numpy as np

# 為了能從工具資料夾直接執行，將專案根目錄加入模組搜尋路徑
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.hardware.virtual_teensy import VirtualTeensy
from src.core.state import SimulationState
# [修改] 不再導入 construct_observation_51
# from src.controllers.hardware_controller import construct_observation_51
# [修改] 改為導入 ObservationManager
from src.simulation.observation_manager import ObservationManager
from src.core.config import AppConfig, TuningParamsConfig, FloatingControllerConfig


class _DummySim:
    """極簡模擬器，用於驗證虛擬Teensy輸出格式。"""
    def __init__(self):
        self._t = 0.0
        # [新增] 為新架構提供必要的屬性
        self.model = type("DummyModel", (), {"sensor_adr": [0], "sensor_dim": [0]})()
        self.data = type("DummyData", (), {"sensordata": np.zeros(3)})()
        self.accelerometer_id = -1
        self.default_pose = np.zeros(12)

    def step(self):
        self._t += 0.02
        
    # ... 其他 dummy 方法保持不變 ...
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

class _DummyConfig(AppConfig):
    """
    [修改] 繼承自 AppConfig 以確保所有欄位都存在，避免 AttributeError。
    """
    def __init__(self):
        # 為所有 AppConfig 欄位提供虛擬值
        super().__init__(
            use_virtual_teensy=True,
            mujoco_model_file="",
            onnx_models={},
            policy_transition_duration=0.5,
            num_motors=12,
            physics_timestep=0.004,
            control_freq=50.0,
            control_dt=0.02,
            warmup_duration=0.0,
            command_scaling_factors=[1.0, 1.0, 1.0],
            keyboard_velocity_adjust_step=0.1,
            gamepad_sensitivity={},
            param_adjust_steps={},
            initial_tuning_params=TuningParamsConfig(kp=0, kd=0, action_scale=1, bias=0),
            floating_controller=FloatingControllerConfig(target_height=0, kp_vertical=0, kd_vertical=0, kp_attitude=0, kd_attitude=0)
        )


def main():
    state = SimulationState(_DummyConfig())
    state.sim = _DummySim()

    # [新增] 創建 ObservationManager 實例
    obs_manager = ObservationManager(state)
    # [新增] 定義一個符合 agile_model 的配方來進行測試
    test_recipe = [
        'linear_velocity', 'angular_velocity', 'gravity_vector', 
        'joint_positions', 'joint_velocities', 'last_action', 'commands'
    ]
    obs_manager.set_recipe(test_recipe)

    vt = VirtualTeensy(state, rate_hz=50.0)
    vt.write(b"monitor p\n")

    last_print = time.time()
    got = 0
    expected_dim = sum(obs_manager.component_dims.values())

    print(f"測試開始，期望的觀測維度: {expected_dim}")

    while got < 250:  # 約 5 秒
        b = vt.readline()
        if not b:
            time.sleep(0.001)
            continue
        
        # --- 模擬 HardwareController 的行為 ---
        try:
            parts = b.decode("utf-8").strip().split(",")
            if len(parts) != 34:
                continue
            data_vec = np.array(parts, dtype=np.float32)

            # 1. 將解析出的數據寫入 state.raw_...
            with state.lock:
                state.raw_torso_angular_velocity_world[:] = data_vec[0:3]
                state.raw_gravity_vector[:] = data_vec[3:6]
                state.raw_accelerometer[:] = data_vec[6:9]
                state.raw_joint_positions[:] = data_vec[10:22]
                state.raw_joint_velocities[:] = data_vec[22:34]
                # 在真實場景中，linear_velocity 會由 SimulationController 更新，
                # 但在這個測試中，我們假設它為零，因為 VirtualTeensy 不提供此數據。
                state.raw_torso_linear_velocity_world.fill(0.0)
                # 模擬 last_action 和 command
                state.raw_last_action.fill(0.1) 
                state.command = np.array([0.5, 0.2, -0.1])
        except (ValueError, IndexError):
            continue
        
        # 2. 呼叫 ObservationManager 來生成觀測向量
        obs = obs_manager.get_observation()
        got += 1

        # 驗證維度
        assert obs.shape[0] == expected_dim, f"觀測維度應為 {expected_dim}，實得 {obs.shape[0]}"
        
        now = time.time()
        if now - last_print > 1.0:
            # 格式化輸出以方便閱讀
            gyro_str = np.array2string(state.raw_torso_angular_velocity_world, precision=3, suppress_small=True)
            q0_val = state.raw_joint_positions[0]
            print(
                f"[sample] 34D gyro={gyro_str}, q0={q0_val:.3f}; "
                f"Generated Obs shape={obs.shape}"
            )
            last_print = now

    vt.close()
    print("OK. 測試成功，觀測向量維度符合預期。")


if __name__ == "__main__":
    main()