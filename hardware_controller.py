# hardware_controller.py
import serial
import serial.tools.list_ports
import threading
import time
import re
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig
    from policy import PolicyManager
    from state import SimulationState
    from serial_communicator import SerialCommunicator

class RobotStateHardware:
    def __init__(self):
        self.imu_gyro_radps = np.zeros(3, dtype=np.float32)
        self.imu_acc_g = np.zeros(3, dtype=np.float32)
        self.joint_positions_rad = np.zeros(12, dtype=np.float32)
        self.joint_velocities_radps = np.zeros(12, dtype=np.float32)
        self.lin_vel_local = np.zeros(3, dtype=np.float32)
        self.gravity_vector_local = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.timestamp_ms = 0
        self.robot_mode = "N/A"
        self.is_calibrated = False
        self.rpy_rad = np.zeros(3, dtype=np.float32)
        self.target_current_ma = np.zeros(12, dtype=np.float32)
        self.actual_current_ma = np.zeros(12, dtype=np.float32)
        self.prev_rpy_rad = np.zeros(3, dtype=np.float32)
        self.prev_rpy_time = 0.0
        self.last_update_time = 0.0

class HardwareController:
    """【修改版】管理與實體硬體的AI控制迴圈，從SerialCommunicator借用連接。"""
    
    def __init__(self, config: 'AppConfig', policy: 'PolicyManager', global_state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        """【修改】初始化時接收 SerialCommunicator 的參考。"""
        self.config = config
        self.policy = policy
        self.global_state = global_state
        self.serial_comm = serial_comm # 【新增】儲存參考
        
        self.ser = None 
        self.is_running = False 
        self.read_thread = None 
        self.control_thread = None 
        
        self.hw_state = RobotStateHardware()
        self.lock = threading.Lock()
        self.ai_control_enabled = threading.Event()

        self.foot_positions_in_body = np.array([
            [-0.0804, -0.1759, -0.1964],
            [ 0.0806, -0.1759, -0.1964],
            [-0.0804,  0.0239, -0.1964],
            [ 0.0806,  0.0239, -0.1964],
        ], dtype=np.float32)
        print("✅ 硬體控制器已初始化。")

    def connect_and_start(self) -> bool:
        """【核心重構】不再自己建立連接，而是從 SerialCommunicator 獲取已建立的連接。"""
        if self.is_running:
            print("硬體控制器已在運行中。")
            return True
            
        if not self.serial_comm.is_connected:
            print("❌ 硬體模式錯誤：請先按 'U' 鍵連接序列埠。")
            return False
        
        self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            print("❌ 硬體模式錯誤：無法從 SerialCommunicator 獲取有效的序列埠連接。")
            return False
            
        print(f"✅ 硬體控制器已接管序列埠 {self.ser.port} 的控制權。")
        self.serial_comm.is_managed_by_hardware_controller = True
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
        self.read_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("✅ 硬體控制執行緒已啟動。")
        return True

    def stop(self):
        """【修改】停止時，將序列埠的控制權交還，但不關閉連接。"""
        if not self.is_running: return
        
        print("正在停止硬體控制器...")
        self.is_running = False
        self.disable_ai()
        self.ai_control_enabled.set()
        
        if self.control_thread and self.control_thread.is_alive(): self.control_thread.join(timeout=1)
        if self.read_thread and self.read_thread.is_alive(): self.read_thread.join(timeout=1)
        
        if self.serial_comm:
            self.serial_comm.is_managed_by_hardware_controller = False
            print("序列埠控制權已交還。")
        
        self.ser = None
        print("硬體控制器已完全停止。")
        
    def enable_ai(self):
        if not self.is_running:
            print("無法啟用 AI：硬體控制器未運行。")
            return
        print("🤖 AI 控制已啟用。")
        self.policy.reset()
        self.ai_control_enabled.set()
        self.global_state.hardware_ai_is_active = True

    def disable_ai(self):
        print("⏸️ AI 控制已暫停。")
        self.ai_control_enabled.clear()
        self.global_state.hardware_ai_is_active = False
        if self.ser and self.ser.is_open:
            try: self.ser.write(b"stop\n")
            except serial.SerialException as e: print(f"發送停止指令失敗: {e}")

    def parse_teensy_data(self, line: str):
        try:
            parts = line.split(',')
            if len(parts) != 57: return
            with self.lock:
                current_time = time.time()
                self.hw_state.timestamp_ms = int(parts[0])
                self.hw_state.robot_mode = parts[1]
                self.hw_state.is_calibrated = (parts[2] == '1')
                rpy_deg = np.array(parts[3:6], dtype=np.float32)
                self.hw_state.rpy_rad = rpy_deg * (np.pi / 180.0)
                self.hw_state.imu_acc_g = np.array(parts[6:9], dtype=np.float32)
                self.hw_state.joint_positions_rad = np.array(parts[9:21], dtype=np.float32)
                self.hw_state.joint_velocities_radps = np.array(parts[21:33], dtype=np.float32)
                self.hw_state.target_current_ma = np.array(parts[33:45], dtype=np.float32)
                self.hw_state.actual_current_ma = np.array(parts[45:57], dtype=np.float32)
                self.hw_state.last_update_time = current_time
        except (ValueError, IndexError) as e:
            print(f"❌ 解析硬體數據時出錯: {e} | 原始數據: {line}")

    def estimate_linear_velocity(self):
        with self.lock:
            acc_g = self.hw_state.imu_acc_g.copy()
            w_body = self.hw_state.imu_gyro_radps.copy()
        body_gravity_vec = acc_g * 9.81
        world_gravity_vec = np.array([0, 0, -9.81])
        if np.linalg.norm(body_gravity_vec) < 1e-6: return
        try:
            rot_body_to_world, _ = Rotation.align_vectors(world_gravity_vec.reshape(1, -1), body_gravity_vec.reshape(1, -1))
        except (ValueError, np.linalg.LinAlgError): return
        w_world = rot_body_to_world.apply(w_body)
        foot_velocities_world = []
        for i in range(4):
            r_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[i])
            v_foot_world = np.cross(w_world, r_foot_in_world)
            foot_velocities_world.append(np.linalg.norm(v_foot_world))
        stance_foot_idx = np.argmin(foot_velocities_world)
        r_stance_foot_in_world = rot_body_to_world.apply(self.foot_positions_in_body[stance_foot_idx])
        v_body_world_est = -np.cross(w_world, r_stance_foot_in_world)
        with self.lock:
            self.hw_state.gravity_vector_local = body_gravity_vec
            self.hw_state.lin_vel_local = rot_body_to_world.inv().apply(v_body_world_est)

    def construct_observation(self) -> np.ndarray:
        with self.lock:
            current_time = self.hw_state.last_update_time
            dt = current_time - self.hw_state.prev_rpy_time
            if dt > 1e-6 and self.hw_state.prev_rpy_time > 0:
                delta_rpy = self.hw_state.rpy_rad - self.hw_state.prev_rpy_rad
                if delta_rpy[2] > np.pi: delta_rpy[2] -= 2 * np.pi
                if delta_rpy[2] < -np.pi: delta_rpy[2] += 2 * np.pi
                estimated_gyro = delta_rpy / dt
                self.hw_state.imu_gyro_radps = estimated_gyro
            self.hw_state.prev_rpy_rad = self.hw_state.rpy_rad
            self.hw_state.prev_rpy_time = current_time
        self.estimate_linear_velocity()
        with self.lock:
            self.hw_state.command = self.global_state.command * np.array(self.config.command_scaling_factors)
            obs_list = {
                'linear_velocity': self.hw_state.lin_vel_local,
                'angular_velocity': self.hw_state.imu_gyro_radps,
                'gravity_vector': self.hw_state.gravity_vector_local / 9.81,
                'accelerometer': self.hw_state.imu_acc_g,
                'joint_positions': self.hw_state.joint_positions_rad,
                'joint_velocities': self.hw_state.joint_velocities_radps,
                'last_action': self.hw_state.last_action,
                'commands': self.hw_state.command,
            }
            recipe = self.policy.get_active_recipe()
            if not recipe:
                print("⚠️ 警告: 無法從策略管理器獲取有效的觀察配方。")
                return np.array([])
            final_obs_list = [obs_list[key] for key in recipe if key in obs_list]
            return np.concatenate(final_obs_list).astype(np.float32)

    def _read_from_port(self):
        print("[硬體讀取線程已啟動] 等待來自 Teensy 的數據...")
        while self.is_running:
            if not self.ser or not self.ser.is_open:
                self.stop(); break
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line: self.parse_teensy_data(line)
            except (serial.SerialException, OSError):
                print("❌ 錯誤：序列埠斷開連接或讀取錯誤。"); self.stop(); break
            except Exception as e: print(f"❌ _read_from_port 發生未知錯誤: {e}")
                
    def _control_loop(self):
        print("\n--- 硬體控制線程已就緒，等待 AI 啟用 ---")
        default_pose_hardware = self.global_state.sim.default_pose
        while self.is_running:
            self.ai_control_enabled.wait()
            if not self.is_running: break
            loop_start_time = time.perf_counter()
            observation = self.construct_observation()
            if observation.size == 0:
                time.sleep(0.02); continue
            _, action_raw = self.policy.get_action_for_hardware(observation)
            with self.lock:
                self.hw_state.last_action[:] = action_raw
            final_command = default_pose_hardware + action_raw * self.global_state.tuning_params.action_scale
            action_str = ' '.join(f"{a:.4f}" for a in final_command)
            command_to_send = f"move all {action_str}\n"
            if self.ser and self.ser.is_open:
                try: self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException: self.stop()
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0: time.sleep(sleep_time)