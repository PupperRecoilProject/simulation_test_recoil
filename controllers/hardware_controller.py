"""Hardware controller service for Teensy."""
import serial
import threading
import time
import numpy as np
import re
from typing import TYPE_CHECKING

from utils.logger import log
from state import OperatingMode, ControlSubMode

if TYPE_CHECKING:  # 型別提示避免循環匯入
    from utils.config import AppConfig
    from core.policy import PolicyManager
    from state import SimulationState
    from serial_communicator import SerialCommunicator


def crc8_maxim(data: bytes) -> int:
    """計算 CRC-8/MAXIM 校驗碼"""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc


class HardwareController:
    """重構版硬體控制器，負責與 Teensy 溝通。"""

    def __init__(self, config: 'AppConfig', policy: 'PolicyManager',
                 global_state: 'SimulationState', serial_comm: 'SerialCommunicator'):
        self.config = config
        self.policy = policy
        self.global_state = global_state
        self.serial_comm = serial_comm

        self.ser: serial.Serial | None = None
        self._is_running = threading.Event()
        self._read_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # --- 健壯解析所需的內部狀態 ---
        self._partial_line = ""               # 尚未解析完成的殘餘字串
        self._csv_regex = re.compile(r'[\s,]+')  # 同時處理逗號/空白
        self._EXPECTED_FIELDS = 42            # 40資料 + CRC + 空字串
        self._bad_crc_count = 0               # CRC 錯誤計數
        self._mismatch_count = 0              # 欄位錯誤計數

        # 最新感測資料緩衝
        self._raw_angular_velocity = np.zeros(3)
        self._raw_gravity_vector = np.zeros(3)
        self._raw_linear_velocity = np.zeros(3)
        self._raw_accel = np.zeros(3)
        self._raw_joint_positions = np.zeros(12)
        self._raw_joint_velocities = np.zeros(12)
        self._last_action = np.zeros(12)
        self._last_update_time = 0.0

        log.info("✅ 重構版硬體控制器已初始化。")

    @property
    def is_running(self) -> bool:
        """公開查詢是否運行中的屬性"""
        return self._is_running.is_set()

    # ----------------------
    # lifecycle 生命週期
    # ----------------------
    def attach_serial(self, ser: serial.Serial | None) -> None:
        """由 SerialCommunicator 呼叫，將已連線的序列埠交給硬體控制器。"""
        self.ser = ser
        if ser is not None:
            self.serial_comm.is_managed_by_hardware_controller = True

    def start(self) -> bool:
        """啟動背景執行緒並接管序列埠。"""
        if self._is_running.is_set():
            log.info("硬體控制器已在運行中。")
            return False
        if not self.serial_comm.is_connected:
            log.error("❌ 硬體模式錯誤：請先連接序列埠。")
            return False
        if not self.ser:
            self.ser = self.serial_comm.get_serial_connection()
        if not self.ser:
            log.error("❌ 無法取得有效序列埠連接。")
            return False
        self.serial_comm.is_managed_by_hardware_controller = True
        try:
            log.info("-> 切換 Teensy 至 CSV_42 串流模式...")
            self.ser.write(b"monitor csv42\n")
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            log.error(f"❌ 無法切換模式: {e}")
            self.serial_comm.is_managed_by_hardware_controller = False
            return False
        self._is_running.set()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        with self.global_state.lock:
            self.global_state.hardware.is_connected = True
        log.info("✅ 硬體控制執行緒已啟動。")
        return True

    def stop(self) -> None:
        if not self._is_running.is_set():
            return
        log.info("正在停止硬體控制器...")
        self._is_running.clear()
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"stop\n")
                time.sleep(0.05)
                self.ser.write(b"monitor h\n")
            except serial.SerialException:
                log.warning("發送停止指令失敗。")
        if self._control_thread:
            self._control_thread.join(timeout=1)
        if self._read_thread:
            self._read_thread.join(timeout=1)
        self.serial_comm.is_managed_by_hardware_controller = False
        self.ser = None
        with self.global_state.lock:
            self.global_state.hardware.is_connected = False
            self.global_state.hardware.ai_is_active = False
        log.info("硬體控制器已完全停止。")

    # ----------------------
    # internal helpers
    # ----------------------
    def _parse_policy_stream(self, line: str) -> None:
        """解析帶 CRC8 的 42 欄位 CSV 字串"""
        if self._partial_line:
            line = self._partial_line + line
            self._partial_line = ""

        parts = self._csv_regex.split(line.strip())
        if len(parts) < self._EXPECTED_FIELDS:
            self._partial_line = line
            return
        if len(parts) > self._EXPECTED_FIELDS:
            parts = parts[:self._EXPECTED_FIELDS]
        if len(parts) != self._EXPECTED_FIELDS:
            self._mismatch_count += 1
            if self._mismatch_count % 10 == 1:
                log.warning(f"欄位長度不符: {len(parts)} (需 {self._EXPECTED_FIELDS})")
            return
        try:
            data_parts = parts[:40]
            received_crc = int(parts[40])
            data_bytes = bytearray()
            for p in data_parts:
                data_bytes.extend(np.float32(p).tobytes())
            calc_crc = crc8_maxim(data_bytes)
            if received_crc != calc_crc:
                self._bad_crc_count += 1
                if self._bad_crc_count % 10 == 1:
                    log.warning(f"CRC錯誤 Recv:{received_crc} Calc:{calc_crc} (累計{self._bad_crc_count})")
                return
            data_vec = np.array(data_parts, dtype=np.float32)
            with self._lock:
                self._raw_angular_velocity[:] = data_vec[0:3]
                self._raw_gravity_vector[:] = data_vec[3:6]
                self._raw_linear_velocity[:] = data_vec[6:9]
                self._raw_accel[:] = data_vec[9:12]
                self._raw_joint_positions[:] = data_vec[12:24]
                self._raw_joint_velocities[:] = data_vec[24:36]
                self._last_update_time = time.time()
        except (ValueError, IndexError) as e:
            log.warning(f"解析失敗: {e}")

    def _read_loop(self) -> None:
        while self._is_running.is_set():
            if not self.ser or not self.ser.is_open:
                self.stop()
                break
            try:
                line = self.ser.readline().decode('utf-8', 'ignore').strip()
                if line:
                    self._parse_policy_stream(line)
            except (serial.SerialException, OSError):
                log.error("❌ 序列埠斷開連接，停止硬體控制器。")
                self.stop()
                break

    def _control_loop(self) -> None:
        default_pose = self.global_state.sim.default_pose if self.global_state.sim else np.zeros(12)
        while self._is_running.is_set():
            loop_start = time.perf_counter()
            with self._lock, self.global_state.lock:
                state = self.global_state
                # 將解析後的感測值複製到全域狀態，供 UI 顯示
                state.hardware.angular_velocity_radps = self._raw_angular_velocity.copy()
                state.hardware.gravity_vector = self._raw_gravity_vector.copy()
                state.hardware.linear_velocity = self._raw_linear_velocity.copy()
                state.hardware.accelerometer = self._raw_accel.copy()
                state.hardware.joint_positions_rad = self._raw_joint_positions.copy()
                state.hardware.joint_velocities_radps = self._raw_joint_velocities.copy()
                state.hardware.last_update_time = self._last_update_time
                state.hardware.crc_error_count = self._bad_crc_count
                state.hardware.mismatch_count = self._mismatch_count
                sub_mode = state.control_sub_mode
                ai_active = sub_mode in (ControlSubMode.WALKING, ControlSubMode.FLOATING)
                state.hardware.ai_is_active = ai_active

            command_to_send = None
            onnx_input = np.array([])
            action_raw = np.zeros(12)
            final_cmd = np.zeros(12)

            if sub_mode in (ControlSubMode.WALKING, ControlSubMode.FLOATING):
                obs_components = {
                    'angular_velocity': self._raw_angular_velocity,
                    'gravity_vector': self._raw_gravity_vector,
                    'linear_velocity': self._raw_linear_velocity,
                    'accelerometer': self._raw_accel,
                    'joint_positions': self._raw_joint_positions,
                    'joint_velocities': self._raw_joint_velocities,
                    'last_action': self._last_action,
                    'commands': state.command * self.config.command_scaling_factors,
                }
                recipe = self.policy.get_active_recipe()
                try:
                    obs_list = [obs_components[key] for key in recipe]
                    onnx_input = np.concatenate(obs_list).astype(np.float32)
                except KeyError as e:
                    log.error(f"硬體觀察向量構建失敗: 缺少 {e}")
                    continue
                _, action_raw = self.policy.get_action_for_hardware(onnx_input)
                self._last_action[:] = action_raw
                final_cmd = default_pose + action_raw * state.tuning_params.action_scale
            elif sub_mode == ControlSubMode.JOINT_TEST:
                final_cmd = default_pose + state.joint_test_offsets
            elif sub_mode == ControlSubMode.MANUAL_CTRL:
                final_cmd = state.manual_final_ctrl
            elif sub_mode == ControlSubMode.IDLE:
                command_to_send = "stop\n"
                final_cmd = default_pose

            if command_to_send is None:
                action_str = ' '.join(f"{a:.4f}" for a in final_cmd)
                command_to_send = f"move all {action_str}\n"

            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException:
                    self.stop()

            with self.global_state.lock:
                state.hardware.latest_onnx_input = onnx_input.copy()
                state.hardware.latest_action_raw = action_raw.copy()
                state.hardware.latest_final_ctrl = final_cmd.copy()

            loop_duration = time.perf_counter() - loop_start
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
