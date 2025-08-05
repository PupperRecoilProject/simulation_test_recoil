import serial
import threading
import time
import numpy as np
import struct
import re
from typing import TYPE_CHECKING

from utils.logger import log
from state import OperatingMode, ControlSubMode

if TYPE_CHECKING:  # 型別提示，避免循環匯入
    from utils.config import AppConfig
    from core.policy import PolicyManager
    from state import SimulationState
    from serial_communicator import SerialCommunicator

# -------------------------------------------------------------
# 常數：Teensy 實際傳送 40 個浮點數 + 1 個 CRC 欄位
# -------------------------------------------------------------
EXPECTED_FLOATS = 40                              # 資料欄位數 (不含 CRC)
EXPECTED_CSV_FIELDS = EXPECTED_FLOATS + 1         # 40 floats + CRC
_CSV_REGEX = re.compile(r"[,\s]+")              # 允許逗號或空白分隔


def _crc8(data: bytes) -> int:
    """計算簡易 CRC‑8: poly = x^8 + x^2 + x + 1 (0x07)"""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


class HardwareController:
    """重構版硬體控制器，已適配 40 維數據流並具有 CRC 驗證。"""

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
        self._partial_line: list[str] = []  # 斷包暫存

        # --- 最新感測資料緩衝 (符合 40 維定義) ---
        self._raw_angular_velocity = np.zeros(3)
        self._raw_gravity_vector = np.zeros(3)
        self._raw_linear_velocity = np.zeros(3)
        self._raw_accelerometer = np.zeros(3)
        self._raw_joint_positions = np.zeros(12)
        self._raw_joint_velocities = np.zeros(12)
        self._last_action = np.zeros(12)
        self._last_update_time = 0.0

        log.info("✅ 重構版硬體控制器已初始化 (40-dim stream)。")

    @property
    def is_running(self) -> bool:
        """公開查詢是否運行中的屬性"""
        return self._is_running.is_set()

    # -------------------------------------------------------------
    # lifecycle 生命週期
    # -------------------------------------------------------------
    def attach_serial(self, ser: serial.Serial | None) -> None:
        """由 SerialCommunicator 呼叫，將已連線的序列埠交給硬體控制器。"""
        self.ser = ser
        if ser is not None:
            self.serial_comm.attach_serial(ser)

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
        self.serial_comm.attach_serial(self.ser)
        try:
            log.info("-> 正在命令 Teensy 切換至 POLICY_STREAM 模式...")
            self.ser.write(b"monitor p\n")
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            log.error(f"❌ 發送模式切換指令失敗: {e}")
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
        self.serial_comm.detach_serial()
        self.ser = None
        with self.global_state.lock:
            self.global_state.hardware.is_connected = False
            self.global_state.hardware.ai_is_active = False
        log.info("硬體控制器已完全停止。")

    # -------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------
    def _parse_policy_stream(self, line: str) -> None:
        """解析來自 Teensy 的 CSV 資料並進行 CRC 驗證"""
        parts = _CSV_REGEX.split(line.strip())

        # 1) 若上輪有殘包，先拼接
        if self._partial_line:
            parts = self._partial_line + parts
            self._partial_line = []

        # 2) 欄位不足：暫存待補
        if len(parts) < EXPECTED_CSV_FIELDS:
            self._partial_line = parts
            return

        # 3) 欄位過多：截斷並記錄
        if len(parts) > EXPECTED_CSV_FIELDS:
            with open("bad_lines.log", "a", encoding='utf-8') as f:
                f.write(f"LONG_LINE: {line}\n")
            parts = parts[:EXPECTED_CSV_FIELDS]

        try:
            float_values = [float(p) for p in parts[:-1]]  # 先解析所有浮點數
            crc_from_teensy = int(parts[-1]) & 0xFF        # 最後一欄為 CRC
        except ValueError as e:
            with open("bad_lines.log", "a", encoding='utf-8') as f:
                f.write(f"PARSE_ERROR: {line} | {e}\n")
            with self.global_state.lock:
                self.global_state.hardware.mismatch_count += 1
                self.global_state.hardware.status_text = "Parse Error!"
            return

        if len(float_values) != EXPECTED_FLOATS:
            with open("bad_lines.log", "a", encoding='utf-8') as f:
                f.write(f"DIM_MISMATCH: {line}\n")
            with self.global_state.lock:
                self.global_state.hardware.mismatch_count += 1
                self.global_state.hardware.status_text = "Data dim mismatch!"
            return

        float_bytes = struct.pack('<' + 'f'*EXPECTED_FLOATS, *float_values)
        calculated_crc = _crc8(float_bytes)
        if calculated_crc != crc_from_teensy:
            with self.global_state.lock:
                self.global_state.hardware.crc_error_count += 1
                self.global_state.hardware.status_text = (
                    f"CRC Error! PC:{calculated_crc} != Teensy:{crc_from_teensy}")
            with open("bad_lines.log", "a", encoding='utf-8') as f:
                f.write(
                    f"CRC_ERROR: {line} | PC_CRC:{calculated_crc} | TEENSY_CRC:{crc_from_teensy}\n"
                )
            return

        data_vec = np.frombuffer(float_bytes, dtype=np.float32)
        with self._lock:
            self._raw_angular_velocity[:] = data_vec[0:3]
            self._raw_gravity_vector[:] = data_vec[3:6]
            self._raw_linear_velocity[:] = data_vec[6:9]
            self._raw_accelerometer[:] = data_vec[9:12]
            self._raw_joint_positions[:] = data_vec[12:24]
            self._raw_joint_velocities[:] = data_vec[24:36]
            # 36~39 為保留欄位，目前忽略
            self._last_update_time = time.time()  # 只有 CRC 成功才更新時間
            with self.global_state.lock:
                self.global_state.hardware.status_text = "OK"

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

            # 先同步最新感測值到全域狀態，並抓取目前的控制子模式
            with self._lock, self.global_state.lock:
                state = self.global_state
                state.hardware.angular_velocity_radps = self._raw_angular_velocity.copy()
                state.hardware.gravity_vector = self._raw_gravity_vector.copy()
                state.hardware.linear_velocity = self._raw_linear_velocity.copy()
                state.hardware.accelerometer = self._raw_accelerometer.copy()
                state.hardware.joint_positions_rad = self._raw_joint_positions.copy()
                state.hardware.joint_velocities_radps = self._raw_joint_velocities.copy()
                state.hardware.last_update_time = self._last_update_time
                sub_mode = state.control_sub_mode
                state.hardware.ai_is_active = (
                    sub_mode in (ControlSubMode.WALKING, ControlSubMode.FLOATING)
                )

            # --- 無論 AI 是否啟動，都先建構 ONNX 觀察向量 ---
            obs_components = {
                'angular_velocity': self._raw_angular_velocity,
                'gravity_vector': self._raw_gravity_vector,
                'linear_velocity': self._raw_linear_velocity,
                'accelerometer': self._raw_accelerometer,
                'joint_positions': self._raw_joint_positions,
                'joint_velocities': self._raw_joint_velocities,
                'last_action': self._last_action,
                # commands 仍需納入，以確保維度一致
                'commands': state.command * self.config.command_scaling_factors,
            }
            recipe = self.policy.get_active_recipe()
            try:
                obs_list = [obs_components[key] for key in recipe]
                onnx_input = np.concatenate(obs_list).astype(np.float32)
            except KeyError as e:
                log.error(f"硬體觀察向量構建失敗: 缺少 '{e}'")
                onnx_input = np.array([])

            action_raw = np.zeros(12)
            final_cmd = np.zeros(12)
            command_to_send = None

            if sub_mode in (ControlSubMode.WALKING, ControlSubMode.FLOATING):
                # 只有在 AI 模式下才進行 ONNX 推論
                if onnx_input.size == 0:
                    time.sleep(0.02)
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

            # 若尚未決定指令字串，則根據 final_cmd 產生 move 指令
            if command_to_send is None:
                action_str = ' '.join(f"{v:.4f}" for v in final_cmd)
                command_to_send = f"move all {action_str}\n"

            # 實際寫入序列埠
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(command_to_send.encode('utf-8'))
                except serial.SerialException:
                    self.stop()

            # 更新全域狀態，供 UI 即時顯示
            with self.global_state.lock:
                state.hardware.latest_onnx_input = onnx_input.copy()
                state.hardware.latest_action_raw = action_raw.copy()
                state.hardware.latest_final_ctrl = final_cmd.copy()

            loop_duration = time.perf_counter() - loop_start
            sleep_time = (1.0 / self.config.control_freq) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
