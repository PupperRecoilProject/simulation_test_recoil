# serial_communicator.py
import serial
import time
import threading
import serial.tools.list_ports
from utils.serial_utils import select_serial_port
from utils.logger import log


class SerialCommunicator:
    """統一管理序列埠連線的服務"""

    def __init__(self, max_log_lines: int = 50):
        self.ser: serial.Serial | None = None
        self.read_thread: threading.Thread | None = None
        self.exit_signal = threading.Event()
        self.is_connected = False
        self.port_name: str | None = None
        self.is_managed_by_hardware_controller = False
        log.info("序列埠通訊器已初始化 (等待連接指令)。")

    # --------------------------------------------------------------
    # connection helpers
    # --------------------------------------------------------------
    def get_serial_connection(self) -> serial.Serial | None:
        """返回已建立的 serial.Serial 物件，供 HardwareController 使用。"""
        if self.is_connected:
            return self.ser
        return None

    def scan_and_connect(self) -> bool:
        """掃描並連接序列埠。"""
        if self.is_connected:
            log.info("序列埠已連接，無需重新掃描。")
            return True
        selected_port = self._select_serial_port()
        if selected_port:
            self.port_name = selected_port
            return self.connect()
        return False

    def _select_serial_port(self):
        return select_serial_port()

    def connect(self, baud_rate: int = 115200) -> bool:
        """連接到指定序列埠並啟動讀取執行緒。"""
        if not self.port_name:
            return False
        try:
            log.info(f"正在連接到 {self.port_name}...")
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            self.exit_signal.clear()
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            self.is_connected = True
            log.info(f"✅ 序列埠 {self.port_name} 連接成功。")
            return True
        except serial.SerialException as e:
            log.error(f"❌ 序列埠連接失敗: {e}")
            self.is_connected = False
            return False

    # --------------------------------------------------------------
    # background reading
    # --------------------------------------------------------------
    def _read_from_port(self) -> None:
        """背景執行緒：持續從序列埠讀取資料。"""
        while not self.exit_signal.is_set():
            if self.is_managed_by_hardware_controller:
                time.sleep(0.1)
                continue
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8', 'ignore').strip()
                    if response:
                        log.info(f"[Teensy]: {response}")
            except serial.SerialException:
                log.error("[ERROR] Serial port disconnected.")
                self.is_connected = False
                break
            time.sleep(0.01)

    # --------------------------------------------------------------
    # send / attach / detach
    # --------------------------------------------------------------
    def send_command(self, command: str) -> None:
        """向序列埠發送一個字串指令 (未接管時使用)。"""
        if self.is_connected and command and not self.is_managed_by_hardware_controller:
            try:
                self.ser.write((command + '\n').encode('utf-8'))
            except serial.SerialException as e:
                log.error(f"[ERROR] Send failed: {e}")
                self.is_connected = False

    def attach_serial(self, ser: serial.Serial) -> None:
        """讓 HardwareController 接管序列埠，本類進入旁路模式"""
        self.is_managed_by_hardware_controller = True
        self.ser = ser  # 重用相同 serial 物件，避免重開

    def detach_serial(self) -> None:
        self.is_managed_by_hardware_controller = False
        if self.ser:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    # --------------------------------------------------------------
    # teardown
    # --------------------------------------------------------------
    def close(self) -> None:
        """安全地關閉序列埠和讀取執行緒。"""
        if self.is_managed_by_hardware_controller:
            return

        if self.read_thread and self.read_thread.is_alive():
            self.exit_signal.set()
            self.read_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            log.info(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False

