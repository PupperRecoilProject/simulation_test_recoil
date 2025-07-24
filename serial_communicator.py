# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from collections import deque

class SerialCommunicator:
    """
    【修改版】一個類別，統一管理序列埠的連接與通訊。
    它作為唯一的連接建立者，可以將已建立的連接「出借」給其他模組（如HardwareController）使用。
    """
    def __init__(self, max_log_lines=15):
        """初始化通訊器。"""
        self.ser = None # serial.Serial 物件，在連接成功後被賦值
        self.read_thread = None # 用於讀取序列埠數據的背景執行緒
        self.exit_signal = threading.Event() # 一個安全地停止背景執行緒的信號
        self.is_connected = False # 標記當前是否已連接
        self.port_name = None # 儲存已連接的序列埠名稱
        self.message_log = deque(maxlen=max_log_lines) # 儲存最新的訊息日誌
        self.is_managed_by_hardware_controller = False # 【新增】旗標，當為True時，表示連接由HardwareController管理，本類別暫停活動
        print("✅ 序列埠通訊器已初始化 (等待連接指令)。")

    def get_serial_connection(self) -> serial.Serial | None:
        """【新增】返回已建立的 serial.Serial 物件，供 HardwareController 使用。"""
        if self.is_connected:
            return self.ser
        return None

    def scan_and_connect(self) -> bool:
        """掃描、讓使用者選擇並連接序列埠。"""
        if self.is_connected:
            print("序列埠已連接，無需重新掃描。")
            return True
            
        selected_port = self._select_serial_port()
        if selected_port:
            self.port_name = selected_port
            return self.connect()
        return False

    def _select_serial_port(self):
        """掃描並在終端機列出所有可用的序列埠供使用者選擇。"""
        print("\n正在掃描可用的序列埠...")
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("--- 錯誤: 未找到任何序列埠。請檢查您的設備連接。 ---")
            return None

        TEENSY_VID = 0x16C0
        TEENSY_PID = 0x0483
        teensy_ports = [p for p in ports if p.vid == TEENSY_VID and p.pid == TEENSY_PID]
        if len(teensy_ports) == 1:
            print(f"自動檢測到 Teensy: {teensy_ports[0].device}")
            return teensy_ports[0].device

        print("\n請從以下列表中選擇您的 Teensy 設備:")
        for i, port in enumerate(ports):
            vid = f"{port.vid:04X}" if port.vid is not None else "----"
            pid = f"{port.pid:04X}" if port.pid is not None else "----"
            print(f"  [{i}] {port.device} - {port.description} (VID:PID={vid}:{pid})")

        while True:
            try:
                choice = int(input(f"請輸入選擇的編號 (0-{len(ports)-1}): "))
                if 0 <= choice < len(ports):
                    return ports[choice].device
                else:
                    print("輸入無效，請重新輸入。")
            except (ValueError, IndexError):
                print("輸入無效，請輸入列表中的數字。")

    def connect(self, baud_rate=115200) -> bool:
        """連接到指定的序列埠並啟動讀取執行緒。"""
        if not self.port_name: return False
        try:
            print(f"正在連接到 {self.port_name}...")
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            self.exit_signal.clear()
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            self.is_connected = True
            print(f"✅ 序列埠 {self.port_name} 連接成功。")
            return True
        except serial.SerialException as e:
            print(f"❌ 序列埠連接失敗: {e}")
            self.is_connected = False
            return False

    def _read_from_port(self):
        """[背景執行緒函式] 持續地從序列埠讀取數據並存入日誌。"""
        while not self.exit_signal.is_set():
            # 【修改】如果控制權已交給硬體控制器，則此執行緒進入休眠，避免資源衝突
            if self.is_managed_by_hardware_controller:
                time.sleep(0.1)
                continue
                
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8', 'ignore').strip()
                    if response:
                        self.message_log.append(response)
            except serial.SerialException:
                self.message_log.append("[錯誤] 序列埠已斷開。")
                self.is_connected = False
                break
            time.sleep(0.01)

    def send_command(self, command: str):
        """向序列埠發送一個字串指令，僅在 SERIAL_MODE 下有效。"""
        if self.is_connected and command and not self.is_managed_by_hardware_controller:
            try:
                command_to_send = command + '\n'
                self.ser.write(command_to_send.encode('utf-8'))
                self.message_log.append(f"> {command}")
            except serial.SerialException as e:
                 self.message_log.append(f"[錯誤] 發送失敗: {e}")
                 self.is_connected = False

    def get_latest_messages(self) -> list:
        """獲取日誌中的所有訊息，用於UI顯示。"""
        return list(self.message_log)

    def close(self):
        """安全地關閉序列埠和讀取執行緒。"""
        # 如果硬體控制器正在管理連接，則本類別不應關閉它
        if self.is_managed_by_hardware_controller: return

        if self.read_thread and self.read_thread.is_alive():
            self.exit_signal.set()
            self.read_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False