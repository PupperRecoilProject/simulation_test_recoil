# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from collections import deque

class SerialCommunicator:
    """
    一個類別，封裝了與序列埠設備的通訊邏輯。
    它使用背景執行緒來非阻塞地讀取數據。
    """
    def __init__(self, max_log_lines=15):
        """
        初始化通訊器，掃描並讓使用者選擇埠。
        """
        self.ser = None
        self.read_thread = None
        self.exit_signal = threading.Event()
        self.is_connected = False
        
        # 使用 deque 來儲存最新的幾行日誌，用於在畫面上顯示
        self.message_log = deque(maxlen=max_log_lines)
        
        self.port_name = self._select_serial_port()
        if self.port_name:
            self.connect()

    def _select_serial_port(self):
        """
        掃描並讓使用者選擇序列埠。優先自動檢測 Teensy。
        返回選定的序列埠名稱，如果找不到則返回 None。
        """
        print("正在掃描可用的序列埠...")
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("--- 警告: 未找到任何序列埠。序列埠功能將被禁用。 ---")
            return None

        # 嘗試根據 VID/PID 自動識別 Teensy
        teensy_ports = [p for p in ports if p.vid == 0x16C0 and p.pid == 0x0483]
        if len(teensy_ports) == 1:
            print(f"自動檢測到 Teensy: {teensy_ports[0].device}")
            return teensy_ports[0].device
        
        # 如果自動識別失敗，讓使用者手動選擇
        print("\n請從以下列表中選擇您的 Teensy 設備:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
        while True:
            try:
                # 允許使用者直接按 Enter 跳過選擇
                choice_str = input(f"請輸入選擇的編號 (0-{len(ports)-1}) 或直接按 Enter 跳過: ")
                if not choice_str:
                    print("已跳過序列埠選擇。")
                    return None
                choice = int(choice_str)
                if 0 <= choice < len(ports):
                    return ports[choice].device
                else:
                    print("輸入無效，請重新輸入。")
            except (ValueError, IndexError):
                print("輸入無效，請輸入列表中的數字。")

    def connect(self, baud_rate=115200):
        """連接到指定的序列埠並啟動讀取執行緒。"""
        if not self.port_name: return
        try:
            print(f"\n正在連接到 {self.port_name}...")
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1)
            time.sleep(0.5) # 等待設備重啟
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print("✅ 序列埠連接成功。")

            self.exit_signal.clear()
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True)
            self.read_thread.start()
            self.is_connected = True

        except serial.SerialException as e:
            print(f"❌ 序列埠連接失敗: {e}")
            self.is_connected = False

    def _read_from_port(self):
        """在背景執行緒中讀取數據並存入日誌。"""
        while not self.exit_signal.is_set():
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8', 'ignore').strip()
                    if response:
                        self.message_log.append(response) # 將訊息加入日誌
            except serial.SerialException:
                self.message_log.append("[錯誤] 序列埠已斷開。")
                self.is_connected = False
                break
            time.sleep(0.01)

    def send_command(self, command: str):
        """向序列埠發送指令。"""
        if self.is_connected and command:
            try:
                command_to_send = command + '\n'
                self.ser.write(command_to_send.encode('utf-8'))
                self.message_log.append(f"> {command}") # 將發送的指令也加入日誌
            except serial.SerialException as e:
                 self.message_log.append(f"[錯誤] 發送失敗: {e}")
                 self.is_connected = False


    def get_latest_messages(self) -> list:
        """獲取日誌中的所有訊息。"""
        return list(self.message_log)

    def close(self):
        """安全地關閉序列埠和讀取執行緒。"""
        if self.read_thread:
            self.exit_signal.set()
            if self.read_thread.is_alive():
                self.read_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False