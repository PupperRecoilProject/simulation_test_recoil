# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from collections import deque

class SerialCommunicator:
    """
    一個類別，封裝了與序列埠設備（例如 Teensy 或 Arduino）的通訊邏輯。
    它使用一個背景執行緒來非阻塞地讀取數據，避免主程式被I/O操作卡住。
    """
    def __init__(self, max_log_lines=15):
        """
        初始化通訊器，但不立即連接。
        Args:
            max_log_lines (int): 在畫面上顯示的最新訊息的最大行數。
        """
        self.ser = None # serial.Serial 物件，在連接成功後被賦值
        self.read_thread = None # 用於讀取序列埠數據的背景執行緒
        self.exit_signal = threading.Event() # 一個安全地停止背景執行緒的信號
        self.is_connected = False # 標記當前是否已連接
        self.port_name = None # 儲存已連接的序列埠名稱，如 "COM3"
        self.message_log = deque(maxlen=max_log_lines) # 使用雙向佇列儲存最新的訊息日誌
        print("✅ 序列埠通訊器已初始化 (等待連接指令)。")

    def scan_and_connect(self) -> bool:
        """
        掃描系統上所有可用的序列埠，讓使用者在終端機中進行選擇，並嘗試連接。
        返回:
            bool: 連接是否成功。
        """
        if self.is_connected: # 如果已經連接，則無需重複操作
            print("序列埠已連接，無需重新掃描。")
            return True
            
        selected_port = self._select_serial_port() # 呼叫內部函式來掃描並讓使用者選擇
        if selected_port: # 如果使用者選擇了一個埠
            self.port_name = selected_port # 儲存選擇的埠名稱
            return self.connect() # 嘗試連接
        return False # 如果使用者未選擇，返回失敗

    def _select_serial_port(self):
        """
        掃描並在終端機列出所有可用的序列埠，讓使用者選擇。
        會優先自動檢測已知的 Teensy 設備。
        返回:
            str or None: 選定的序列埠名稱，如果找不到或跳過則返回 None。
        """
        print("\n" + "="*20 + " 正在掃描序列埠 " + "="*20)
        ports = serial.tools.list_ports.comports() # 獲取所有可用序列埠的列表
        if not ports: # 如果列表為空
            print("--- 未找到任何序列埠 ---")
            return None

        # 嘗試自動尋找 Teensy (VID=0x16C0, PID=0x0483 是 Teensy 的標準識別碼)
        teensy_ports = [p for p in ports if p.vid == 0x16C0 and p.pid == 0x0483]
        if len(teensy_ports) == 1: # 如果剛好找到一個 Teensy
            print(f"自動檢測到 Teensy: {teensy_ports[0].device}")
            return teensy_ports[0].device
        
        # 如果無法自動檢測，則讓使用者手動選擇
        print("\n請從以下列表中選擇您的設備:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
        while True:
            try:
                choice_str = input(f"請輸入選擇的編號 (0-{len(ports)-1}) 或直接按 Enter 跳過: ")
                if not choice_str: # 如果使用者直接按 Enter
                    print("已跳過序列埠選擇。")
                    return None
                choice = int(choice_str)
                if 0 <= choice < len(ports): # 如果輸入的數字在有效範圍內
                    return ports[choice].device
                else:
                    print("輸入無效，請重新輸入。")
            except (ValueError, IndexError):
                print("輸入無效，請輸入列表中的數字。")

    def connect(self, baud_rate=115200) -> bool:
        """連接到指定的序列埠並啟動讀取執行緒。"""
        if not self.port_name: return False # 如果沒有指定埠名稱，直接失敗
        try:
            print(f"正在連接到 {self.port_name}...")
            # 初始化 serial.Serial 物件，設定埠、波特率和超時
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1)
            time.sleep(0.5) # 等待一小段時間讓序列埠穩定
            self.ser.reset_input_buffer() # 清空輸入緩衝區
            self.ser.reset_output_buffer() # 清空輸出緩衝區
            
            self.exit_signal.clear() # 重置停止信號
            # 建立並啟動一個背景執行緒來持續讀取數據
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
        while not self.exit_signal.is_set(): # 只要沒收到停止信號就一直循環
            try:
                # 檢查序列埠是否正常且有數據在等待讀取
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    # 讀取一行數據，解碼為 utf-8 字串，並移除頭尾的空白字元
                    response = self.ser.readline().decode('utf-8', 'ignore').strip()
                    if response: # 如果讀到了非空內容
                        self.message_log.append(response) # 將其加入訊息日誌
            except serial.SerialException: # 如果發生序列埠錯誤 (例如設備被拔掉)
                self.message_log.append("[錯誤] 序列埠已斷開。")
                self.is_connected = False
                break # 跳出循環，結束執行緒
            time.sleep(0.01) # 短暫休眠，避免 CPU 佔用過高

    def send_command(self, command: str):
        """向序列埠發送一個字串指令。"""
        if self.is_connected and command:
            try:
                command_to_send = command + '\n' # 在指令末尾加上換行符，這是很多設備的指令結束標誌
                self.ser.write(command_to_send.encode('utf-8')) # 將字串編碼為位元組並發送
                self.message_log.append(f"> {command}") # 將發送的指令也記錄到日誌中
            except serial.SerialException as e:
                 self.message_log.append(f"[錯誤] 發送失敗: {e}")
                 self.is_connected = False

    def get_latest_messages(self) -> list:
        """獲取日誌中的所有訊息，用於UI顯示。"""
        return list(self.message_log)

    def close(self):
        """安全地關閉序列埠和讀取執行緒。"""
        if self.read_thread and self.read_thread.is_alive(): # 如果讀取執行緒還在運行
            self.exit_signal.set() # 發送停止信號
            self.read_thread.join(timeout=1) # 等待執行緒結束，最多等待1秒
        if self.ser and self.ser.is_open: # 如果序列埠還開著
            self.ser.close() # 關閉它
            print(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False