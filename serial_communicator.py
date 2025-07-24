# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from serial_utils import select_serial_port
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
        if self.is_connected: # 如果已連接
            return self.ser # 返回序列埠物件
        return None # 否則返回 None

    def scan_and_connect(self) -> bool:
        """掃描、讓使用者選擇並連接序列埠。"""
        if self.is_connected: # 如果已連接
            print("序列埠已連接，無需重新掃描。")
            return True
            
        selected_port = self._select_serial_port() # 讓使用者選擇序列埠
        if selected_port: # 如果選擇了
            self.port_name = selected_port # 儲存埠名
            return self.connect() # 執行連接
        return False

    def _select_serial_port(self):
        """掃描並在終端機列出所有可用的序列埠供使用者選擇。"""
        return select_serial_port() # 呼叫工具函式

    def connect(self, baud_rate=115200) -> bool:
        """連接到指定的序列埠並啟動讀取執行緒。"""
        if not self.port_name: return False # 如果沒有埠名，返回失敗
        try:
            print(f"正在連接到 {self.port_name}...")
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1) # 建立序列埠物件
            time.sleep(0.5) # 等待硬體初始化
            self.ser.reset_input_buffer() # 清空輸入緩衝區
            self.ser.reset_output_buffer() # 清空輸出緩衝區
            
            self.exit_signal.clear() # 重置退出信號
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True) # 建立讀取執行緒
            self.read_thread.start() # 啟動執行緒
            self.is_connected = True # 設定連接旗標
            print(f"✅ 序列埠 {self.port_name} 連接成功。")
            return True
        except serial.SerialException as e: # 捕捉連接錯誤
            print(f"❌ 序列埠連接失敗: {e}")
            self.is_connected = False
            return False

    def _read_from_port(self):
        """[背景執行緒函式] 持續地從序列埠讀取數據並存入日誌。"""
        while not self.exit_signal.is_set(): # 當未收到退出信號時
            # 【修改】如果控制權已交給硬體控制器，則此執行緒進入休眠，避免資源衝突
            if self.is_managed_by_hardware_controller:
                time.sleep(0.1) # 短暫休眠
                continue # 繼續下一輪迴圈
                
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0: # 如果序列埠可用且有數據
                    response = self.ser.readline().decode('utf-8', 'ignore').strip() # 讀取一行數據
                    if response: # 如果讀到內容
                        self.message_log.append(response) # 加入日誌
            except serial.SerialException: # 捕捉序列埠錯誤
                # 【中文化修正】將添加到日誌中的錯誤訊息改為英文
                self.message_log.append("[ERROR] Serial port disconnected.")
                self.is_connected = False # 更新連接狀態
                break # 退出迴圈
            time.sleep(0.01) # 短暫休眠

    def send_command(self, command: str):
        """向序列埠發送一個字串指令，僅在 SERIAL_MODE 下有效。"""
        if self.is_connected and command and not self.is_managed_by_hardware_controller: # 檢查發送條件
            try:
                command_to_send = command + '\n' # 加上換行符
                self.ser.write(command_to_send.encode('utf-8')) # 發送指令
                self.message_log.append(f"> {command}") # 將發送的指令也加入日誌
            except serial.SerialException as e: # 捕捉發送錯誤
                 # 【中文化修正】將添加到日誌中的錯誤訊息改為英文
                 self.message_log.append(f"[ERROR] Send failed: {e}")
                 self.is_connected = False # 更新連接狀態

    def get_latest_messages(self) -> list:
        """獲取日誌中的所有訊息，用於UI顯示。"""
        return list(self.message_log) # 返回日誌列表

    def close(self):
        """安全地關閉序列埠和讀取執行緒。"""
        # 如果硬體控制器正在管理連接，則本類別不應關閉它
        if self.is_managed_by_hardware_controller: return

        if self.read_thread and self.read_thread.is_alive(): # 如果讀取執行緒在運行
            self.exit_signal.set() # 發送退出信號
            self.read_thread.join(timeout=1) # 等待執行緒結束
        if self.ser and self.ser.is_open: # 如果序列埠已開啟
            self.ser.close() # 關閉序列埠
            print(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False # 更新連接狀態