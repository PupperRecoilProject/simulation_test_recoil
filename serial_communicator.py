# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from serial_utils import select_serial_port
from collections import deque
from logger import log

class SerialCommunicator:
    """
    【修改版】一個類別，統一管理序列埠的連接與通訊。
    它作為唯一的連接建立者，可以將已建立的連接「出借」給其他模組（如HardwareController）使用。
    """
    def __init__(self, max_log_lines=50): # 【容量加大】將日誌行數從 15 增加到 50
        """初始化通訊器。"""
        self.ser = None
        self.read_thread = None
        self.exit_signal = threading.Event()
        self.is_connected = False
        self.port_name = None
        self.is_managed_by_hardware_controller = False
        log.info("序列埠通訊器已初始化 (等待連接指令)。")

    def get_serial_connection(self) -> serial.Serial | None:
        """返回已建立的 serial.Serial 物件，供 HardwareController 使用。"""
        if self.is_connected: # 如果已連接
            return self.ser # 返回序列埠物件
        return None # 否則返回 None

    def scan_and_connect(self) -> bool:
        """掃描、讓使用者選擇並連接序列埠。"""
        if self.is_connected: # 如果已連接
            log.info("序列埠已連接，無需重新掃描。")
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
            log.info(f"正在連接到 {self.port_name}...")
            self.ser = serial.Serial(self.port_name, baud_rate, timeout=0.1) # 建立序列埠物件
            time.sleep(0.5) # 等待硬體初始化
            self.ser.reset_input_buffer() # 清空輸入緩衝區
            self.ser.reset_output_buffer() # 清空輸出緩衝區
            
            self.exit_signal.clear() # 重置退出信號
            self.read_thread = threading.Thread(target=self._read_from_port, daemon=True) # 建立讀取執行緒
            self.read_thread.start() # 啟動執行緒
            self.is_connected = True # 設定連接旗標
            log.info(f"✅ 序列埠 {self.port_name} 連接成功。")
            return True
        except serial.SerialException as e: # 捕捉連接錯誤
            log.error(f"❌ 序列埠連接失敗: {e}")
            self.is_connected = False
            return False

    def _read_from_port(self):
        """[背景執行緒函式] 持續地從序列埠讀取數據並存入日誌。"""
        while not self.exit_signal.is_set(): # 當未收到退出信號時
            if self.is_managed_by_hardware_controller: # 如果控制權已交給硬體控制器
                time.sleep(0.1) # 短暫休眠，避免資源競爭
                continue # 繼續下一輪迴圈
                
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8', 'ignore').strip()
                    if response:
                        log.info(f"[Teensy]: {response}")
            except serial.SerialException:
                log.error("[ERROR] Serial port disconnected.")
                self.is_connected = False
                break
            time.sleep(0.01) # 短暫休眠

    def send_command(self, command: str):
        """向序列埠發送一個字串指令。"""
        if self.is_connected and command and not self.is_managed_by_hardware_controller:
            try:
                command_to_send = command + '\n'
                self.ser.write(command_to_send.encode('utf-8'))
            except serial.SerialException as e:
                log.error(f"[ERROR] Send failed: {e}")
                self.is_connected = False


    def close(self):
        """安全地關閉序列埠和讀取執行緒。"""
        if self.is_managed_by_hardware_controller: return # 如果硬體控制器正在管理連接，則本類別不應關閉它

        if self.read_thread and self.read_thread.is_alive(): # 如果讀取執行緒在運行
            self.exit_signal.set() # 發送退出信號
            self.read_thread.join(timeout=1) # 等待執行緒結束
        if self.ser and self.ser.is_open: # 如果序列埠已開啟
            self.ser.close() # 關閉序列埠
            log.info(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False # 更新連接狀態