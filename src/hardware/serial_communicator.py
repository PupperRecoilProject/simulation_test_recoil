# serial_communicator.py
import serial
import time
import sys
import threading
import serial.tools.list_ports
from src.utils.serial_utils import select_serial_port
from collections import deque
from src.core.logger import log

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
        # 【v4.10.2 新增】用於執行緒間安全握手的新旗標
        self._pause_requested = threading.Event() # 主執行緒用來請求暫停
        self._is_paused_flag = threading.Event()  # 背景執行緒用來宣告自己已暫停
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

    def relinquish_control(self, timeout: float = 2.0) -> bool:
        """
        【v4.10.2 新增】請求背景讀取執行緒暫停，並阻塞式地等待其確認。

        這是實現安全資源交接的核心「握手」函式。

        Args:
            timeout (float): 等待背景執行緒確認暫停的超時時間（秒）。

        Returns:
            bool: 如果背景執行緒在超時內確認暫停，則返回 True，否則返回 False。
        """
        if not self.read_thread or not self.read_thread.is_alive():
            log.warning("SerialCommunicator 的讀取執行緒未運行，無需交接控制權。")
            return True # 如果執行緒本來就沒在跑，視為交接成功

        log.info("正在請求 SerialCommunicator 暫停其讀取執行緒...")
        self._is_paused_flag.clear() # 先放下「已暫停」的旗子
        self._pause_requested.set() # 升起「請求暫停」的旗子

        # 等待背景執行緒升起它自己的「我已暫停」的旗子
        is_paused = self._is_paused_flag.wait(timeout=timeout)

        if is_paused:
            log.info("✅ SerialCommunicator 已確認暫停，控制權已安全交接。")
            return True
        else:
            log.error(f"❌ 等待 SerialCommunicator 暫停超時 ({timeout}s)！控制權交接失敗。")
            self._pause_requested.clear() # 取消暫停請求
            return False

    def resume_control(self):
        """【v4.10.2 新增】通知背景讀取執行緒恢復正常運作。"""
        log.info("正在將序列埠控制權歸還給 SerialCommunicator...")
        self._pause_requested.clear()
        self._is_paused_flag.clear()

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
        """
        【v4.10.2 修改】增加了響應暫停請求的邏輯。
        
        [背景執行緒函式] 持續地從序列埠讀取數據並存入日誌。
        """
        while not self.exit_signal.is_set():
            # --- 【v4.10.2 新增】紅綠燈檢查邏輯 ---
            if self._pause_requested.is_set():
                if not self._is_paused_flag.is_set():
                    self._is_paused_flag.set() # 升起「我已暫停」的旗子，通知主執行緒
                time.sleep(0.1) # 保持暫停狀態，等待請求被取消
                continue

            # 如果暫停請求被取消，確保「我已暫停」的旗子也被放下
            if self._is_paused_flag.is_set():
                self._is_paused_flag.clear()
            # ------------------------------------
                
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
        """
        【v4.10.2 修改】在發送前增加暫停狀態檢查。

        向序列埠發送一個字串指令。
        """
        # 【v4.10.2 修改】使用 _pause_requested 旗標取代舊旗標
        if self.is_connected and command and not self._pause_requested.is_set():
            try:
                command_to_send = command + '\n'
                self.ser.write(command_to_send.encode('utf-8'))
            except serial.SerialException as e:
                log.error(f"[ERROR] Send failed: {e}")
                self.is_connected = False


    def close(self):
        """
        【v4.10.2 修改】在關閉前確保執行緒已恢復，避免死鎖。
        
        安全地關閉序列埠和讀取執行緒。
        """
        # 【v4.10.2 修改】使用 _pause_requested 旗標取代舊旗標
        if self._pause_requested.is_set():
            log.warning("SerialCommunicator 處於暫停狀態，可能由 HardwareController 管理。跳過關閉操作。")
            return

        if self.read_thread and self.read_thread.is_alive(): # 如果讀取執行緒在運行
            self.exit_signal.set() # 發送退出信號
            self.read_thread.join(timeout=1) # 等待執行緒結束
        if self.ser and self.ser.is_open: # 如果序列埠已開啟
            self.ser.close() # 關閉序列埠
            log.info(f"序列埠 {self.port_name} 已安全關閉。")
        self.is_connected = False # 更新連接狀態