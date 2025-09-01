# src/hardware/teensy_api.py
import serial
import numpy as np
import time
from src.core.logger import log
from typing import TYPE_CHECKING

# 【v4.10.1 新增】導入 State 和 Enum 以進行類型提示
from src.core.state import SimulationState, HardwareLinkStatus

if TYPE_CHECKING:
    from src.hardware.serial_communicator import SerialCommunicator

class TeensyAPI:
    """
    【v4.9.0 新增】Teensy 通訊協定封裝層。
    【v4.10.1 修改】...加入了阻塞式回傳驗證和基於狀態機的安全模式。
    
    這個類別是與 Teensy 硬體進行指令通訊的唯一介面，
    將所有 "魔法字串" 指令封裝成有意義的函式，以提升程式碼的
    可讀性、可維護性與職責清晰度。
    """
    def __init__(self, serial_comm: 'SerialCommunicator', state: 'SimulationState'):
        """
        【v4.10.1 修改】新增 state 參數以讀取安全模式旗標。

        初始化 TeensyAPI。

        Args:
            ser_instance (serial.Serial): 一個已經開啟的 Pyserial 連線實例。
        """
        self.serial_comm = serial_comm
        self.state = state
        self.ser: serial.Serial | None = None

    def _send_command(self, command: str) -> bool:
        """
        【v4.10.1 修改】增加基於 `hardware_link_status` 狀態機的安全守衛。
        
        統一的指令發送輔助函式，包含錯誤處理。
        """
        # 只有在連結狀態為 'VERIFIED' 時，指令才會被實際發送。
        if self.state.hardware_link_status != HardwareLinkStatus.VERIFIED:
            log.info(f"[{self.state.hardware_link_status.value}] 指令已靜默: '{command}'")
            return True # 假裝成功，讓上層邏輯可以繼續
        
        if self.ser and self.ser.is_open:
            try:
                # 將指令加上換行符並編碼為 UTF-8 發送
                self.ser.write((command + '\n').encode('utf-8'))
                log.debug(f"已發送指令給 Teensy: '{command}'")
                return True
            except serial.SerialException as e:
                log.error(f"發送指令 '{command}' 到 Teensy 失敗: {e}")
                return False
        else:
            log.warning(f"因序列埠未連接，無法發送指令: '{command}'")
            return False
        
    def send_command_and_wait_for_ok(self, command: str, timeout: float = 1.0) -> bool:
        """
        【v4.10.1 新增】發送一個指令，並阻塞式地等待 Teensy 回應 '[OK]'。
        
        這個函式對於需要確認執行的關鍵指令（如模式切換）至關重要。
        它會暫時忽略 Mute 狀態來發送指令，專門用於啟動驗證流程。

        Args:
            command (str): 要發送的指令。
            timeout (float): 等待回應的超時時間（秒）。

        Returns:
            bool: 如果在超時內收到了包含 '[OK]' 的回應，則返回 True，否則返回 False。
        """
        # 步驟 1: 直接透過 pyserial 發送，繞過 _send_command 的靜默檢查
        try:
            if not (self.ser and self.ser.is_open):
                log.warning(f"無法發送驗證指令 '{command}'：序列埠未連接。")
                return False
            self.ser.write((command + '\n').encode('utf--8'))
            log.debug(f"已發送驗證指令給 Teensy: '{command}'")
        except serial.SerialException as e:
            log.error(f"發送驗證指令 '{command}' 到 Teensy 失敗: {e}")
            return False

        # 步驟 2: 阻塞式等待回應
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    log.debug(f"等待回應中，收到: '{line}'")
                    # 我們只關心是否包含 '[OK]'，忽略大小寫和前後文
                    if '[ok]' in line.lower():
                        log.info(f"指令 '{command}' 已收到 '[OK]' 確認。")
                        return True
                except Exception:
                    pass # 忽略解碼錯誤的行
            time.sleep(0.01) # 短暫休眠，避免 CPU 佔用過高
        
        log.warning(f"等待指令 '{command}' 的 '[OK]' 回應超時 ({timeout}s)。")
        return False


    def set_telemetry_frequency(self, hz: int) -> bool:
        """【v4.10.1 新增】指令：設定遙測數據的更新頻率。"""
        return self._send_command(f"monitor freq {hz}")

    def set_mode_policy_stream(self) -> bool:
        """指令：要求 Teensy 切換到 POLICY_STREAM 模式，開始發送數據幀。"""
        return self._send_command("monitor p")

    def set_mode_human_readable(self) -> bool:
        """指令：要求 Teensy 切換回人類可讀的監控模式。"""
        return self._send_command("monitor h")
        
    def stop_ai_and_motors(self) -> bool:
        """指令：要求 Teensy 停止 AI 控制並將馬達力矩歸零。"""
        return self._send_command("stop")

    def send_motor_commands(self, angles: np.ndarray) -> bool:
        """指令：發送所有馬達的目標角度。"""
        action_str = ' '.join(f"{a:.4f}" for a in angles)
        return self._send_command(f"move all {action_str}")