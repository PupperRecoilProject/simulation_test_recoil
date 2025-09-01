# src/hardware/teensy_api.py
import serial
import numpy as np
from src.core.logger import log

class TeensyAPI:
    """【v4.9.0 新增】Teensy 通訊協定封裝層。
    
    這個類別是與 Teensy 硬體進行指令通訊的唯一介面，
    將所有 "魔法字串" 指令封裝成有意義的函式，以提升程式碼的
    可讀性、可維護性與職責清晰度。
    """
    def __init__(self, ser_instance: serial.Serial):
        """
        初始化 TeensyAPI。

        Args:
            ser_instance (serial.Serial): 一個已經開啟的 Pyserial 連線實例。
        """
        self.ser = ser_instance

    def _send_command(self, command: str) -> bool:
        """統一的指令發送輔助函式，包含錯誤處理。"""
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