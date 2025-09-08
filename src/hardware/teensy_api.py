# src/hardware/teensy_api.py
import serial
import numpy as np
import time
from src.core.logger import log
from typing import TYPE_CHECKING
from enum import Enum

# 【v4.10.1 新增】導入 State 和 Enum 以進行類型提示
from src.core.state import SimulationState, HardwareLinkStatus

if TYPE_CHECKING:
    from src.hardware.serial_communicator import SerialCommunicator

class TeensyAPI:
    """
    【v4.10.4 修改】升級為協定感知的通訊介面層。
    
    本模組將所有 "魔法字串" 指令封裝成有意義的函式，並根據內部定義的
    通訊協定，智能地處理不同指令的回應類型（例如，是否需要等待 [OK] 確認）。
    它還遵循「建構即初始化」原則，確保實例在創建時就完全可用。

    版本演進歷史:
    - v4.10.4: 引入指令協定字典和統一的 execute_command 方法，修正回應驗證邏輯。
    - v4.9.0: 首次引入，將字串指令封裝為函式。
    """
    # 【v4.10.4 新增】定義指令的預期回應類型
    ResponseType = Enum('ResponseType', ['NONE', 'OK'])

    # 【v4.10.4 新增】指令協定字典 (Command Protocol Dictionary)
    # 這是 Teensy 通訊協定的「單一真相來源」。
    # 它將指令映射到其預期的回應類型。
    _COMMAND_PROTOCOLS = {
        'stop': ResponseType.NONE,
        'monitor p': ResponseType.NONE,
        'monitor h': ResponseType.NONE,
        'monitor freq': ResponseType.OK,
        'move all': ResponseType.NONE,
    }

    def __init__(self, serial_comm: 'SerialCommunicator', state: 'SimulationState'):
        """
        【v4.10.1 修改】新增 state 參數以讀取安全模式旗標。
        【v4.10.3 修改】遵循「建構即初始化」原則。

        初始化 TeensyAPI。

        Args:
            ser_instance (serial.Serial): 一個已經開啟的 Pyserial 連線實例。
        """
        self.serial_comm = serial_comm
        self.state = state
        # 【v4.10.3 修改】在建構時立即獲取連線，而不是等待外部呼叫。
        # 這一行是解決 BUG-HW-INIT-RACE 的核心。
        self.ser: serial.Serial | None = self.serial_comm.get_serial_connection()

        # 【v4.10.3 新增】增加日誌，用於在初始化失敗時提供清晰的除錯線索。
        if self.ser is None:
            log.warning("TeensyAPI 在初始化時未能獲取有效的序列埠連線。")

    # 【v4.10.4 新增】新的統一指令執行器
    def execute_command(self, command: str, timeout: float = 1.0) -> bool:
        """
        【v4.10.4 新增】統一的、協定感知的指令執行器。
        
        此方法會自動查詢內部協定字典，以決定應採取的驗證策略
        （例如，僅發送，或發送並等待 [OK]）。
        這是與 Teensy 交互的主要入口點。

        Args:
            command (str): 要發送的指令字串（不含換行符）。
            timeout (float): 等待回應的超時時間（秒），僅對需要回應的指令有效。

        Returns:
            bool: 如果指令成功執行（根據其協定），則返回 True。
        """
        # 從指令中提取主關鍵字以查詢協定
        command_key = command.split(' ')[0]
        protocol = self._COMMAND_PROTOCOLS.get(command_key, self.ResponseType.NONE)

        # 繞過 Mute 檢查，直接發送指令。這是因為 execute_command 主要用於
        # 啟動和停止序列，這些指令必須在任何情況下都能發送。
        if not self._send_raw_command(command):
            return False

        # 根據協定決定後續操作
        if protocol == self.ResponseType.OK:
            return self._wait_for_ok(command, timeout)
        elif protocol == self.ResponseType.NONE:
            # 對於無需回應的指令，只要成功發送即視為成功
            return True
        
        return False # 未知協定

    # 【v4.10.4 重構】將 _send_command 重構為一個更底層的 raw sender
    def _send_raw_command(self, command: str) -> bool:
        """
        【v4.10.4 重構】底層的、無條件的指令發送器。
        
        此私有方法只負責將字串發送到序列埠，不包含任何協定或安全狀態檢查。
        """
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((command + '\n').encode('utf-8'))
                log.debug(f"已發送 Raw 指令給 Teensy: '{command}'")
                return True
            except serial.SerialException as e:
                log.error(f"發送 Raw 指令 '{command}' 到 Teensy 失敗: {e}")
                return False
        else:
            log.warning(f"因序列埠未連接，無法發送 Raw 指令: '{command}'")
            return False
        
    # 【v4.10.4 重構】將 send_command_and_wait_for_ok 拆分為專用的 _wait_for_ok
    def _wait_for_ok(self, context_command: str, timeout: float) -> bool:
        """
        【v4.10.4 重構】阻塞式地等待 Teensy 回應 '[OK]'。
        
        此私有方法專門用於處理需要 [OK] 確認的指令的回應部分。
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    log.debug(f"等待 [OK] 中，收到: '{line}'")
                    if '[ok]' in line.lower():
                        log.info(f"指令 '{context_command}' 已收到 '[OK]' 確認。")
                        return True
                except Exception:
                    pass 
            time.sleep(0.01)
        
        log.warning(f"等待指令 '{context_command}' 的 '[OK]' 回應超時 ({timeout}s)。")
        return False
    

    def send_motor_commands(self, angles: np.ndarray) -> bool:
        """
        【v4.10.4 修改】指令：發送所有馬達的目標角度。
        【v4.13.0 修改】在發送前記錄完整的指令內容。
        
        此方法現在會通過安全守衛，只有在連結狀態為 VERIFIED 時才實際發送。
        """
        # 【v4.10.4 新增】安全守衛
        if self.state.hardware_link_status != HardwareLinkStatus.VERIFIED:
            # 靜默狀態下不發送，但返回 True，避免上層邏輯因靜默而報錯
            return True
        
        action_str = ' '.join(f"{a:.4f}" for a in angles)
        command_to_send = f"move all {action_str}"

        # 【v4.13.0 新增】記錄發送的完整指令，便於除錯
        log.info(f"[PC -> Teensy]: {command_to_send}")
        
        # 使用 execute_command 來發送，它會自動處理 'move all' 的 NONE 回應協定
        return self.execute_command(command_to_send)
