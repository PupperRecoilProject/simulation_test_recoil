# 提供序列埠溝通能力
import serial
# 用於列舉系統中的序列埠
import serial.tools.list_ports

# Teensy 官方使用的 VID/PID
TEENSY_VID = 0x16C0
TEENSY_PID = 0x0483


def select_serial_port() -> str | None:
    """掃描序列埠並讓使用者選擇欲連接的裝置。

    當偵測到唯一的 Teensy 時會自動選擇該裝置，
    找不到任何埠則回傳 ``None``。
    """
    print("正在掃描可用的序列埠...")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("--- 錯誤: 未找到任何序列埠。請檢查您的設備連接。 ---")
        return None

    # 嘗試根據 VID/PID 自動尋找 Teensy 裝置
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
            # 讓使用者輸入列表中的序號
            choice = int(input(f"請輸入選擇的編號 (0-{len(ports)-1}): "))
            if 0 <= choice < len(ports):
                return ports[choice].device
            print("輸入無效，請重新輸入。")
        except (ValueError, IndexError):
            # 捕獲非數字或範圍外的輸入
            print("輸入無效，請輸入列表中的數字。")

