import serial
import serial.tools.list_ports

TEENSY_VID = 0x16C0
TEENSY_PID = 0x0483


def select_serial_port() -> str | None:
    """Scan available serial ports and prompt the user to select one.

    If a single Teensy device is detected by VID/PID it is selected
    automatically. Returns the chosen device name or ``None`` if no
    ports are found.
    """
    print("正在掃描可用的序列埠...")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("--- 錯誤: 未找到任何序列埠。請檢查您的設備連接。 ---")
        return None

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
            print("輸入無效，請重新輸入。")
        except (ValueError, IndexError):
            print("輸入無效，請輸入列表中的數字。")

