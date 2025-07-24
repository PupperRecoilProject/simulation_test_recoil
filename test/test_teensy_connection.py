import importlib
import os
import sys
import pytest

# 若系統未安裝 pyserial，直接略過此測試
serial_spec = importlib.util.find_spec("serial")
if serial_spec is None:
    pytest.skip("pyserial not installed", allow_module_level=True)

import serial
import serial.tools.list_ports

# 直接執行此檔案時，需手動將專案根目錄加入 ``sys.path``
# 以便匯入 ``serial_utils`` 模組
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from serial_utils import TEENSY_VID, TEENSY_PID  # 用於辨識 Teensy 的 VID/PID


def test_teensy_connection():
    """確認是否能順利開啟 Teensy 的序列埠。"""
    print("\n[資訊] 正在尋找 Teensy 裝置...")
    ports = [p.device for p in serial.tools.list_ports.comports()
             if p.vid == TEENSY_VID and p.pid == TEENSY_PID]
    if not ports:
        pytest.skip("未偵測到 Teensy 裝置")

    port = ports[0]
    try:
        print(f"[資訊] 嘗試連接 {port}...")
        ser = serial.Serial(port, 115200, timeout=1)
    except serial.SerialException as exc:
        pytest.fail(f"無法開啟 {port}: {exc}")
    else:
        print(f"[成功] 已開啟 {port}")
        ser.close()


if __name__ == "__main__":
    # 允許此檔案被單獨執行以進行測試
    print("執行 Teensy 連線測試...")
    raise SystemExit(pytest.main([__file__, "-s"]))
