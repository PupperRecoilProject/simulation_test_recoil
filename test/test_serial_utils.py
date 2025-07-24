import importlib.util
import os
import sys
import builtins
import types
import pytest

serial_spec = importlib.util.find_spec("serial")
if serial_spec is None:
    pytest.skip("pyserial not installed", allow_module_level=True)

# 讓測試可以匯入專案根目錄下的模組
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from serial_utils import select_serial_port, TEENSY_VID, TEENSY_PID

class DummyPort:
    def __init__(self, device, description, vid, pid):
        self.device = device
        self.description = description
        self.vid = vid
        self.pid = pid


def test_no_ports(monkeypatch, capsys):
    """當沒有找到任何序列埠時應回傳 None 並提示錯誤。"""
    monkeypatch.setattr('serial.tools.list_ports.comports', lambda: [])
    port = select_serial_port()
    captured = capsys.readouterr().out
    assert port is None, '未回傳 None'
    assert '未找到任何序列埠' in captured


def test_auto_select_teensy(monkeypatch):
    """只有一個 Teensy 裝置時應自動選擇該埠。"""
    dummy = DummyPort('COM3', 'Teensy', TEENSY_VID, TEENSY_PID)
    monkeypatch.setattr('serial.tools.list_ports.comports', lambda: [dummy])
    port = select_serial_port()
    assert port == 'COM3', '未正確自動選擇 Teensy 序列埠'


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

