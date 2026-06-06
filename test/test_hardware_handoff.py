"""硬體進出 / 序列埠交接的特徵與回歸測試（T3 / T3b）。

涵蓋三件事：
1. `_execute_stop` 在控制執行緒上執行不會 self-join（鎖死當前修正後行為；防未來重新引入 self-join）。
2. `_execute_start` 在持有控制權期間遇「非預期例外」仍會歸還控制權（resume_control 必被呼叫）。
3. `SerialCommunicator.close()` 即使處於暫停狀態也會關閉序列埠（修序列埠洩漏 → 下次連不上）。

僅用輕量假物件，不需實體硬體。
"""
import importlib.util
import os
import sys
import threading
import pytest

if importlib.util.find_spec("serial") is None:
    pytest.skip("pyserial not installed", allow_module_level=True)

repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import serial
from unittest import mock

from src.controllers.hardware_controller import HardwareController, HWState
from src.core.state import HardwareLinkStatus
import src.controllers.hardware_controller as hc_mod
from src.hardware.serial_communicator import SerialCommunicator


# --- 輕量假物件 -------------------------------------------------------------
class FakeState:
    """只實作 HardwareController 會碰到的屬性與一把真鎖。"""
    def __init__(self):
        self.lock = threading.RLock()
        self.hardware_ai_is_active = False
        self.hardware_is_running = False
        self.hardware_link_status = HardwareLinkStatus.UNVERIFIED


class FakeConfig:
    control_freq = 5.0


def _make_hwc(serial_comm):
    return HardwareController(
        config=FakeConfig(),
        policy=mock.MagicMock(name="policy"),
        state=FakeState(),
        serial_comm=serial_comm,
    )


# --- 測試 1：_execute_stop 不 self-join ------------------------------------
def test_execute_stop_does_not_self_join():
    """在「被指定為 control_thread 的執行緒」上跑 _execute_stop，應正常完成、不丟 RuntimeError。

    若有人重新引入 `self.control_thread.join()`（join 自己），這裡會以 RuntimeError 失敗。
    """
    serial_comm = mock.MagicMock(name="serial_comm")
    serial_comm.is_connected = False
    hwc = _make_hwc(serial_comm)
    hwc.teensy_api = None
    hwc.ser = None
    hwc._set_internal_state(HWState.RUNNING)

    captured = {}

    def worker():
        hwc.control_thread = threading.current_thread()
        try:
            hwc._execute_stop()
        except BaseException as e:  # noqa: BLE001 - 測試需捕捉所有例外
            captured["err"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "_execute_stop 卡死（可能 self-join 阻塞）"
    assert "err" not in captured, f"_execute_stop 丟出例外（疑似 self-join）：{captured.get('err')!r}"
    assert hwc.internal_state == HWState.STOPPED
    serial_comm.resume_control.assert_called()  # 交接時必喚醒讀取迴圈


# --- 測試 2：_execute_start 遇非預期例外仍歸還控制權 ------------------------
def test_execute_start_resumes_control_on_unexpected_exception():
    """持有控制權期間若拋非 SerialException 例外，finally 必呼叫 resume_control，避免序列埠洩漏。"""
    serial_comm = mock.MagicMock(name="serial_comm")
    serial_comm.is_connected = True
    serial_comm.relinquish_control.return_value = True
    serial_comm.get_serial_connection.return_value = mock.MagicMock(name="ser")

    class FakeTeensyAPI:
        def __init__(self, sc, st):
            self.ser = mock.MagicMock(name="teensy_ser")
        def execute_command(self, *a, **k):
            raise ValueError("unexpected boom")  # 非 SerialException

    hwc = _make_hwc(serial_comm)

    with mock.patch.object(hc_mod, "TeensyAPI", FakeTeensyAPI):
        hwc._execute_start()

    serial_comm.resume_control.assert_called_once()
    assert hwc.internal_state == HWState.FAILED
    assert hwc.teensy_api is None


def test_execute_start_keeps_control_on_success():
    """成功啟動時不可歸還控制權（本控制器要保留序列埠獨佔）。"""
    serial_comm = mock.MagicMock(name="serial_comm")
    serial_comm.is_connected = True
    serial_comm.relinquish_control.return_value = True
    serial_comm.get_serial_connection.return_value = mock.MagicMock(name="ser")

    class FakeTeensyAPI:
        def __init__(self, sc, st):
            self.ser = mock.MagicMock(name="teensy_ser")
        def execute_command(self, *a, **k):
            return True  # 全部成功

    hwc = _make_hwc(serial_comm)

    with mock.patch.object(hc_mod, "TeensyAPI", FakeTeensyAPI):
        hwc._execute_start()

    assert hwc.internal_state == HWState.RUNNING
    serial_comm.resume_control.assert_not_called()


# --- 測試 3：close() 即使暫停也關閉序列埠 ----------------------------------
def test_close_releases_port_even_when_paused():
    """暫停狀態下 close() 仍應關閉序列埠並清旗標，避免洩漏導致下次連不上。"""
    sc = SerialCommunicator()
    sc._pause_requested.set()
    sc.read_thread = None  # 跳過 join
    fake_ser = mock.MagicMock(name="ser")
    fake_ser.is_open = True
    sc.ser = fake_ser
    sc.is_connected = True
    sc.port_name = "FAKE"

    sc.close()

    fake_ser.close.assert_called_once()
    assert sc.is_connected is False
    assert not sc._pause_requested.is_set()
