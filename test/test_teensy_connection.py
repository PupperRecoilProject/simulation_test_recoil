import importlib
import pytest

serial_spec = importlib.util.find_spec("serial")
if serial_spec is None:
    pytest.skip("pyserial not installed", allow_module_level=True)

import serial
import serial.tools.list_ports
from serial_utils import TEENSY_VID, TEENSY_PID


def test_teensy_connection():
    """Check that a Teensy board can be opened via serial."""
    ports = [p.device for p in serial.tools.list_ports.comports()
             if p.vid == TEENSY_VID and p.pid == TEENSY_PID]
    if not ports:
        pytest.skip("No Teensy devices detected")

    port = ports[0]
    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except serial.SerialException as exc:
        pytest.fail(f"Failed to open {port}: {exc}")
    else:
        ser.close()
