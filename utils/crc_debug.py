"""簡易 CRC-8 驗證工具 (CRC-8/ATM, poly=0x07)\n
This script outputs the CRC-8 for a fixed float array so that\nTeensy firmware can cross-check its implementation.\n"""

import struct
import numpy as np

from controllers.hardware_controller import _crc8


def main() -> None:
    # 固定測試資料: 1.0 ~ 40.0 (float32)
    test_data = np.arange(1.0, 41.0, dtype=np.float32)
    # 以 little-endian 打包成 bytes (與硬體端一致)
    float_bytes = struct.pack('<' + 'f'*len(test_data), *test_data)
    crc = _crc8(float_bytes)
    print("PC calculated CRC:", crc)


if __name__ == "__main__":
    main()
