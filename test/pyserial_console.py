# test/pyserial_console.py
"""
【v4.4.2 修改】【註解規範教學範例】Teensy 數據流實時驗證工具（穩定版）

本檔案旨在作為專案「註解與文檔字符串規範 v4.2.5」的標準教學範例。
它通過從 v4.4.0 → v4.4.1 → v4.4.2 的迭代，展示「功能完整、可並行輸入、可穩定刷新顯示」的實作要點。

主要用途：
- 作為一個功能齊全且穩定的序列埠控制台。
- 根據預定義的「數據契約（data contract）」即時解析、驗證與格式化顯示 Teensy 的數據流。
- 監控數據幀接收頻率（Hz）與延遲抖動（jitter）以評估通信穩定性。
- 允許使用者輸入命令發送到 Teensy，與數據監聽並行，且避免印出互斥（STDIN/STDOUT race）。
- 可選擇把原始資料（raw）與驗證後資料（parsed）寫入檔案以便離線分析。

版本演進歷史（範例）：
- v4.4.0：新增核心數據流解析、驗證與格式化顯示。
- v4.4.1：重構顯示邏輯，修復資源釋放，恢復用戶指令輸入。
- v4.4.2：加入輸入輸出互斥（thread-safe I/O）、降頻刷新（rate limiting）、
            頻率估計去抖（EMA + 滑動窗）、無資料 watchdog、健壯關閉流程，以及 CLI 參數。
"""
import argparse
import os
import sys
import time
import threading
from collections import deque
from queue import Queue, Empty

import numpy as np
import serial

# 【v4.4.0 修改】強化路徑設置，確保無論從何處執行都能找到 src
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.serial_utils import select_serial_port  # 專案既有功能：互動式挑選序列埠

# =========================
# Teensy-PC 數據契約（Data Contract）
# =========================
# 【v4.4.0 新增】【維持不動】數據契約 v1.1
DATA_CONTRACT = {
    'total_fields': 34,
    'fields': {
        'angular_velocity': {'start': 0, 'len': 3, 'unit': 'rad/s'},
        'gravity_vector':   {'start': 3, 'len': 3, 'unit': 'norm'},
        'accelerometer':    {'start': 6, 'len': 3, 'unit': 'm/s^2'},
        'pitch_angle':      {'start': 9, 'len': 1, 'unit': 'rad'},
        'joint_positions':  {'start': 10, 'len': 12, 'unit': 'rad'},
        'joint_velocities': {'start': 22, 'len': 12, 'unit': 'rad/s'},
    }
}

# =========================
# 全域旗標與同步原語
# =========================
exit_signal = threading.Event()          # 程式終止旗標
print_lock = threading.Lock()            # 【v4.4.2 新增】避免讀線程與輸入提示互相覆寫
display_rate_hz = 10.0                   # 【v4.4.2 新增】限制畫面更新頻率（避免閃爍與高 CPU）
no_data_timeout_s = 5.0                  # 【v4.4.2 新增】無資料 watchdog

# =========================
# 工具函式
# =========================
def parse_and_validate_line(line: str) -> dict:
    """【v4.4.0 新增】根據 DATA_CONTRACT 解析並驗證單行數據。"""
    parts = line.strip().split(',')
    if len(parts) != DATA_CONTRACT['total_fields']:
        return {
            'success': False,
            'error': f"欄位數量錯誤 (預期 {DATA_CONTRACT['total_fields']}, 實際 {len(parts)})",
            'raw_line': line
        }

    try:
        data_vec = np.array(parts, dtype=np.float32)
        parsed_data = {}
        for name, spec in DATA_CONTRACT['fields'].items():
            start_idx = spec['start']
            end_idx = start_idx + spec['len']
            parsed_data[name] = data_vec[start_idx:end_idx]
        return {'success': True, 'data': parsed_data, 'raw': data_vec}
    except ValueError as e:
        return {
            'success': False,
            'error': f"數據轉換為浮點數失敗: {e}",
            'raw_line': line
        }


def _format_data_for_print(result: dict) -> str:
    """【v4.4.2 新增】把驗證結果格式化為字串，供穩定印出。"""
    lines = []
    if result['success']:
        lines.append("✅ 數據幀驗證通過\n")
        for name, data_array in result['data'].items():
            spec = DATA_CONTRACT['fields'][name]
            unit = spec['unit']
            data_str = np.array2string(
                data_array,
                precision=4,
                suppress_small=True,
                formatter={'float_kind': lambda x: f"{x:8.4f}"}
            )
            lines.append(f"{name.replace('_', ' ').title():<18} ({unit}): {data_str}")
    else:
        lines.append("❌ 數據幀驗證失敗!")
        lines.append(f"   錯誤: {result['error']}")
        lines.append(f"   原始數據: \"{result['raw_line'].strip()}\"")
    return "\n".join(lines)


def _clear_screen():
    """【v4.4.2 新增】抽象化清屏，便於之後切換到 curses 或增量刷新。"""
    os.system('cls' if os.name == 'nt' else 'clear')


def _safe_print(block: str):
    """【v4.4.2 新增】標準輸出互斥，避免與輸入提示互相覆蓋。"""
    with print_lock:
        sys.stdout.write(block + ("\n" if not block.endswith("\n") else ""))
        sys.stdout.flush()


# =========================
# 背景線程：資料讀取與顯示
# =========================
def read_loop(ser: serial.Serial, args, cmd_queue: Queue):
    """
    【v4.4.2 修改】持續讀取、解析、驗證數據，並以「限速刷新」穩定顯示。
    - 新增 EMA 頻率估計（exponential moving average）與滑動窗平均。
    - 新增 no-data watchdog，在數秒無資料時提示並自動退出。
    - 把「顯示」與「輸入提示」整合成安全輸出，不打斷使用者輸入體驗。
    """
    _safe_print("\n[讀取線程已啟動] 等待來自 Teensy 的數據流...")

    # 頻率估計
    frame_times = deque(maxlen=200)     # 用於滑動窗估計
    ema_freq = None                     # 指數移動平均（EMA）
    ema_alpha = 0.2

    last_display_ts = 0.0
    last_data_ts = time.perf_counter()

    raw_log = None
    parsed_log = None
    try:
        if args.raw_log:
            raw_log = open(args.raw_log, "a", encoding="utf-8")
        if args.parsed_log:
            parsed_log = open(args.parsed_log, "a", encoding="utf-8")
    except Exception as e:
        _safe_print(f"[警告] 開啟檔案失敗：{e}")

    try:
        while not exit_signal.is_set():
            # 讀資料
            line = None
            try:
                if ser and ser.is_open and ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore')
            except serial.SerialException:
                _safe_print("\n[讀取線程錯誤] 序列埠已斷開。")
                exit_signal.set()
                break

            now = time.perf_counter()

            if line:
                last_data_ts = now
                frame_times.append(now)

                result = parse_and_validate_line(line)

                # 寫檔（非必要）
                try:
                    if raw_log:
                        raw_log.write(line if line.endswith("\n") else line + "\n")
                    if parsed_log and result.get('success'):
                        # 寫入 CSV 形式，方便 pandas 讀取
                        flat = []
                        for name in DATA_CONTRACT['fields'].keys():
                            flat.extend(result['data'][name].tolist())
                        parsed_log.write(",".join(str(x) for x in flat) + "\n")
                except Exception as e:
                    _safe_print(f"[警告] 寫檔失敗：{e}")

                # 頻率估計（Hz）
                freq = 0.0
                if len(frame_times) > 1:
                    freq = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                ema_freq = freq if ema_freq is None else (ema_alpha * freq + (1 - ema_alpha) * ema_freq)

                # 限速刷新（避免每幀清屏閃爍）
                if now - last_display_ts >= 1.0 / max(1.0, display_rate_hz):
                    last_display_ts = now
                    # 組裝畫面
                    header = [
                        "--- Teensy 數據流實時驗證工具 (v4.4.2) ---",
                        f"連接埠: {ser.port} | 鮑率: {ser.baudrate}",
                        f"數據幀頻率: {ema_freq:.2f} Hz" if ema_freq is not None else "數據幀頻率: (計算中...)",
                        ""
                    ]
                    body = _format_data_for_print(result)
                    footer = [
                        "",
                        "------------------------------------",
                        "請輸入指令 > "
                    ]
                    # 清屏並印出
                    with print_lock:
                        _clear_screen()
                        sys.stdout.write("\n".join(header) + "\n")
                        sys.stdout.write(body + "\n")
                        sys.stdout.write("\n".join(footer))
                        sys.stdout.flush()
            else:
                # 無資料 watchdog
                if now - last_data_ts >= no_data_timeout_s:
                    _safe_print(f"\n[警告] {no_data_timeout_s:.0f} 秒未收到資料，可能連線中斷或裝置無輸出。")
                    exit_signal.set()
                    break

            # 處理要送往 Teensy 的命令（由輸入線程丟進來）
            try:
                cmd = cmd_queue.get_nowait()
            except Empty:
                cmd = None
            if cmd is not None and ser and ser.is_open:
                try:
                    ser.write((cmd + "\n").encode("utf-8"))
                except serial.SerialException:
                    _safe_print("[錯誤] 指令發送失敗，序列埠可能已斷開。")
                    exit_signal.set()
                    break

            time.sleep(0.0005)
    finally:
        for f in (raw_log, parsed_log):
            try:
                if f:
                    f.close()
            except Exception:
                pass


# =========================
# 背景線程：非阻塞輸入
# =========================
def input_loop(cmd_queue: Queue):
    """
    【v4.4.2 新增】獨立輸入線程，避免與讀線程互相覆寫輸出。
    - 使用 Queue 傳遞指令給 read_loop。
    - 支援 'exit' 指令關閉程式。
    """
    try:
        while not exit_signal.is_set():
            with print_lock:
                # 顯式顯示提示符（避免被讀線程刷新吃掉）
                sys.stdout.write("請輸入指令 > ")
                sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                # EOF（例如管線），結束
                exit_signal.set()
                break
            cmd = line.strip()
            if cmd.lower() == "exit":
                _safe_print("收到 'exit' 指令，正在關閉...")
                exit_signal.set()
                break
            cmd_queue.put(cmd)
    except (KeyboardInterrupt, EOFError):
        exit_signal.set()


# =========================
# 主流程
# =========================
def build_argparser():
    """【v4.4.2 新增】命令列參數，方便測試與自動化腳本呼叫。"""
    p = argparse.ArgumentParser(description="Teensy Serial Console with Real-time Validation")
    p.add_argument("--port", type=str, default=None, help="指定序列埠名稱（覆寫自動選擇）")
    p.add_argument("--baud", type=int, default=921600, help="鮑率（baud rate），預設 921600")
    p.add_argument("--raw-log", type=str, default=None, help="將原始字串輸出到檔案")
    p.add_argument("--parsed-log", type=str, default=None, help="將解析後資料（CSV）輸出到檔案")
    p.add_argument("--no-data-timeout", type=float, default=no_data_timeout_s, help="無資料自動離開秒數")
    p.add_argument("--display-rate", type=float, default=display_rate_hz, help="畫面更新頻率上限（Hz）")
    return p


def main():
    """【v4.4.2 修改】主流程：
       - 解析 CLI 參數
       - 開啟 Serial
       - 啟動「讀線程」與「輸入線程」
       - 確保 clean shutdown
    """
    parser = build_argparser()
    args = parser.parse_args()

    # 套用可調參數
    global display_rate_hz, no_data_timeout_s
    display_rate_hz = max(1.0, float(args.display_rate))
    no_data_timeout_s = max(1.0, float(args.no_data_timeout))

    # 選擇序列埠
    serial_port = args.port or select_serial_port()
    if not serial_port:
        sys.exit(1)

    ser = None
    read_t = None
    input_t = None
    cmd_queue = Queue()

    try:
        _safe_print(f"\n正在連接到 {serial_port}（baud={args.baud}）...")
        ser = serial.Serial(serial_port, args.baud, timeout=0.05)
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        _safe_print("連接成功，正在啟動背景線程...")

        read_t = threading.Thread(target=read_loop, args=(ser, args, cmd_queue), daemon=True)
        input_t = threading.Thread(target=input_loop, args=(cmd_queue,), daemon=True)
        read_t.start()
        input_t.start()

        # 主線程只負責等待結束
        while not exit_signal.is_set():
            time.sleep(0.05)

    except serial.SerialException as e:
        _safe_print(f"--- 致命錯誤：無法打開序列埠 {serial_port}。錯誤詳情：{e}")
    except (KeyboardInterrupt, EOFError):
        _safe_print("\n偵測到程序終止信號，正在關閉...")
    except Exception as e:
        _safe_print(f"發生未知錯誤：{e}")
    finally:
        _safe_print("正在執行清理工作...")
        exit_signal.set()

        # 等待線程結束
        for t in (input_t, read_t):
            try:
                if t and t.is_alive():
                    t.join(timeout=1.5)
            except Exception:
                pass

        if ser and ser.is_open:
            try:
                ser.close()
                _safe_print(f"序列埠 {serial_port} 已安全關閉。")
            except Exception:
                pass

        _safe_print("程式已退出。")


if __name__ == "__main__":
    main()
