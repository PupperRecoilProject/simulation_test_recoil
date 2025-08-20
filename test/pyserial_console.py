# test/pyserial_console.py
"""
【v4.4.1 修改】【註解規範教學範例】Teensy 數據流實時驗證工具

本檔案旨在作為專案「註解與文檔字符串規範 v4.2.5」的標準教學範例。
它通過模擬從 v4.4.0 到 v4.4.1 的版本迭代，展示了所有類型的註解規範。

主要用途:
- 作為一個功能齊全的、獨立的序列埠控制台。
- 根據預定義的「數據契約」，實時解析、驗證和格式化顯示來自 Teensy 的數據流。
- 監控數據幀的接收頻率，以評估通信穩定性。
- 允許使用者在主控台輸入指令並發送給 Teensy，與數據監聽並行。

版本演進歷史 (範例):
- v4.4.0: 新增了核心的數據流解析、驗證和格式化顯示功能。
- v4.4.1: 重構了顯示邏輯，修復了資源釋放的 Bug，並恢復了用戶指令輸入功能。
"""
import serial
import time
import sys
import threading
import os
import numpy as np
from collections import deque

# 【v4.4.0 修改】強化路徑設置，確保無論從何處執行都能找到 src
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.serial_utils import select_serial_port

# 【v4.4.0 新增】Teensy-PC 數據契約 v1.1
# 將數據契約程式碼化，作為解析和驗證的唯一依據。
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

# 全域旗標，用於在主執行緒與讀取、寫入執行緒間傳遞退出訊號
exit_signal = threading.Event()

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
            
        return {'success': True, 'data': parsed_data}
        
    except ValueError as e:
        return {
            'success': False,
            'error': f"數據轉換為浮點數失敗: {e}",
            'raw_line': line
        }

def display_validated_data(ser_port: str, result: dict, freq: float):
    """【v4.4.1 新增】【重構】將數據顯示邏輯從讀取線程中分離出來。"""
    # 清除終端螢幕以便刷新顯示
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("--- Teensy 數據流實時驗證工具 (v4.4.1) ---")
    print(f"連接埠: {ser_port} | 在下方輸入指令 (或 'exit' 退出)")
    
    if freq > 0:
        print(f"數據幀頻率: {freq:.2f} Hz\n")
    else:
        print("數據幀頻率: (計算中...)\n")

    if result['success']:
        print("✅ 數據幀驗證通過\n")
        # 格式化打印
        for name, data_array in result['data'].items():
            spec = DATA_CONTRACT['fields'][name]
            unit = spec['unit']
            data_str = np.array2string(data_array, precision=4, suppress_small=True, formatter={'float_kind':lambda x: f"{x:8.4f}"})
            print(f"{name.replace('_', ' ').title():<18} ({unit}): {data_str}")
    else:
        print(f"❌ 數據幀驗證失敗!")
        print(f"   錯誤: {result['error']}")
        print(f"   原始數據: \"{result['raw_line'].strip()}\"")
    
    # 【v4.4.1 新增】為用戶輸入提供一個清晰的提示符
    print("\n------------------------------------")
    print("請輸入指令 > ", end='', flush=True)


def read_from_port(ser):
    """【v4.4.1 修改】【重構】在背景執行緒中持續讀取、解析、驗證數據，並調用顯示函式。"""
    print("\n[讀取線程已啟動] 等待來自 Teensy 的數據流...")
    
    frame_times = deque(maxlen=100)
    
    while not exit_signal.is_set():
        try:
            if ser and ser.is_open and ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore')
                if not response:
                    continue

                frame_times.append(time.perf_counter())
                
                result = parse_and_validate_line(response)
                
                freq = 0.0
                if len(frame_times) > 1:
                    freq = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

                # 【v4.4.1 重構】調用獨立的顯示函式
                display_validated_data(ser.port, result, freq)

        except serial.SerialException:
            print("\n[讀取線程錯誤] 序列埠已斷開。")
            exit_signal.set() # 【v4.4.1 修復】確保在出錯時也能觸發退出信號
            break
        except Exception as e:
            print(f"\n[讀取線程未知錯誤]: {e}")
            exit_signal.set() # 【v4.4.1 修復】確保在出錯時也能觸發退出信號
            break
        
        time.sleep(0.001)

def main():
    """【v4.4.1 修改】主流程，現在管理讀取線程和用戶指令輸入。"""
    SERIAL_PORT = select_serial_port()
    if not SERIAL_PORT:
        sys.exit(1)
    BAUD_RATE = 115200
    ser = None
    read_thread = None
    
    try:
        print(f"\n正在連接到 {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("連接成功，正在啟動讀取線程...")
        
        read_thread = threading.Thread(target=read_from_port, args=(ser,))
        read_thread.daemon = True
        read_thread.start()
        
        # 【v4.4.1 修改】在主線程中恢復用戶指令輸入功能
        while not exit_signal.is_set():
            command = input()
            if command.lower() == 'exit':
                print("收到 'exit' 指令，正在關閉...")
                break # 退出主迴圈
            
            if ser and ser.is_open:
                command_to_send = command + '\n'
                ser.write(command_to_send.encode('utf-8'))
            else:
                print("序列埠未連接，無法發送指令。")
                break

    except serial.SerialException as e:
        print(f"--- 致命錯誤: 無法打開序列埠 {SERIAL_PORT}。錯誤詳情: {e}")
    except (KeyboardInterrupt, EOFError):
        # 【v4.4.1 修改】捕獲 EOFError，當 input() 沒有更多輸入時（例如在管道中）
        print("\n偵測到程序終止信號，正在關閉...")
    except Exception as e:
        print(f"發生未知錯誤: {e}")
    finally:
        # 【v4.4.1 修改】確保退出信號被設置，以便所有線程都能 cleanly 退出
        print("正在執行清理工作...")
        exit_signal.set() 
        
        if read_thread and read_thread.is_alive():
            read_thread.join(timeout=1)
        
        if ser and ser.is_open:
            ser.close()
            print(f"序列埠 {SERIAL_PORT} 已安全關閉。")
            
        print("程式已退出。")

if __name__ == "__main__":
    main()