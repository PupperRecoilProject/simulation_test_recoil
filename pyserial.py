import serial
import time
import sys
import threading
from serial_utils import select_serial_port

exit_signal = threading.Event()

def read_from_port(ser):
    print("\n[讀取線程已啟動] 等待來自 Teensy 的消息...")
    while not exit_signal.is_set():
        try:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    sys.stdout.write(f"\r[Teensy]: {response}\n")
                    sys.stdout.flush()
        except serial.SerialException:
            print("\n[讀取線程錯誤] 序列埠已斷開。")
            break
        except Exception as e:
            print(f"\n[讀取線程未知錯誤]: {e}")
            break
        time.sleep(0.01)


def main():
    SERIAL_PORT = select_serial_port()
    if not SERIAL_PORT:
        sys.exit(1)
    BAUD_RATE = 115200
    ser = None
    try:
        print(f"\n正在嘗試連接到您選擇的埠 {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print("等待 Teensy 初始化 (0.5秒)...")
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("緩衝區已清空，連接準備就緒。")
        read_thread = threading.Thread(target=read_from_port, args=(ser,))
        read_thread.daemon = True
        read_thread.start()
        print("\n--- Teensy 控制台已啟動 ---")
        print("您可以輸入指令 (例如: 'stop', 'printon'), 然後按 Enter。")
        print("輸入 'exit' 來退出程式。")
        while True:
            command = input()
            if command.lower() == 'exit':
                break
            command_to_send = command + '\n'
            ser.write(command_to_send.encode('utf-8'))
    except serial.SerialException as e:
        print(f"--- 致命錯誤 ---")
        print(f"無法打開或操作序列埠 {SERIAL_PORT}。")
        print(f"錯誤詳情: {e}")
    except KeyboardInterrupt:
        print("\n偵測到 Ctrl+C，正在終止程式...")
    except Exception as e:
        print(f"發生未知錯誤: {e}")
    finally:
        exit_signal.set()
        if ser and ser.is_open:
            ser.close()
            print(f"序列埠 {SERIAL_PORT} 已安全關閉。")
        if 'read_thread' in locals() and read_thread.is_alive():
            read_thread.join(timeout=1)
        print("程式已退出。")

if __name__ == "__main__":
    main()
