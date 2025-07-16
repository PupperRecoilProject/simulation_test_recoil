import os
import sys

# --- 組態設定 (可依需求修改) ---

# 1. 要忽略的資料夾名稱 (使用集合 set 查詢速度較快)
#    通常是版本控制、虛擬環境、相依性套件等
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.vscode', 
    'dist', 'build', 'env', '.idea', 'target', '.DS_Store'
}

# 2. 要忽略的檔案類型 (副檔名)
#    通常是二進位檔案、日誌、壓縮檔等
EXCLUDE_EXTENSIONS = {
    '.pyc', '.pyo', '.o', '.so', '.dll', '.exe',
    '.img', '.iso', '.zip', '.tar', '.gz', '.rar',
    '.pdf', '.docx', '.xlsx', '.pptx',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
    '.lock', '.log', '.swp', '.swo'
}

# 3. (可選) 如果你只想包含特定類型的檔案，可以設定這個清單
#    如果此清單不是空的，腳本將只處理這些副檔名的檔案。
#    例如: INCLUDE_EXTENSIONS = {'.py', '.js', '.html', '.css'}
INCLUDE_EXTENSIONS = set() 

# --- 腳本主體 ---

def generate_code_dump(root_dir, output_filename):
    """
    遞迴掃描一個目錄，並將所有非忽略檔案的內容寫入單一輸出檔。
    """
    # 檢查根目錄是否存在
    if not os.path.isdir(root_dir):
        print(f"錯誤：目錄 '{root_dir}' 不存在。")
        return

    processed_files_count = 0
    
    try:
        # 使用 'w' (寫入模式) 和 utf-8 編碼開啟檔案
        # errors='ignore' 會在遇到無法解碼的字元時忽略它，避免因二進位檔案出錯
        with open(output_filename, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 在輸出檔案的開頭加上標題
            outfile.write(f"# 專案程式碼彙整: {os.path.abspath(root_dir)}\n")
            outfile.write("=" * 80 + "\n\n")

            # os.walk 會遞迴地走過目錄樹
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                
                # --- 過濾資料夾 ---
                # 這是 os.walk 的一個技巧：直接修改 dirnames 列表，
                # os.walk 就不會再進入這些被移除的資料夾。
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

                # 排序檔案，讓輸出結果更一致
                for filename in sorted(filenames):
                    # 取得檔案副檔名
                    _, extension = os.path.splitext(filename)
                    extension = extension.lower()

                    # --- 過濾檔案 ---
                    if extension in EXCLUDE_EXTENSIONS:
                        continue
                    
                    if INCLUDE_EXTENSIONS and extension not in INCLUDE_EXTENSIONS:
                        continue

                    # 組合完整檔案路徑
                    file_path = os.path.join(dirpath, filename)
                    
                    # 取得相對於根目錄的路徑，讓輸出更簡潔
                    relative_path = os.path.relpath(file_path, root_dir)

                    try:
                        # 寫入檔案分隔線和路徑標題
                        header = f"--- FILE: {relative_path.replace(os.sep, '/')} ---"
                        outfile.write("=" * 80 + "\n")
                        outfile.write(header + "\n")
                        outfile.write("-" * len(header) + "\n\n")
                        
                        print(f"正在處理: {relative_path}")

                        # 讀取原始檔案內容並寫入輸出檔
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write("\n\n")
                        
                        processed_files_count += 1

                    except Exception as e:
                        # 處理讀取檔案時可能發生的其他錯誤
                        error_msg = f"--- 無法讀取檔案: {relative_path} (錯誤: {e}) ---\n\n"
                        outfile.write(error_msg)
                        print(f"警告: {error_msg.strip()}")

        # 顯示成功訊息
        print("\n" + "=" * 80)
        print(f"✅ 成功！共處理了 {processed_files_count} 個檔案。")
        print(f"輸出結果已儲存至: {os.path.abspath(output_filename)}")
        print("=" * 80)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{output_filename}'。請檢查權限。 ({e})")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    # 決定要掃描的根目錄
    # 如果執行時有給參數，就用第一個參數當作路徑，否則使用當前目錄 "."
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # 決定輸出檔案的名稱
    output_file = 'project_dump.txt'
    
    # 執行主函式
    generate_code_dump(target_dir, output_file)