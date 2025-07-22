import os
import sys

# --- 組態設定 (可依需求修改) ---

# 1. 要忽略的資料夾名稱 (使用集合 set 查詢速度較快)
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.vscode', 
    'dist', 'build', 'env', '.idea', 'target', '.DS_Store'
}

# 2. 要忽略的檔案類型 (副檔名)
EXCLUDE_EXTENSIONS = {
    '.pyc', '.pyo', '.o', '.so', '.dll', '.exe',
    '.img', '.iso', '.zip', '.tar', '.gz', '.rar',
    '.pdf', '.docx', '.xlsx', '.pptx', '.ort',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
    '.lock', '.log', '.swp', '.swo', '.stl', '.onnx'
}

# 3. (可選) 如果只想包含特定類型的檔案，可以設定這個清單
INCLUDE_EXTENSIONS = set() 

# --- 樹狀結構產生函式 ---

def generate_tree_structure(root_dir):
    """產生專案的目錄樹結構字串。"""
    tree_lines = []
    
    # 內部的遞迴函式
    def _generate_tree_recursive(current_dir, prefix=""):
        # 獲取目錄下的所有檔案和資料夾，並過濾掉要忽略的
        items = []
        try:
            items = os.listdir(current_dir)
        except OSError as e:
            print(f"警告：無法存取目錄 {current_dir} ({e})")
            return

        # 分離檔案和資料夾以便分別處理和排序
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d)) and d not in EXCLUDE_DIRS])
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_dir, f))])

        # 合併清單，資料夾優先
        all_items = dirs + files
        
        for i, item_name in enumerate(all_items):
            path = os.path.join(current_dir, item_name)
            is_last = (i == len(all_items) - 1)
            
            # 決定連接線的樣式
            connector = "└── " if is_last else "├── "
            
            if os.path.isdir(path):
                tree_lines.append(f"{prefix}{connector}{item_name}/")
                # 準備下一層的縮排
                new_prefix = prefix + ("    " if is_last else "│   ")
                _generate_tree_recursive(path, new_prefix)
            else: # 是檔案
                _, extension = os.path.splitext(item_name)
                # 過濾檔案
                if extension.lower() in EXCLUDE_EXTENSIONS:
                    continue
                if INCLUDE_EXTENSIONS and extension.lower() not in INCLUDE_EXTENSIONS:
                    continue
                tree_lines.append(f"{prefix}{connector}{item_name}")

    # 從根目錄開始遞迴
    tree_lines.append(f"{os.path.basename(os.path.abspath(root_dir))}/")
    _generate_tree_recursive(root_dir)
    return "\n".join(tree_lines)

# --- 程式碼彙整主函式 ---

def generate_code_dump(root_dir, output_filename):
    """
    掃描目錄，產生樹狀結構，並將所有非忽略檔案的內容寫入單一輸出檔。
    """
    if not os.path.isdir(root_dir):
        print(f"錯誤：目錄 '{root_dir}' 不存在。")
        return

    processed_files_count = 0
    
    try:
        with open(output_filename, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 1. 寫入檔案總標題
            outfile.write(f"# 專案程式碼彙整: {os.path.abspath(root_dir)}\n")
            outfile.write("=" * 80 + "\n\n")

            # 2. 產生並寫入專案結構樹
            outfile.write("#" + "-" * 78 + "#\n")
            outfile.write("#" + " " * 30 + "專案目錄結構" + " " * 30 + "#\n")
            outfile.write("#" + "-" * 78 + "#\n\n")
            tree_structure = generate_tree_structure(root_dir)
            outfile.write(tree_structure)
            outfile.write("\n\n\n")

            # 3. 寫入檔案內容分隔標題
            outfile.write("#" + "-" * 78 + "#\n")
            outfile.write("#" + " " * 31 + "各檔案內容" + " " * 32 + "#\n")
            outfile.write("#" + "-" * 78 + "#\n\n")
            
            # 4. 遞迴走訪目錄，寫入各檔案內容
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

                for filename in sorted(filenames):
                    _, extension = os.path.splitext(filename)
                    extension = extension.lower()

                    if extension in EXCLUDE_EXTENSIONS:
                        continue
                    if INCLUDE_EXTENSIONS and extension not in INCLUDE_EXTENSIONS:
                        continue

                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, root_dir)
                    relative_path_str = relative_path.replace(os.sep, '/')

                    try:
                        print(f"正在處理: {relative_path_str}")

                        start_header = f"--- START OF FILE: {relative_path_str} ---"
                        end_header   = f"---  END OF FILE: {relative_path_str}  ---"
                        separator    = "=" * 80
                        
                        outfile.write(f"{separator}\n")
                        outfile.write(f"{start_header}\n")
                        outfile.write(f"{'-' * len(start_header)}\n\n")
                        
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            outfile.write(infile.read())
                        
                        outfile.write(f"\n\n{'-' * len(end_header)}\n")
                        outfile.write(f"{end_header}\n")
                        outfile.write(f"{separator}\n\n")
                        
                        processed_files_count += 1
                    except Exception as e:
                        print(f"警告：無法讀取 {relative_path_str} ({e})")

        # 顯示成功訊息
        print("\n" + "=" * 80)
        print(f"✅ 成功！共處理了 {processed_files_count} 個檔案。")
        print(f"輸出結果已儲存至: {os.path.abspath(output_filename)}")
        print("=" * 80)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{output_filename}'。 ({e})")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_file = 'project_dump.txt'
    generate_code_dump(target_dir, output_file)