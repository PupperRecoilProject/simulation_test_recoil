# test/dump_project.py (最終防遞迴版)
import os
import sys
from datetime import datetime

# --- 組態設定 ---

# 1. 要忽略的資料夾名稱
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.vscode',
    'dist', 'build', 'env', '.idea', 'target', '.DS_Store'
}

# 2. 定義一個「內容跳過清單」。
SKIP_CONTENT_EXTENSIONS = {
    '.onnx', '.stl', '.ort', '.png', '.jpg', '.jpeg',
    '.exe', '.dll', '.so', '.o', '.zip', '.rar', '.gz',
    '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.avi',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pyc', '.pyo', '.lock', '.swp', '.swo',
}

# 3. (可選) 如果只想包含特定類型的檔案，可以設定這個清單
INCLUDE_EXTENSIONS = set()


# --- 樹狀結構產生函式 (無須修改) ---
def generate_tree_structure(root_dir, project_name):
    tree_lines = []
    
    def _generate_tree_recursive(current_dir, prefix=""):
        items = []
        try:
            items = os.listdir(current_dir)
        except OSError as e:
            print(f"警告：無法存取目錄 {current_dir} ({e})")
            return

        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d)) and d not in EXCLUDE_DIRS])
        
        # 【修改】過濾掉我們自己產生的輸出檔案
        files_to_process = []
        for f in sorted([f for f in items if os.path.isfile(os.path.join(current_dir, f))]):
            is_old_dump = f.startswith(f"{project_name}_dump_") and f.endswith(".txt")
            is_legacy_dump = f == "project_dump.txt"
            if not is_old_dump and not is_legacy_dump:
                files_to_process.append(f)
        
        all_items = dirs + files_to_process
        
        for i, item_name in enumerate(all_items):
            path = os.path.join(current_dir, item_name)
            is_last = (i == len(all_items) - 1)
            connector = "└── " if is_last else "├── "
            
            if os.path.isdir(path):
                tree_lines.append(f"{prefix}{connector}{item_name}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                _generate_tree_recursive(path, new_prefix)
            else:
                if INCLUDE_EXTENSIONS and os.path.splitext(item_name)[1].lower() not in INCLUDE_EXTENSIONS:
                    continue
                tree_lines.append(f"{prefix}{connector}{item_name}")

    tree_lines.append(f"{project_name}/")
    _generate_tree_recursive(root_dir)
    return "\n".join(tree_lines)


# --- 程式碼彙整主函式 ---
def generate_code_dump(root_dir, output_filename, project_name):
    if not os.path.isdir(root_dir):
        print(f"錯誤：目錄 '{root_dir}' 不存在。")
        return

    processed_files_count = 0
    
    try:
        with open(output_filename, 'w', encoding='utf-8', errors='ignore') as outfile:
            outfile.write(f"# 專案程式碼彙整: {os.path.abspath(root_dir)}\n")
            outfile.write("=" * 80 + "\n\n")

            outfile.write("#" + "-" * 78 + "#\n")
            outfile.write("#" + " " * 30 + "專案目錄結構" + " " * 30 + "#\n")
            outfile.write("#" + "-" * 78 + "#\n\n")
            # 【修改】將 project_name 傳入
            tree_structure = generate_tree_structure(root_dir, project_name)
            outfile.write(tree_structure)
            outfile.write("\n\n\n")

            outfile.write("#" + "-" * 78 + "#\n")
            outfile.write("#" + " " * 31 + "各檔案內容" + " " * 32 + "#\n")
            outfile.write("#" + "-" * 78 + "#\n\n")
            
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

                for filename in sorted(filenames):
                    # =============================================================
                    # ===        【核心修正：跳過舊的輸出檔】                     ===
                    # =============================================================
                    is_dynamic_dump = filename.startswith(f"{project_name}_dump_") and filename.endswith(".txt")
                    is_legacy_dump = filename == "project_dump.txt"

                    if is_dynamic_dump or is_legacy_dump:
                        print(f"正在跳過 (舊的輸出檔): {filename}")
                        continue
                    # =============================================================

                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, root_dir).replace(os.sep, '/')

                    try:
                        print(f"正在處理: {relative_path}")

                        start_header = f"--- START OF FILE: {relative_path} ---"
                        end_header   = f"---  END OF FILE: {relative_path}  ---"
                        separator    = "=" * 80
                        
                        outfile.write(f"{separator}\n")
                        outfile.write(f"{start_header}\n")
                        outfile.write(f"{'-' * len(start_header)}\n\n")
                        
                        _, extension = os.path.splitext(file_path)
                        
                        if extension.lower() in SKIP_CONTENT_EXTENSIONS:
                            file_size = os.path.getsize(file_path)
                            outfile.write(f"[Content skipped for file type '{extension}': {filename} ({file_size / 1024:.2f} KB)]")
                        else:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='strict') as infile:
                                    outfile.write(infile.read())
                            except (UnicodeDecodeError, IOError):
                                file_size = os.path.getsize(file_path)
                                outfile.write(f"[Content skipped due to read error: {filename} ({file_size / 1024:.2f} KB)]")
                        
                        outfile.write(f"\n\n{'-' * len(end_header)}\n")
                        outfile.write(f"{end_header}\n")
                        outfile.write(f"{separator}\n\n")
                        
                        processed_files_count += 1
                    except Exception as e:
                        print(f"警告：無法讀取 {relative_path} ({e})")

        print("\n" + "=" * 80)
        print(f"✅ 成功！共處理了 {processed_files_count} 個檔案。")
        print(f"輸出結果已儲存至: {os.path.abspath(output_filename)}")
        print("=" * 80)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{output_filename}'。 ({e})")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)
    
    os.chdir(project_root)
    
    target_dir = '.'
    
    project_name = os.path.basename(os.path.abspath(project_root))
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{project_name}_dump_{timestamp_str}.txt"
    
    print(f"設定專案根目錄為: {os.path.abspath(project_root)}")
    print(f"將從 '{os.path.abspath(target_dir)}' 開始掃描...")
    print(f"輸出檔案將命名為: {output_file}")
    
    generate_code_dump(target_dir, output_file, project_name)