# test/dump_project.py
import os
import sys
from datetime import datetime
import subprocess  # 【新增】匯入 subprocess 模組

# --- 組態設定 ---

# 1. 要忽略的資料夾名稱
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.vscode',
    'dist', 'build', 'env', '.idea', 'target', '.DS_Store', 'venv_test_no_mujoco',
    'output' # 【既有】忽略 output 目錄
}

# 2. 定義一個「內容跳過清單」。
SKIP_CONTENT_EXTENSIONS = {
    '.onnx', '.stl', '.ort', '.png', '.jpg', '.jpeg',
    '.exe', '.dll', '.so', '.o', '.zip', '.rar', '.gz',
    '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.avi',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pyc', '.pyo', '.lock', '.swp', '.swo','.pyc',
}

# 3. (可選) 如果只想包含特定類型的檔案，可以設定這個清單
INCLUDE_EXTENSIONS = set()


# --- 【新增】獲取 Git 分支名稱的函式 ---
def get_git_branch():
    """
    嘗試獲取當前的 Git 分支名稱。
    如果成功，返回分支名稱字串。
    如果失敗 (例如不在 Git 倉庫中或未安裝 Git)，則返回 None。
    """
    try:
        # 'git rev-parse --abbrev-ref HEAD' 是獲取當前分支名稱的標準且安全的方式
        result = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.PIPE, # 抑制錯誤訊息輸出到控制台
            text=True, # 將輸出解碼為文字
            encoding='utf-8'
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError: git 命令執行失敗 (例如，不在 git repo 中)
        # FileNotFoundError: 系統中找不到 'git' 命令
        print("警告：無法獲取 Git 分支資訊。可能不在 Git 倉庫中，或未安裝 Git。")
        return None


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

        # 【修改】 EXCLUDE_DIRS 現在包含了 'output'
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d)) and d not in EXCLUDE_DIRS])
        
        # 【修改】過濾掉我們自己產生的輸出檔案
        files_to_process = []
        for f in sorted([f for f in items if os.path.isfile(os.path.join(current_dir, f))]):
            is_old_dump = f.startswith(f"{project_name}_dump_") and f.endswith(".txt")
            is_legacy_dump = f == "project_dump.txt"
            # 這裡的邏輯是檢查文件名本身，所以即使檔案在 `output/` 下，只要文件名符合模式，它就會被忽略。
            # 這確保了 `dump_project.py` 不會嘗試讀取自己的歷史輸出。
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
def generate_code_dump(root_dir, output_filename_base, project_name): # 函式簽名改變，接收基礎文件名
    if not os.path.isdir(root_dir):
        print(f"錯誤：目錄 '{root_dir}' 不存在。")
        return

    # 【修改】指定輸出目錄並確保其存在
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename_base) # 構造完整輸出路徑

    processed_files_count = 0
    
    try:
        with open(full_output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 【修改】在檔案開頭寫入 Git 分支資訊
            outfile.write(f"# 專案程式碼彙整: {os.path.abspath(root_dir)}\n")
            if git_branch:
                outfile.write(f"# Git 當前分支: {git_branch}\n")
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
                # 【修改】在此處過濾掉不應遍歷的目錄 (包括 output/)
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
        print(f"輸出結果已儲存至: {os.path.abspath(full_output_path)}") # 【修改】顯示完整路徑
        print("=" * 80)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{full_output_path}'。 ({e})") # 【修改】顯示完整路徑
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)
    
    os.chdir(project_root) # 確保在專案根目錄下運行
    
    target_dir = '.' # 從專案根目錄開始掃描
    
    project_name = os.path.basename(os.path.abspath(project_root))
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 【修改】只生成基礎文件名，路徑拼接在 generate_code_dump 內部處理
    output_filename_base = f"{project_name}_dump_{timestamp_str}.txt"
    
    # 【新增】呼叫函式以獲取 Git 分支名稱
    current_branch = get_git_branch()
    
    print(f"設定專案根目錄為: {os.path.abspath(project_root)}")
    # 【新增】在控制台輸出中也顯示分支名稱
    if current_branch:
        print(f"偵測到目前 Git 分支為: {current_branch}")
    print(f"將從 '{os.path.abspath(target_dir)}' 開始掃描...")
    print(f"輸出檔案將命名為: {output_filename_base} (儲存於 output/ 目錄)")
    
    # 【修改】將獲取到的分支名稱傳入主函式
    generate_code_dump(target_dir, output_filename_base, project_name, current_branch)