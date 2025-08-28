#python test\dump_project.py -o core_logic.txt mujoco_playground\_src\locomotion\pupper\joystick.py mujoco_playground\_src\locomotion\pupper\joystickwithgun.py mujoco_playground\_src\locomotion\pupper\pupper_locomotion_test.ipynb

# test/dump_project.py
import os
import sys
from datetime import datetime
import subprocess
import argparse

# --- 【新增】引入 nbformat 並處理導入錯誤 ---
try:
    import nbformat
except ImportError:
    print("警告：'nbformat' 模組未安裝。'.ipynb' 檔案將以原始 JSON 格式顯示。")
    print("若要優化 .ipynb 輸出，請執行：pip install nbformat")
    nbformat = None
# ---------------------------------------------

# --- 組態設定 (無須修改) ---
EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.vscode', 'dist', 'build', 'env', '.idea', 'target', '.DS_Store', 'output'}
SKIP_CONTENT_EXTENSIONS = {'.onnx', '.stl', '.ort', '.png', '.jpg', '.jpeg', '.exe', '.dll', '.so', '.o', '.zip', '.rar', '.gz', '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.avi', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.pyc', '.pyo', '.lock', '.swp', '.swo'}
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
            stderr=subprocess.PIPE, text=True, encoding='utf-8'
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError: git 命令執行失敗 (例如，不在 git repo 中)
        # FileNotFoundError: 系統中找不到 'git' 命令
        print("警告：無法獲取 Git 分支資訊。可能不在 Git 倉庫中，或未安裝 Git。")
        return None


# --- 樹狀結構產生函式 (無須修改) ---
def generate_tree_structure(root_dir, project_name):
    # ... (此函式內容不變，為節省空間已省略)
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


# --- 【新增】.ipynb 檔案解析函式 ---
def parse_ipynb_file(file_path):
    """
    解析 .ipynb 檔案，提取程式碼和 Markdown 儲存格，並格式化為易讀的文字。
    """
    if not nbformat:
        # 如果 nbformat 模組沒有成功導入，則退回原始讀取模式
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f"[nbformat 未安裝，顯示原始 JSON 內容]\n\n" + f.read()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        output_parts = []
        for i, cell in enumerate(nb.cells):
            output_parts.append(f"#{'-'*25} In[{i+1}]: Cell type: {cell.cell_type} {'-'*25}\n")
            
            if cell.cell_type == 'code':
                # 對於程式碼儲存格，直接加入原始碼
                output_parts.append(cell.source)
            elif cell.cell_type == 'markdown':
                # 對於 Markdown 儲存格，將每一行轉換為註解
                commented_markdown = '\n'.join([f"# {line}" for line in cell.source.split('\n')])
                output_parts.append(commented_markdown)
            
            output_parts.append('\n\n')
            
        return "".join(output_parts)
    except Exception as e:
        # 如果解析出錯，也退回原始讀取模式，並附上錯誤訊息
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f"[解析 .ipynb 檔案時發生錯誤: {e}]\n\n" + f.read()


# --- 程式碼彙整核心邏輯 ---

def write_file_content_to_dump(outfile, file_path, root_dir):
    """將單一檔案的內容寫入到彙整檔中 (重構出的共用函式)"""
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
        
        # --- 【核心修改點】 ---
        if extension.lower() == '.ipynb':
            content = parse_ipynb_file(file_path)
            outfile.write(content)
        elif extension.lower() in SKIP_CONTENT_EXTENSIONS:
            file_size = os.path.getsize(file_path)
            outfile.write(f"[Content skipped for file type '{extension}': {relative_path} ({file_size / 1024:.2f} KB)]")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='strict') as infile:
                    outfile.write(infile.read())
            except (UnicodeDecodeError, IOError):
                file_size = os.path.getsize(file_path)
                outfile.write(f"[Content skipped due to read error: {relative_path} ({file_size / 1024:.2f} KB)]")
        # --- 【修改結束】 ---
        
        outfile.write(f"\n\n{'-' * len(end_header)}\n")
        outfile.write(f"{end_header}\n")
        outfile.write(f"{separator}\n\n")
        
        return 1 # 表示成功處理 1 個檔案
    except Exception as e:
        print(f"警告：無法讀取 {relative_path} ({e})")
        return 0 # 表示處理失敗

# 【模式一】完整目錄掃描函式
def generate_code_dump_from_directory(root_dir, output_filename_base, project_name, git_branch):
    """掃描整個目錄並產生彙整檔"""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename_base)

    processed_files_count = 0
    
    try:
        with open(full_output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 寫入標頭資訊
            outfile.write(f"# 專案程式碼彙整: {os.path.abspath(root_dir)}\n")
            if git_branch:
                outfile.write(f"# Git 當前分支: {git_branch}\n")
            outfile.write("=" * 80 + "\n\n")

            # 寫入目錄結構
            outfile.write("#" * 80 + "\n# 專案目錄結構\n" + "#" * 80 + "\n\n")
            tree_structure = generate_tree_structure(root_dir, project_name)
            outfile.write(tree_structure)
            outfile.write("\n\n\n")

            outfile.write("#" * 80 + "\n# 各檔案內容\n" + "#" * 80 + "\n\n")
            
            # 遍歷目錄
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                for filename in sorted(filenames):
                    # 避免讀取到自己或其他 dump 檔案
                    is_current_output_file = os.path.normpath(os.path.join(dirpath, filename)) == os.path.normpath(full_output_path)
                    if filename == os.path.basename(__file__) or is_current_output_file:
                        continue
                    
                    file_path = os.path.join(dirpath, filename)
                    processed_files_count += write_file_content_to_dump(outfile, file_path, root_dir)

        print_summary(processed_files_count, full_output_path)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{full_output_path}'。 ({e})")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

# 【模式二】指定檔案彙整函式
def generate_code_dump_from_files(file_list, root_dir, output_filename_base, project_name, git_branch):
    """僅彙整提供的檔案列表"""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename_base)

    processed_files_count = 0

    try:
        with open(full_output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 寫入標頭資訊
            outfile.write(f"# 指定檔案彙整: {os.path.abspath(root_dir)}\n")
            if git_branch:
                outfile.write(f"# Git 當前分支: {git_branch}\n")
            outfile.write("=" * 80 + "\n\n")

            # 寫入被包含的檔案列表，取代目錄樹
            outfile.write("#" * 80 + "\n# 包含的檔案清單\n" + "#" * 80 + "\n\n")
            for file_path in file_list:
                outfile.write(f"- {os.path.relpath(file_path, root_dir).replace(os.sep, '/')}\n")
            outfile.write("\n\n\n")

            outfile.write("#" * 80 + "\n# 各檔案內容\n" + "#" * 80 + "\n\n")
            
            # 只處理指定的檔案
            for file_path in file_list:
                processed_files_count += write_file_content_to_dump(outfile, file_path, root_dir)

        print_summary(processed_files_count, full_output_path)

    except IOError as e:
        print(f"錯誤：無法寫入輸出檔案 '{full_output_path}'。 ({e})")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

def print_summary(count, path):
    """印出最終的成功訊息"""
    print("\n" + "=" * 80)
    print(f"✅ 成功！共處理了 {count} 個檔案。")
    print(f"輸出結果已儲存至: {os.path.abspath(path)}")
    print("=" * 80)

# --- 主程式進入點 (if __name__ == "__main__":) ---
# ... (此區塊內容不變，為節省空間已省略)
if __name__ == "__main__":
    # --- 【核心修改】設定命令列參數，支援指定檔案 ---
    parser = argparse.ArgumentParser(
        description="彙整專案程式碼成單一文字檔。可掃描整個目錄，或只彙整指定的檔案。",
        formatter_class=argparse.RawTextHelpFormatter # 保持換行格式
    )
    parser.add_argument(
        'files', # 位置參數
        nargs='*', # 0 或多個
        help="要彙整的一個或多個檔案路徑。\n如果留空，則會掃描整個專案目錄。"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="指定輸出的檔案名稱 (例如 'my_snapshot.txt')。\n如果未指定，將會自動產生檔名。"
    )
    args = parser.parse_args()
    # ----------------------------------------------------

    #【建議修改】移除 os.chdir，讓路徑處理更可靠
    script_path = os.path.dirname(os.path.abspath(__file__))
    # 假設 dump_project.py 在專案的某個子目錄 (如 'test/')
    project_root = os.path.dirname(script_path) 
    
    # os.chdir(project_root) # <--- 建議註解或刪除此行

    project_name = os.path.basename(os.path.abspath(project_root))
    current_branch = get_git_branch()
    
    # 決定輸出檔名
    output_filename_base = ""
    if args.output:
        output_filename_base = args.output
        print(f"✅ 使用者已指定輸出檔名...")
    else:
        print("未指定輸出檔名，將自動產生...")
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if current_branch:
            safe_branch_name = current_branch.replace('/', '-')
            output_filename_base = f"{project_name}_{safe_branch_name}_dump_{timestamp_str}.txt"
        else:
            output_filename_base = f"{project_name}_dump_{timestamp_str}.txt"

    print(f"設定專案根目錄為: {os.path.abspath(project_root)}")
    if current_branch:
        print(f"偵測到目前 Git 分支為: {current_branch}")
    print(f"輸出檔案將命名為: {output_filename_base} (儲存於 output/ 目錄)")

    # --- 【核心修改】根據是否有指定檔案，決定執行哪個模式 ---
    if args.files:
        # **檔案模式**
        print(f"\n✅ 偵測到指定檔案模式，將處理 {len(args.files)} 個檔案...")
        
        # 驗證檔案是否存在
        valid_files = []
        for f in args.files:
            abs_f = os.path.abspath(f) # <--- 【新增】轉換為絕對路徑
            if os.path.isfile(abs_f):  # <--- 【修改】用絕對路徑來檢查
                valid_files.append(abs_f) # <--- 【修改】儲存絕對路徑
            else:
                # 警告訊息中也使用絕對路徑，方便除錯
                print(f"警告：檔案 '{abs_f}' 不存在或不是一個檔案，將被跳過。")
        
        if not valid_files:
            print("\n錯誤：所有指定的路徑都無效，沒有任何檔案可以處理。")
        else:
            # 傳入絕對路徑列表
            generate_code_dump_from_files(valid_files, project_root, output_filename_base, project_name, current_branch)
    else:
        # **目錄模式**
        print(f"\n✅ 偵測到目錄掃描模式，將從 '{os.path.abspath(project_root)}' 開始掃描...")
        generate_code_dump_from_directory(project_root, output_filename_base, project_name, current_branch)