# test/project_overview.py

import os
import sys
import argparse
import fnmatch
import yaml
from datetime import datetime
from io import StringIO # 用於在記憶體中緩衝輸出，以便在文件開頭插入摘要
from typing import Dict, Any, List, Optional # 類型提示

# ====================================================================
# 1. 語言映射配置 (LANGUAGE_MAP)
#    - 目的: 將檔案的副檔名映射到 Markdown 程式碼區塊所需的語言提示。
#    - 說明: 這些提示會告訴 Markdown 渲染器如何對程式碼進行語法高亮。
# ====================================================================
LANGUAGE_MAP: Dict[str, str] = {
    '.py': 'python',
    '.xml': 'xml',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.md': 'markdown',
    '.txt': 'text',
    '.json': 'json',
    '.cpp': 'cpp',
    '.h': 'cpp',
    '.c': 'c',
    '.js': 'javascript',
    '.html': 'html',
    '.css': 'css',
    '.sh': 'bash',
    '.gitignore': 'text', # .gitignore 檔案雖然沒有語言高亮，但歸類為 text
    '.conf': 'text',       # .conf 檔案也歸類為 text
    # 可以根據需要添加更多副檔名到對應的語言提示
}

# ====================================================================
# 2. 默認配置參數 (DEFAULT_CONFIG)
#    - 目的: 提供 project_overview.py 工具的所有可配置選項的默認值。
#    - 說明: 這些值會在沒有配置文件或命令行參數提供時生效。
# ====================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # 專案掃描根目錄的絕對路徑。如果留空，腳本會使用執行時的當前工作目錄。
    'project_root_path': '',
    
    # 在生成目錄樹和遍歷文件時要完全排除的目錄名稱列表。
    # 這些目錄及其內容將不會出現在報告中。
    'exclude_directories': [
        '.git',             # Git 版本控制文件
        'node_modules',     # JavaScript 依賴包
        '__pycache__',      # Python 編譯緩存
        'venv',             # Python 虛擬環境
        '.vscode',          # VS Code 配置文件
        '.idea',            # PyCharm/IntelliJ IDEA 配置文件
        'dist',             # 編譯後的發佈目錄
        'build',            # 編譯中間文件
        'env',              # 通用環境目錄
        'target',           # Maven/Gradle 等的編譯輸出
        '.DS_Store',        # macOS 系統文件
        'venv_test_no_mujoco', # 特定虛擬環境
        'output',           # project_overview.py 自身的輸出目錄 (防止循環包含)
        '.mypy_cache'       # MyPy 類型檢查器緩存
    ],
    
    # 內容將被跳過，只在報告中顯示文件名和大小的檔案副檔名列表。
    # 通常用於二進制文件或大型非文本資產，避免報告過大或顯示亂碼。
    'skip_content_extensions': [
        '.onnx', '.stl', '.ort',          # 模型文件
        '.png', '.jpg', '.jpeg', '.gif',  # 圖片文件
        '.bmp', '.ico',                   # 圖片文件
        '.mp3', '.mp4', '.avi',           # 媒體文件
        '.pdf', '.doc', '.docx', '.xls',  # 文檔文件
        '.xlsx', '.ppt', '.pptx',         # 文檔文件
        '.pyc', '.pyo',                   # Python 字節碼
        '.lock', '.swp', '.swo'           # 臨時文件/鎖定文件
    ],
    
    # 如果此列表不為空，則只有這些副檔名的檔案內容會被包含在報告中。
    # 其他所有檔案（即使不在 skip_content_extensions 中）的內容都將被跳過。
    # 默認留空，表示此功能禁用，腳本將包含所有未被明確跳過或排除的內容。
    'include_only_extensions': [], 
    
    # 排除符合 fnmatch 模式的檔案名列表。
    # 用於排除特定名稱或模式的檔案，例如舊的 dump 文件，或特定的測試腳本。
    'exclude_file_patterns': [
        "project_dump.txt",         # 舊版 dump_project.py 生成的單一文件
        "*_dump_*.txt",             # 匹配動態生成的舊版 dump 文件
        "project_overview.py",      # 工具自身的源碼 (防止報告中包含報告源碼)
        "test_pyserial_console.py", # 特定測試腳本
        "test_joystick.py",         # 特定測試腳本
        "test_serial_utils.py",     # 特定測試腳本
        "test_teensy_connection.py",# 特定測試腳本
        "verify_model_mode.py"      # 特定測試腳本
    ],
    
    # 每個文本檔案的內容最大行數。如果檔案內容超過此限制，將被截斷。
    # 設置為 0 或 None 表示不進行截斷，包含文件所有內容。
    'max_lines_per_file': 300,
    
    # 報告的輸出目錄。相對於腳本執行目錄。
    'output_directory': 'output',
    
    # 報告的輸出格式: "text" (純文本，適合終端機) 或 "markdown" (Markdown 格式，適合閱讀器或 AI 分析)。
    'output_format': 'markdown',
    
    # 專案名稱。用於生成輸出文件的文件名和報告標題。
    # 如果留空，腳本會自動從 'project_root_path' 推斷專案名稱。
    'project_name': '',
    
    # 輸出文件名的前綴。例如，如果設置為 "my_report_"，文件名將是 "my_report_projectname_overview_..."。
    # 如果留空，文件名將直接以 'project_name' 開頭。
    'output_filename_prefix': '',
    
    # 是否在輸出文件名中包含時間戳 (例如 project_name_overview_YYYYMMDD_HHMMSS.md)。
    'add_timestamp_to_filename': True,
}

# ====================================================================
# 3. 輔助函式：載入配置 (load_overview_config)
#    - 目的: 從指定的 YAML 配置文件載入配置，並與默認配置合併。
#    - 運作邏輯:
#      - 複製 DEFAULT_CONFIG 作為基礎。
#      - 檢查用戶提供的配置文件是否存在。
#      - 如果存在，安全地載入並使用其中的值覆蓋默認值。
#        - 對於列表類型的配置（如 exclude_directories），用戶配置文件中的列表將完全替換默認列表。
#      - 如果載入失敗，則發出警告並繼續使用默認配置。
# ====================================================================
def load_overview_config(config_file_path: str) -> Dict[str, Any]:
    """
    從指定的 YAML 配置文件載入配置，並與默認配置合併。
    命令行參數將在後續步驟中覆蓋此處載入的配置。
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                for key, value in user_config.items():
                    if key in config and isinstance(config[key], list) and isinstance(value, list):
                        config[key] = value # 用戶列表完全替換默認列表
                    else:
                        config[key] = value # 其他類型或新鍵直接覆蓋/添加
            print(f"✅ 載入配置檔案: '{config_file_path}'")
        except Exception as e:
            print(f"❌ 警告: 無法載入或解析配置檔案 '{config_file_path}': {e}。\n"
                  f"    將使用默認配置或部分載入的配置。")
    return config

# ====================================================================
# 4. 樹狀結構生成 (generate_ascii_tree_structure)
#    - 目的: 生成專案的目錄樹結構，以 ASCII Art 形式返回字符串。
#    - 設計考量: 這個函式獨立於最終的輸出格式（Markdown/Text），
#      只負責生成純粹的 ASCII Art 內容。這樣可以確保無論何種輸出格式，
#      底層的樹狀圖生成邏輯都是一致且獨立可測試的。
#      在 Markdown 模式下，這個 ASCII Art 內容會被包裹在一個程式碼區塊內。
# ====================================================================
def generate_ascii_tree_structure(root_dir: str, project_name: str, exclude_dirs: List[str]) -> str:
    """
    生成專案的目錄樹結構，以 ASCII Art 形式返回字符串。
    
    Args:
        root_dir (str): 專案的根目錄路徑。
        project_name (str): 專案的名稱。
        exclude_dirs (List[str]): 要在樹狀圖中標記為「排除」的目錄名稱列表。
        
    Returns:
        str: 包含 ASCII Art 目錄樹的字符串。
    """
    tree_lines: List[str] = []
    # 頂層顯示專案名稱作為根目錄
    tree_lines.append(f"{project_name}/\n")

    def _recursive_tree(current_dir: str, prefix: str = "", is_last_parent: bool = False) -> None:
        """
        遞歸生成目錄樹。
        
        Args:
            current_dir (str): 當前遍歷的目錄路徑。
            prefix (str): 當前行前的連接線前綴 (如 "│   " 或 "    ")。
            is_last_parent (bool): 父目錄是否是其父級的最後一個子項，影響連接線的繪製。
        """
        try:
            items_in_dir = os.listdir(current_dir)
        except OSError as e:
            # 如果無法訪問目錄，打印錯誤信息並返回
            tree_lines.append(f"{prefix}├── <無法訪問目錄: {current_dir} - {e}>\n")
            return

        dirs_to_process: List[str] = []
        files_to_process: List[str] = []

        for item_name in sorted(items_in_dir):
            item_path = os.path.join(current_dir, item_name)
            
            # 判斷是否為目錄且在排除列表中。
            # 如果是，在樹狀圖中列出並標記為 (Excluded)，但不會遞歸其內容。
            if os.path.isdir(item_path) and item_name in exclude_dirs:
                tree_lines.append(f"{prefix}├── {item_name}/ (Excluded)\n")
                continue # 跳過對其內容的處理
                
            if os.path.isdir(item_path):
                dirs_to_process.append(item_name)
            elif os.path.isfile(item_path):
                files_to_process.append(item_name)

        # 將目錄和文件合併並排序，以便一致性輸出
        all_entries_sorted = sorted(dirs_to_process) + sorted(files_to_process)

        for i, item_name in enumerate(all_entries_sorted):
            is_last_entry = (i == len(all_entries_sorted) - 1) # 當前項是否是最後一個子項
            connector = "└── " if is_last_entry else "├── " # 決定是 "L" 形連接還是 "T" 形連接
            
            # 計算當前行的前綴：繼承父級前綴，並根據父級是否是最後一個子項來決定畫 '|' 或 ' '
            current_item_prefix = prefix + ("    " if is_last_parent else "│   ")

            # 將當前項添加到樹狀圖列表中
            tree_lines.append(f"{current_item_prefix}{connector}{item_name}{'/' if os.path.isdir(item_path) else ''}\n")
            
            # 如果是目錄，則遞歸調用自身以處理其子項
            if os.path.isdir(item_path):
                _recursive_tree(item_path, current_item_prefix, is_last_entry)
        
    # 從根目錄開始遞歸生成樹狀圖
    _recursive_tree(root_dir, prefix="", is_last_parent=False)
    
    # 將所有行連接成一個字符串並返回
    return "".join(tree_lines)


# ====================================================================
# 5. 程式碼彙整主函式 (generate_project_overview)
#    - 目的: 根據載入的配置和命令行參數，生成最終的專案概覽報告。
#    - 運作邏輯:
#      a. 讀取並確定最終配置。
#      b. 構建輸出文件名和路徑。
#      c. 初始化統計數據和錯誤列表。
#      d. 使用 `StringIO` 作為緩衝區，將報告的「文件內容」部分寫入記憶體。
#      e. 遍歷專案目錄，對每個文件進行過濾、內容讀取、截斷、格式化。
#      f. 在完成所有文件處理後，根據收集到的統計數據和格式，組裝最終報告的各個部分（主標題、配置、摘要、目錄樹）。
#      g. 最後將組裝好的報告內容寫入實際的輸出檔案。
# ====================================================================
def generate_project_overview(config: Dict[str, Any]):
    """根據配置生成專案概覽檔案。"""

    # 從最終配置中提取核心參數
    project_root = config['project_root_path']
    output_dir = config['output_directory']
    output_format = config['output_format']
    project_name = config['project_name']
    
    # 構建輸出文件名
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") if config['add_timestamp_to_filename'] else ""
    output_ext = "md" if output_format == "markdown" else "txt"
    
    filename_parts: List[str] = []
    if config['output_filename_prefix']:
        filename_parts.append(config['output_filename_prefix'])
    
    sanitized_project_name = project_name.replace(' ', '_').lower() # 專案名轉為文件名友好格式
    filename_parts.append(sanitized_project_name)
    
    filename_parts.append("overview") # 文件名固定部分，明確其為概覽報告
    
    if timestamp_str:
        filename_parts.append(timestamp_str)
    
    output_filename_base = "_".join(filter(None, filename_parts)) + f".{output_ext}"
    full_output_path = os.path.join(output_dir, output_filename_base)

    # 執行前的檢查與準備
    if not os.path.isdir(project_root):
        print(f"❌ 錯誤：掃描根目錄 '{project_root}' 不存在。請檢查 'project_root_path' 配置。")
        return

    os.makedirs(output_dir, exist_ok=True) # 確保輸出目錄存在

    # 初始化統計數據和錯誤記錄
    total_files_processed = 0
    files_with_content_skipped = 0
    files_with_content_truncated = 0
    files_with_read_errors = 0
    total_lines_truncated = 0
    errors_encountered: List[str] = []
    
    # 使用 StringIO 作為記憶體緩衝區，先將「各檔案內容」部分寫入其中。
    # 這樣，我們可以在遍歷完所有文件後，計算出統計數據，
    # 再將這些數據（以及配置）寫入最終文件的開頭。
    content_buffer = StringIO()

    # 根據輸出格式定義不同部分的文本樣式和分隔符
    # 這樣可以避免在每個打印點進行 if/else 判斷，使程式碼更簡潔。
    if output_format == 'markdown':
        section_title_level = "##" # 主要章節標題 (e.g., ## 專案目錄結構)
        file_title_level = "###"  # 單個檔案標題 (e.g., ### 檔案: `path/to/file.py`)
        header_separator = "\n---\n\n" # 各章節之間的分隔線 (Markdown 橫線)
        code_block_start = lambda lang: f"```{lang}\n" # 程式碼塊開始標記 (Markdown 語法)
        # 【修改】確保程式碼塊結束標記前有換行
        code_block_end = "\n```\n"                    # 程式碼塊結束標記 (Markdown 語法)
        # 內容被跳過、讀取失敗、內容截斷的註解格式 (Markdown 註解)
        skipped_content_comment = lambda ext, filename, size: f"<!-- Content skipped, type '{ext}': {filename} ({size / 1024:.2f} KB) -->\n"
        read_error_comment = lambda filename, size: f"<!-- Content read failed: {filename} ({size / 1024:.2f} KB) -->\n"
        truncated_comment = lambda lines_skipped: f"<!-- ... content truncated - {lines_skipped} lines skipped ... -->\n"
        # 每個檔案內容結束後的標記 (Markdown 註解)
        file_end_marker = lambda path: f"<!-- END OF FILE: {path} -->\n\n"
    else: # text format (純文本)
        section_title_level = "==" # 主要章節標題 (純文本分隔線)
        file_title_level = "---"   # 單個檔案標題 (純文本分隔線)
        header_separator = "\n" + "=" * 80 + "\n\n" # 各章節之間的分隔線 (純文本)
        code_block_start = lambda lang: "" # 純文本模式無代碼塊開始標記
        code_block_end = ""                # 純文本模式無代碼塊結束標記
        skipped_content_comment = lambda ext, filename, size: f"[Content skipped for file type '{ext}': {filename} ({size / 1024:.2f} KB)]\n"
        read_error_comment = lambda filename, size: f"[Content read failed: {filename} ({size / 1024:.2f} KB)]\n"
        truncated_comment = lambda lines_skipped: f"[...truncated - {lines_skipped} lines skipped...]\n"
        file_end_marker = lambda path: f"\n\n{'-' * (len(path) + 17)}\n--- END OF FILE: {path} ---\n" + "=" * 80 + "\n\n" # 檔案結束標記

    try:
        # -------------------------------------------------------------
        # 步驟 1: 遍歷專案目錄，處理文件內容並寫入記憶體緩衝區
        # -------------------------------------------------------------
        for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
            # 過濾掉 'exclude_directories' 中定義的目錄，確保不遍歷其內部文件。
            # 這會影響 `os.walk` 的行為，使其跳過這些目錄，而不是僅僅在報告中標記它們。
            dirnames[:] = [d for d in dirnames if d not in config['exclude_directories']]

            for filename in sorted(filenames):
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path, project_root).replace(os.sep, '/')
                
                _, extension = os.path.splitext(file_path)
                extension_lower = extension.lower()

                # ----------------- 文件過濾邏輯 -----------------
                # 判斷是否應排除此文件 (根據 exclude_file_patterns)
                if any(fnmatch.fnmatch(filename, pattern) for pattern in config['exclude_file_patterns']):
                    continue # 跳過此文件，不將其添加到報告中
                
                # 判斷是否應只包含特定類型文件 (如果 include_only_extensions 不為空)
                if config['include_only_extensions'] and extension_lower not in config['include_only_extensions']:
                    continue # 跳過此文件，不將其添加到報告中
                # ------------------------------------------------

                print(f"處理中: {relative_path}") # 控制台通知當前處理的文件

                # 將檔案標題寫入緩衝區
                content_buffer.write(f"{file_title_level} 檔案: `{relative_path}`\n\n")
                
                # 檢查是否屬於「跳過內容」的文件類型（例如二進制文件）
                if extension_lower in config['skip_content_extensions']:
                    file_size = os.path.getsize(file_path)
                    content_buffer.write(skipped_content_comment(extension_lower, filename, file_size))
                    files_with_content_skipped += 1 # 統計：內容被跳過的檔案數
                else:
                    # 處理文本文件內容，包括截斷邏輯
                    try:
                        file_content_lines: List[str] = []
                        lines_read = 0               # 實際讀取並添加到緩衝的行數
                        lines_in_file = 0            # 文件的總行數（用於統計截斷了多少行）
                        lines_skipped_current_file = 0 # 當前文件因截斷而跳過的行數
                        
                        with open(file_path, 'r', encoding='utf-8', errors='strict') as infile:
                            for line in infile:
                                lines_in_file += 1 # 統計文件的總行數
                                # 如果配置了最大行數限制，且已達到限制，則後續行將被跳過
                                if config['max_lines_per_file'] is not None and \
                                   config['max_lines_per_file'] > 0 and \
                                   lines_read >= config['max_lines_per_file']:
                                    lines_skipped_current_file += 1
                                else:
                                    file_content_lines.append(line)
                                    lines_read += 1
                        
                        lang = LANGUAGE_MAP.get(extension_lower, 'text')
                        content_buffer.write(code_block_start(lang)) # 寫入程式碼塊的開始標記
                        content_buffer.writelines(file_content_lines) # 寫入文件內容
                        
                        # 如果有內容被截斷，添加截斷提示並更新統計
                        if lines_skipped_current_file > 0:
                            content_buffer.write(truncated_comment(lines_skipped_current_file))
                            files_with_content_truncated += 1 # 統計：內容被截斷的檔案數
                            total_lines_truncated += lines_skipped_current_file # 統計：總計截斷的行數
                            print(f"  ⚠️ 已截斷內容 (跳過 {lines_skipped_current_file} 行): {relative_path}") # 控制台即時通知
                        
                        # 【修改】確保程式碼塊結束標記前有換行
                        # if not file_content_lines[-1].endswith('\n'): # 這種判斷方式只適用於文件不為空的情況
                        # 更通用的做法是，直接在 code_block_end 中包含換行，或者在寫入 code_block_end 前寫入換行。
                        # 目前 code_block_end 中已經包含 `\n`，所以它會獨佔一行。
                        # 要讓它 *前面* 有換行，並且即使內容是空或只有一行也要如此，可以這樣做：
                        # if output_format == 'markdown': # 只有 Markdown 需要這個保證
                        #     if not content_buffer.getvalue().endswith('\n```\n'): # 檢查是否已正確結尾
                        #         content_buffer.write('\n') # 添加一個空行，確保 ```` ` 獨立

                        content_buffer.write(code_block_end) # 寫入程式碼塊的結束標記

                    except (UnicodeDecodeError, IOError) as e:
                        # 處理文件讀取錯誤 (例如：文件不是 UTF-8 編碼，或權限不足)
                        file_size = os.path.getsize(file_path)
                        content_buffer.write(read_error_comment(filename, file_size))
                        files_with_read_errors += 1 # 統計：讀取失敗的檔案數
                        errors_encountered.append(f"讀取錯誤: {relative_path} - {e}")
                        print(f"  ❌ 讀取錯誤: {relative_path} - {e}") # 控制台即時通知
                    except Exception as e:
                        # 捕獲其他未知錯誤
                        content_buffer.write(f"<!-- 未知錯誤: {filename} - {e} -->\n")
                        errors_encountered.append(f"未知錯誤: {relative_path} - {e}")
                        print(f"  ❌ 未知錯誤: {relative_path} - {e}") # 控制台即時通知

                # 每個檔案內容塊結束後，添加統一的結束標記
                content_buffer.write(file_end_marker(relative_path))
                
                total_files_processed += 1 # 統計：總處理文件數（包括跳過內容和截斷的）

        # -------------------------------------------------------------
        # 步驟 2: 將組裝好的報告寫入最終輸出文件
        #   - 在這裡，我們會將報告的各個部分（主標題、配置、摘要、目錄樹、以及之前緩衝的文件內容）
        #     按照順序寫入最終的輸出檔案。
        # -------------------------------------------------------------
        with open(full_output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 2.1 報告主標題和生成時間
            if output_format == 'markdown':
                outfile.write(f"# 專案程式碼概覽: {project_name}\n")
                outfile.write(f"*生成於: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            else:
                outfile.write(f"# 專案程式碼概覽: {project_name} (生成於: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                outfile.write("=" * 80 + "\n\n")

            # 2.2 生成配置部分
            outfile.write(f"{section_title_level} 生成配置\n")
            outfile.write(header_separator)
            outfile.write(code_block_start('yaml')) # 將配置作為 YAML 代碼塊輸出
            # 顯示的配置去除內部處理的 project_root_path 和 project_name，只顯示用戶關心的配置
            config_to_display = config.copy()
            config_to_display.pop('project_root_path', None)
            config_to_display.pop('project_name', None)
            outfile.write(yaml.dump(config_to_display, default_flow_style=False, sort_keys=False, allow_unicode=True))
            outfile.write(code_block_end)
            outfile.write(header_separator)

            # 2.3 報告摘要部分
            outfile.write(f"{section_title_level} 報告摘要\n")
            outfile.write(header_separator)
            summary_lines: List[str] = [
                f"- 掃描根目錄: `{project_root}`",
                f"- 總處理文件數: {total_files_processed}",
                f"- 內容被跳過的二進制/非文本文件數: {files_with_content_skipped}",
                f"- 內容被截斷的文本文件數: {files_with_content_truncated}",
                f"- 總計被截斷行數: {total_lines_truncated}",
                f"- 讀取失敗文件數: {files_with_read_errors}",
            ]
            if not errors_encountered:
                summary_lines.append("- 所有文件均無讀取錯誤。")
            else:
                summary_lines.append(f"- 發現 {len(errors_encountered)} 個讀取錯誤。詳情請參閱底部錯誤列表。")
            
            if output_format == 'markdown':
                outfile.write("\n".join(summary_lines) + "\n\n")
            else:
                for line in summary_lines:
                    outfile.write(line + "\n")
                outfile.write("\n")
            outfile.write(header_separator)

            # 2.4 目錄結構部分
            outfile.write(f"{section_title_level} 專案目錄結構\n")
            outfile.write(header_separator)
            
            # 生成 ASCII Art 樹狀圖，並根據輸出格式決定是否包裹在程式碼塊中
            ascii_tree = generate_ascii_tree_structure(project_root, project_name, config['exclude_directories'])
            if output_format == 'markdown':
                outfile.write(code_block_start('text')) # Markdown 中用 text 代碼塊包裹 ASCII Art
                outfile.write(ascii_tree)
                outfile.write(code_block_end)
            else: # text format
                outfile.write(ascii_tree)
            outfile.write(header_separator)

            # 2.5 各檔案內容部分 (從記憶體緩衝區寫入)
            outfile.write(f"{section_title_level} 各檔案內容\n")
            outfile.write(header_separator)
            outfile.write(content_buffer.getvalue()) # 將緩衝區的所有內容寫入

            # 2.6 錯誤列表部分 (如果有錯誤)
            if errors_encountered:
                outfile.write(f"{section_title_level} 錯誤列表\n")
                outfile.write(header_separator)
                if output_format == 'markdown':
                    for error in errors_encountered:
                        outfile.write(f"- `{error}`\n") # 使用反引號包裹錯誤路徑，使其在 Markdown 中高亮
                else:
                    for error in errors_encountered:
                        outfile.write(f"- {error}\n")
                outfile.write("\n") # 確保末尾有空行

        # -------------------------------------------------------------
        # 步驟 3: 完成所有寫入後，在控制台打印最終總結
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"✅ 成功！共處理了 {total_files_processed} 個檔案。")
        print(f"輸出結果已儲存至: {os.path.abspath(full_output_path)}")
        if errors_encountered:
            print("\n❌ 運行過程中發現以下錯誤:")
            for error in errors_encountered:
                print(f"  - {error}")
        print("=" * 80)

    except IOError as e:
        print(f"❌ 錯誤：無法寫入輸出檔案 '{full_output_path}'。 ({e})")
        print(f"詳情: {e}")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
        import traceback # 打印完整的錯誤堆棧
        traceback.print_exc()


# ====================================================================
# 6. 命令行參數解析與腳本執行入口
#    - 目的: 負責解析命令行參數，載入配置文件，並啟動報告生成流程。
#    - 運作邏輯:
#      - 定義所有可用的命令行參數及其類型和幫助信息。
#      - 首先載入配置文件中定義的默認配置。
#      - 然後，如果用戶在命令行中提供了相應的參數，則命令行參數的值將覆蓋配置文件中的設定。
#      - 處理 project_root_path 和 project_name 的自動推斷邏輯。
#      - 最後調用 generate_project_overview 函式來生成報告。
# ====================================================================
if __name__ == "__main__":
    # 配置命令行參數解析器
    parser = argparse.ArgumentParser(description="生成專案程式碼概覽檔案。")
    parser.add_argument("--config", type=str, default="project_overview_config.yaml",
                        help="指定配置檔案的路徑 (默認: project_overview_config.yaml)")
    parser.add_argument("--output-dir", type=str,
                        help="輸出目錄 (覆蓋配置檔案設定)")
    parser.add_argument("--format", type=str, choices=['text', 'markdown'],
                        help="輸出格式 (text 或 markdown, 覆蓋配置檔案設定)")
    parser.add_argument("--max-lines-per-file", type=int,
                        help="每個檔案的最大行數，超過則截斷 (覆蓋配置檔案設定)")
    parser.add_argument("--project-root", type=str,
                        help="掃描的專案根目錄路徑 (覆蓋配置檔案設定)")
    parser.add_argument("--project-name", type=str,
                        help="專案名稱 (用於報告和文件名，覆蓋配置檔案設定)")
    parser.add_argument("--output-filename-prefix", type=str,
                        help="輸出文件名的前綴 (覆蓋配置檔案設定)")
    parser.add_argument("--add-timestamp", type=lambda x: x.lower() in ('true', '1', 't'),
                        help="是否在文件名中包含時間戳 (true/false, 覆蓋配置檔案設定)")
    
    args = parser.parse_args()

    # 載入配置文件
    config = load_overview_config(args.config)

    # 命令行參數覆蓋配置文件設定 (命令行參數的優先級最高)
    # 這裡使用 `is not None` 而不是 `if args.xxx:` 是為了允許用戶將參數設置為空字串或 0 (例如 max_lines_per_file=0 表示不截斷)
    if args.output_dir is not None:
        config['output_directory'] = args.output_dir
    if args.format is not None:
        config['output_format'] = args.format
    if args.max_lines_per_file is not None:
        config['max_lines_per_file'] = args.max_lines_per_file
    if args.project_root is not None:
        config['project_root_path'] = args.project_root
    if args.project_name is not None:
        config['project_name'] = args.project_name
    if args.output_filename_prefix is not None:
        config['output_filename_prefix'] = args.output_filename_prefix
    if args.add_timestamp is not None:
        config['add_timestamp_to_filename'] = args.add_timestamp
    
    # 後續處理：如果 project_root_path 仍為空，則設置為當前工作目錄的絕對路徑
    if not config['project_root_path']:
        config['project_root_path'] = os.path.abspath(os.getcwd())
    
    # 後續處理：如果 project_name 仍為空，則從 project_root_path 推斷
    if not config['project_name']:
        config['project_name'] = os.path.basename(config['project_root_path'])

    # 執行生成報告
    generate_project_overview(config)