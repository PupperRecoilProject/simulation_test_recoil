# test/project_overview.py

import os
import sys
import argparse
import fnmatch
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional # 【新增】類型提示

# ====================================================================
# 【新增】語言映射：將文件副檔名映射到 Markdown 程式碼區塊的語言提示
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
    '.gitignore': 'text',
    '.conf': 'text', # 例如，為配置文件添加
    # 可以根據需要添加更多
}

# ====================================================================
# 【新增】默認配置參數 (在沒有配置文件或命令行參數時使用)
# ====================================================================
# 默認配置值現在定義在這裡
DEFAULT_CONFIG: Dict[str, Any] = {
    'project_root_path': '', # 空字串表示使用腳本執行時的當前工作目錄
    'exclude_directories': ['.git', 'node_modules', '__pycache__', 'venv', '.vscode',
                            '.idea', 'dist', 'build', 'env', 'target', '.DS_Store',
                            'venv_test_no_mujoco', 'output', '.mypy_cache'],
    'skip_content_extensions': ['.onnx', '.stl', '.ort', '.png', '.jpg', '.jpeg',
                                '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.avi',
                                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt',
                                '.pptx', '.pyc', '.pyo', '.lock', '.swp', '.swo'],
    'include_only_extensions': [], # 默認空列表，表示不限制包含，只按排除規則走
    'exclude_file_patterns': ["project_dump.txt", "*_dump_*.txt"], # 排除舊的或新的 dump 文件自身
    'max_lines_per_file': 300, # 文本文件內容截斷行數
    'output_directory': 'output',
    'output_format': 'markdown', # 默認為 markdown
    'project_name': '', # 默認為空，將從 project_root_path 推斷
    'output_filename_prefix': '', # 默認無前綴
    'add_timestamp_to_filename': True, # 默認添加時間戳
}

# ====================================================================
# 輔助函式：載入配置
# ====================================================================
def load_overview_config(config_file_path: str) -> Dict[str, Any]:
    """從 YAML 檔案載入配置，並與默認配置合併。"""
    config = DEFAULT_CONFIG.copy() # 從默認配置開始
    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                # 遍歷用戶配置並更新默認配置，處理嵌套列表
                for key, value in user_config.items():
                    if key in config and isinstance(config[key], list) and isinstance(value, list):
                        # 對於列表類型，如果用戶提供了，就直接覆蓋，不合併元素
                        config[key] = value
                    else:
                        config[key] = value
            print(f"✅ 載入配置檔案: '{config_file_path}'")
        except Exception as e:
            print(f"❌ 警告: 無法載入或解析配置檔案 '{config_file_path}': {e}。將使用默認或部分載入的配置。")
    return config

# ====================================================================
# 樹狀結構生成 (修改以支持不同格式)
# ====================================================================
def generate_tree_structure(root_dir: str, project_name: str, exclude_dirs: List[str], output_format: str) -> str:
    """生成專案的目錄樹結構，可選純文本或 Markdown 格式。"""
    tree_lines: List[str] = []
    
    # 根據格式調整縮進和連接符
    if output_format == 'markdown':
        # Markdown 列表使用 "- " 或 "* "，巢狀通過兩個空格縮進
        # 標題從 H2 開始，因為 H1 是整個檔案的標題
        tree_lines.append(f"## {project_name}/\n")
        indent_per_level = "  "
        list_prefix = "- "
    else: # text
        tree_lines.append(f"{project_name}/\n")
        indent_per_level = "    "
        list_prefix = "├── " # 頂層將特別處理

    def _generate_tree_recursive(current_dir: str, level: int = 0, parent_is_last: bool = False, current_text_prefix: str = "") -> None:
        try:
            items_in_dir = os.listdir(current_dir)
        except OSError as e:
            if output_format == 'markdown':
                tree_lines.append(f"{indent_per_level * level}{list_prefix}*無法訪問目錄*: `{current_dir}` - {e}\n")
            else:
                tree_lines.append(f"{current_text_prefix}├── <無法訪問目錄: {current_dir} - {e}>\n")
            return

        dirs_to_process: List[str] = []
        files_to_process: List[str] = []

        for item_name in sorted(items_in_dir):
            path = os.path.join(current_dir, item_name)
            
            # 在樹狀圖中，我們仍會顯示排除目錄的父級，但不會展開其內容。
            # 這裡的排除dirs主要是在遍歷時不進入這些目錄，但在樹狀圖中可以選擇性顯示。
            # 這裡簡單地跳過，保持與os.walk行為一致，不顯示其子項。
            if os.path.isdir(path) and item_name in exclude_dirs:
                # 如果是排除的目錄，就直接作為一個葉子節點列出，但不會再遞歸其內容
                if output_format == 'markdown':
                    tree_lines.append(f"{indent_per_level * level}{list_prefix}{item_name}/ (Excluded)\n")
                else:
                    tree_lines.append(f"{current_text_prefix}├── {item_name}/ (Excluded)\n")
                continue
                
            if os.path.isdir(path):
                dirs_to_process.append(item_name)
            elif os.path.isfile(path):
                files_to_process.append(item_name)

        all_entries_sorted = sorted(dirs_to_process) + sorted(files_to_process)

        for i, item_name in enumerate(all_entries_sorted):
            is_last_entry = (i == len(all_entries_sorted) - 1)
            item_path = os.path.join(current_dir, item_name)
            is_dir = os.path.isdir(item_path)

            if output_format == 'markdown':
                tree_lines.append(f"{indent_per_level * level}{list_prefix}{item_name}{'/' if is_dir else ''}\n")
                if is_dir:
                    _generate_tree_recursive(item_path, level + 1, is_last_entry) # Markdown 模式下，parent_is_last 影響不大
            else: # text format
                connector = "└── " if is_last_entry else "├── "
                # 計算當前行的前綴，根據父級是否是最後一個子項來決定畫 '|' 或 ' '
                new_text_prefix = current_text_prefix + ("    " if parent_is_last else "│   ")
                
                line = f"{current_text_prefix}{connector}{item_name}{'/' if is_dir else ''}"
                tree_lines.append(line)
                
                if is_dir:
                    _generate_tree_recursive(item_path, level + 1, is_last_entry, new_text_prefix)

    # 針對 text 格式，頂層調用需要特別處理 (無初始前綴)
    if output_format == 'markdown':
        _generate_tree_recursive(root_dir)
    else: # text
        _generate_tree_recursive(root_dir, level=0, parent_is_last=False, current_text_prefix="")

    return "".join(tree_lines)


# ====================================================================
# 程式碼彙整主函式
# ====================================================================
def generate_project_overview(config: Dict[str, Any]):
    """根據配置生成專案概覽檔案。"""

    project_root = config['project_root_path']
    output_dir = config['output_directory']
    output_format = config['output_format']
    project_name = config['project_name']
    
    # 輸出文件名的動態部分
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") if config['add_timestamp_to_filename'] else ""
    output_ext = "md" if output_format == "markdown" else "txt"
    
    # 構建輸出文件名
    filename_parts = []
    if config['output_filename_prefix']:
        filename_parts.append(config['output_filename_prefix'])
    filename_parts.append(project_name.replace(' ', '_').lower()) # 將專案名轉為小寫和下劃線
    filename_parts.append("overview")
    if timestamp_str:
        filename_parts.append(timestamp_str)
    
    output_filename_base = "_".join(filter(None, filename_parts)) + f".{output_ext}"
    full_output_path = os.path.join(output_dir, output_filename_base)

    if not os.path.isdir(project_root):
        print(f"❌ 錯誤：掃描根目錄 '{project_root}' 不存在。請檢查 'project_root_path' 配置。")
        return

    os.makedirs(output_dir, exist_ok=True) # 確保輸出目錄存在

    processed_files_count = 0
    errors_encountered: List[str] = []
    
    # 根據格式調整文本輸出樣式
    if output_format == 'markdown':
        section_title_level = "##" # 子節點標題級別
        file_title_level = "###" # 檔案標題級別
        header_separator = "\n---\n\n"
        code_block_start = lambda lang: f"```{lang}\n"
        code_block_end = "```\n"
        skipped_content_comment = lambda ext, filename, size: f"<!-- Content skipped, type '{ext}': {filename} ({size / 1024:.2f} KB) -->\n"
        read_error_comment = lambda filename, size: f"<!-- Content read failed: {filename} ({size / 1024:.2f} KB) -->\n"
        truncated_comment = lambda lines_skipped: f"<!-- ... content truncated - {lines_skipped} lines skipped ... -->\n"
    else: # text
        section_title_level = "=="
        file_title_level = "---"
        header_separator = "\n" + "=" * 80 + "\n\n"
        code_block_start = lambda lang: "" # 純文本模式無代碼塊
        code_block_end = ""
        skipped_content_comment = lambda ext, filename, size: f"[Content skipped for file type '{ext}': {filename} ({size / 1024:.2f} KB)]\n"
        read_error_comment = lambda filename, size: f"[Content read failed: {filename} ({size / 1024:.2f} KB)]\n"
        truncated_comment = lambda lines_skipped: f"[...truncated - {lines_skipped} lines skipped...]\n"


    try:
        with open(full_output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
            # 檔案總標題
            if output_format == 'markdown':
                outfile.write(f"# 專案程式碼概覽: {project_name}\n")
                outfile.write(f"*生成於: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            else:
                outfile.write(f"# 專案程式碼概覽: {project_name} (生成於: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                outfile.write("=" * 80 + "\n\n")

            # 目錄結構部分
            outfile.write(f"{section_title_level} 專案目錄結構\n")
            outfile.write(header_separator)
            
            tree_structure = generate_tree_structure(project_root, project_name, config['exclude_directories'], output_format)
            outfile.write(tree_structure)
            outfile.write(header_separator)

            # 各檔案內容部分
            outfile.write(f"{section_title_level} 各檔案內容\n")
            outfile.write(header_separator)
            
            # 使用 `exclude_directories` 進行 `os.walk` 的過濾
            for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
                # 過濾不應遍歷的目錄
                dirnames[:] = [d for d in dirnames if d not in config['exclude_directories']]

                for filename in sorted(filenames):
                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, project_root).replace(os.sep, '/')
                    
                    _, extension = os.path.splitext(file_path)
                    extension_lower = extension.lower()

                    # 檢查是否應排除此文件 (根據模式)
                    if any(fnmatch.fnmatch(filename, pattern) for pattern in config['exclude_file_patterns']):
                        print(f"跳過 (匹配排除模式): {relative_path}")
                        continue

                    # 檢查是否應只包含特定類型文件 (如果 include_only_extensions 不為空)
                    if config['include_only_extensions'] and extension_lower not in config['include_only_extensions']:
                        print(f"跳過 (不符合包含列表): {relative_path}")
                        continue

                    print(f"處理中: {relative_path}") # 【修改】控制台通知

                    outfile.write(f"{file_title_level} 檔案: `{relative_path}`\n\n") # 【修改】檔案標題格式
                    
                    if extension_lower in config['skip_content_extensions']:
                        file_size = os.path.getsize(file_path)
                        outfile.write(skipped_content_comment(extension_lower, filename, file_size))
                    else:
                        try:
                            file_content_lines: List[str] = []
                            lines_read = 0
                            lines_skipped_count = 0
                            
                            with open(file_path, 'r', encoding='utf-8', errors='strict') as infile:
                                for line in infile:
                                    if config['max_lines_per_file'] is not None and config['max_lines_per_file'] > 0 and lines_read >= config['max_lines_per_file']:
                                        lines_skipped_count += 1
                                    else:
                                        file_content_lines.append(line)
                                        lines_read += 1
                            
                            lang = LANGUAGE_MAP.get(extension_lower, 'text')
                            outfile.write(code_block_start(lang))
                            outfile.writelines(file_content_lines)
                            if lines_skipped_count > 0:
                                outfile.write(truncated_comment(lines_skipped_count))
                                print(f"  ⚠️ 已截斷內容 (跳過 {lines_skipped_count} 行): {relative_path}") # 【新增】截斷通知
                            outfile.write(code_block_end)

                        except (UnicodeDecodeError, IOError) as e:
                            file_size = os.path.getsize(file_path)
                            outfile.write(read_error_comment(filename, file_size))
                            errors_encountered.append(f"讀取錯誤: {relative_path} - {e}")
                            print(f"  ❌ 讀取錯誤: {relative_path} - {e}") # 【新增】錯誤通知
                        except Exception as e:
                            outfile.write(f"<!-- 未知錯誤: {filename} - {e} -->\n") # 輸出為 Markdown 註解
                            errors_encountered.append(f"未知錯誤: {relative_path} - {e}")
                            print(f"  ❌ 未知錯誤: {relative_path} - {e}") # 【新增】錯誤通知

                    outfile.write("\n") # 確保每個檔案內容塊後有空行
                    
                    processed_files_count += 1

        print("\n" + "=" * 80)
        print(f"✅ 成功！共處理了 {processed_files_count} 個檔案。")
        print(f"輸出結果已儲存至: {os.path.abspath(full_output_path)}")
        if errors_encountered:
            print("\n❌ 運行過程中發現以下錯誤:")
            for error in errors_encountered:
                print(f"  - {error}")
        print("=" * 80)

    except IOError as e:
        print(f"❌ 錯誤：無法寫入輸出檔案 '{full_output_path}'。 ({e})")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成專案程式碼概覽檔案。")
    parser.add_argument("--config", type=str, default="project_overview_config.yaml", # 【修改】默認配置文件名
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

    # 命令行參數覆蓋配置文件設定
    if args.output_dir:
        config['output_directory'] = args.output_dir
    if args.format:
        config['output_format'] = args.format
    if args.max_lines_per_file is not None:
        config['max_lines_per_file'] = args.max_lines_per_file
    if args.project_root:
        config['project_root_path'] = args.project_root
    if args.project_name:
        config['project_name'] = args.project_name
    if args.output_filename_prefix is not None: # 允許設置為空字串
        config['output_filename_prefix'] = args.output_filename_prefix
    if args.add_timestamp is not None:
        config['add_timestamp_to_filename'] = args.add_timestamp
    
    # 如果 project_root_path 為空，則設置為當前工作目錄的絕對路徑
    if not config['project_root_path']:
        config['project_root_path'] = os.path.abspath(os.getcwd())
    
    # 如果 project_name 為空，則從 project_root_path 推斷
    if not config['project_name']:
        config['project_name'] = os.path.basename(config['project_root_path'])

    # 執行生成
    generate_project_overview(config)