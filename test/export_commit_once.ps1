<#
================================================================================
 Git Commit Exporter - 單一 Commit 匯出工具
================================================================================

[功能]
  將指定的單一 Git commit 內容匯出成一個文字檔，方便審閱、記錄或分享。

[主要參數說明]
  * -Mode       匯出模式，決定內容的詳細程度。
                - 'summary' : 只有摘要和檔案清單 (最精簡)。
                - 'focus'   : 摘要 + 檔案清單 + 變更處前後幾行 (預設)。
                - 'full'    : 摘要 + 完整的檔案差異 (最詳細)。

  * -Commit     要匯出的 commit，可以是短 hash、長 hash 或 'HEAD'。(預設: 'HEAD')
  * -Output     輸出的「檔名」。(預設: 'commit_export.txt')
  * -OutputDir  輸出的「資料夾」。(預設: './output')
  
  * 進階：可使用 -Paths "src", "*.cs" 篩選路徑；用 -Context 5 調整 focus 模式的行數。

--------------------------------------------------------------------------------
--- 快速上手 (複製 & 修改以下指令即可) ---
--------------------------------------------------------------------------------

# 範例 1：最常用情境 (匯出最新一筆 commit)
# 說明：直接執行，會將最新 commit (HEAD) 的重點內容匯出。
# 結果 → 檔案會存在 '.\test\output\commit_export.txt'

.\test\export_commit_once.ps1


# 範例 2：自訂匯出 (指定 commit 和檔名)
# 說明：修改 -Commit 和 -Output 的值即可。
# 結果 → 檔案會存在 '.\test\output\my-feature.txt'

.\test\export_commit_once.ps1 -Commit 91d1d3b -Output "my-feature.txt"


================================================================================
#>

[CmdletBinding()]
Param(
  [ValidateSet('summary','focus','full')]
  [string]$Mode = 'focus',

  # diff 上下文行數，僅在 focus 模式生效
  [int]$Context = 3,

  # 指定要輸出的 commit，預設 HEAD
  [string]$Commit = 'HEAD',

  # 【修改】新增輸出資料夾參數，預設為 'output'
  [string]$OutputDir = 'output',

  # 【修改】註解更新：現在這只代表「檔名」
  [string]$Output = 'commit_export.txt',

  # 只看特定路徑或副檔名，例如 "src","*.py"
  [string[]]$Paths,

  # UTF-8 是否帶 BOM，預設不帶
  [switch]$Utf8Bom
)

# --- 腳本執行區 ---

# 步驟 1：設定環境與輸出路徑
#------------------------------------------------
try {
  # 確保 PowerShell 輸出為 UTF-8，避免中文亂碼
  $global:OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

# 組合完整的檔案路徑，並確保資料夾存在
if (-not (Test-Path -Path $OutputDir -PathType Container)) {
  Write-Host "建立輸出資料夾：$OutputDir" -ForegroundColor Yellow
  $null = New-Item -Path $OutputDir -ItemType Directory -Force
}
$fullOutputPath = Join-Path -Path $OutputDir -ChildPath $Output


# 步驟 2：驗證 commit 並準備 git 指令
#------------------------------------------------
# 解析 commit hash，同時也確認此 commit 是否真的存在
$fullHash = & git rev-parse $Commit 2>$null
if ($LASTEXITCODE -ne 0 -or -not $fullHash) {
  Write-Error "找不到指定的 commit：$Commit"
  exit 1
}

# 根據參數，組裝要執行的 git show 指令
$prettyFormat = '---`ncommit %H`nAuthor: %an <%ae>`nDate: %ad`nSubject: %s`n'
$gitArgs = @('show', '--date=iso8601', '--no-color', "--pretty=format:$prettyFormat")

switch ($Mode) {
  'summary' { $gitArgs += '--stat' }
  'focus'   { $gitArgs += '--stat', "-U$Context" }
  'full'    { $gitArgs += '-p' }
}

$gitArgs += $fullHash

# 如果有指定路徑，加到指令的最後面
if ($Paths -and $Paths.Count -gt 0) {
  $gitArgs += '--'
  $gitArgs += $Paths
}


# 步驟 3：執行 git 並將結果寫入檔案
#------------------------------------------------
Write-Host "正在執行: git $($gitArgs -join ' ')" -ForegroundColor Cyan

# 如果檔案已存在，先刪除
if (Test-Path $fullOutputPath) {
  Remove-Item $fullOutputPath -Force
}

# 執行 git 指令，並透過管線將輸出直接寫入檔案，編碼為 UTF-8
& git @gitArgs | Out-File -FilePath $fullOutputPath -Encoding UTF8

if ($LASTEXITCODE -ne 0) {
  Write-Warning "git 程序返回錯誤碼 $LASTEXITCODE，請檢查參數。"
} else {
  Write-Host "匯出成功 → $fullOutputPath" -ForegroundColor Green
}