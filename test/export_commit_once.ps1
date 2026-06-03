<#
================================================================================
 Git Commit Exporter - 單一 Commit 匯出工具
================================================================================

[功能]
  將指定的單一 Git commit (包含普通及合併提交) 內容匯出成文字檔。

[預設行為]
  - 模式 (Mode)      : 'focus' (摘要 + 檔案清單 + 變更處上下文 3 行)
  - 目標 (Commit)    : 'HEAD' (最新一筆 commit)
  - 輸出檔           : './output/commit_export.txt'
  - 合併提交 (Merge) : 自動採用 'combined' 模式顯示差異。

[主要參數說明]
  * -Commit       : 指定要匯出的 commit。可以是 hash、tag、分支名 (例如 'HEAD')。
  * -Mode         : 設定匯出的詳細程度。
                    - 'summary' (摘要與檔案清單，最精簡)
                    - 'focus'   (摘要、檔案清單、變更處上下文的 diff，推薦)
                    - 'full'    (摘要與完整的 diff 內容，最詳細)
  * -Output       : 輸出的檔案名稱。(預設: 'commit_export.txt')
  * -OutputDir    : 輸出的資料夾路徑。(預設: './output')

[進階參數說明]
  * -Context      : 在 'focus' 模式下，設定 diff 顯示的上下文行數。預設 3 行。
  * -Paths        : 篩選只看特定路徑或副檔名的變更，可指定多個。例如: "src", "*.py"
  * -MergeDiffMode: **僅當目標是合併提交時生效**。
                    - 'combined' (預設): 顯示相對於所有父級的組合差異。
                    - 'split'           : 顯示與每個父級單獨的差異。(會有多份 diff)
  * -Utf8Bom      : 若指定此參數，輸出的 UTF-8 檔案將包含 BOM。預設不包含。

--------------------------------------------------------------------------------
--- 快速上手 (複製 & 修改以下指令即可) ---
--------------------------------------------------------------------------------

# 範例 1：最常用情境 (匯出最新一筆 commit)
# 說明：直接執行，會將最新 commit (HEAD) 的重點內容匯出到預設位置。
#       不論是普通 commit 或合併 commit，腳本都會自動處理。
# 結果 → 檔案會存在 '.\test\output\commit_export.txt'

.\test\export_commit_once.ps1


# 範例 2：自訂匯出 (指定 commit、檔名，並看完整 diff)
# 說明：修改 -Commit 和 -Output 的值即可。
# 結果 → 檔案會存在 '.\test\output\my-feature-full-diff.txt'

.\test\export_commit_once.ps1 -Commit 91d1d3b -Mode full -Output "my-feature-full-diff.txt"

================================================================================
#>

[CmdletBinding()]
Param(
  [ValidateSet('summary','focus','full')]
  [string]$Mode = 'focus',

  [ValidateSet('combined', 'split')]
  [string]$MergeDiffMode,

  [int]$Context = 3,
  [string]$Commit = 'HEAD',
  [string]$OutputDir = 'output',
  [string]$Output = 'commit_export.txt',
  [string[]]$Paths,
  [switch]$Utf8Bom
)

# --- 腳本執行區 ---

# 步驟 1：設定環境與輸出路徑
#------------------------------------------------
try {
  # 確保 PowerShell 輸出為 UTF-8 (無 BOM)，避免中文亂碼
  # 新增了一個 `[System.Text.UTF8Encoding]::new($Utf8Bom)` 來根據參數控制 BOM
  $global:OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($Utf8Bom.IsPresent)
} catch { }

if (-not (Test-Path -Path $OutputDir -PathType Container)) {
  Write-Host "建立輸出資料夾：$OutputDir" -ForegroundColor Yellow
  $null = New-Item -Path $OutputDir -ItemType Directory -Force
}
$fullOutputPath = Join-Path -Path $OutputDir -ChildPath $Output

# 步驟 2：驗證 commit 並準備 git 指令
#------------------------------------------------
$fullHash = & git rev-parse $Commit 2>$null
if ($LASTEXITCODE -ne 0 -or -not $fullHash) {
  Write-Error "找不到指定的 commit：$Commit"
  exit 1
}

$prettyFormat = '---`ncommit %H`nAuthor: %an <%ae>`nDate: %ad`nSubject: %s`n'
# 為了確保日期格式一致，這裡使用 --date=iso8601-strict
$gitArgs = @('show', '--date=iso8601-strict', '--no-color', "--pretty=format:$prettyFormat")

switch ($Mode) {
  'summary' { $gitArgs += '--stat' }
  'focus'   { $gitArgs += '--stat', "-U$Context" }
  'full'    { $gitArgs += '-p' }
}

# 【智慧型合併提交處理】
# 檢查 commit 的 parent 數量，如果大於 1，就是合併提交
# 改用 git log --pretty=%P 獲取父級 hash 更簡潔可靠
$parentHashes = (git log -n 1 --pretty="%P" $fullHash).Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
if ($parentHashes.Length -gt 1) { # 如果父級數量大於 1，就是合併提交
  Write-Host "偵測到這是一個合併提交 (Merge Commit)。" -ForegroundColor Yellow

  # 決定要使用的模式：優先使用使用者指定的，否則使用預設值 'combined'
  $effectiveMergeMode = if ($PSBoundParameters.ContainsKey('MergeDiffMode')) { $MergeDiffMode } else { 'combined' }

  if (-not $PSBoundParameters.ContainsKey('MergeDiffMode')) {
      Write-Host "自動採用預設的 'combined' 模式顯示合併差異。 (可使用 -MergeDiffMode 'split' 更改)" -ForegroundColor Cyan
  }

  switch ($effectiveMergeMode) {
    'combined' { $gitArgs += '-c' }
    'split'    { $gitArgs += '-m' }
  }
}

$gitArgs += $fullHash

if ($Paths -and $Paths.Count -gt 0) {
  $gitArgs += '--'; $gitArgs += $Paths
}

# 步驟 3：執行 git 並將結果寫入檔案
#------------------------------------------------
Write-Host "正在執行: git $($gitArgs -join ' ')" -ForegroundColor Cyan

if (Test-Path $fullOutputPath) { Remove-Item $fullOutputPath -Force }

# 【修正】明確指定 -NoNewline:$false 來保留換行符
& git @gitArgs | Out-File -FilePath $fullOutputPath -Encoding UTF8 -NoNewline:$false

if ($LASTEXITCODE -ne 0) {
  Write-Warning "git 程序返回錯誤碼 $LASTEXITCODE，請檢查參數。"
} else {
  Write-Host "匯出成功 → $fullOutputPath" -ForegroundColor Green
}