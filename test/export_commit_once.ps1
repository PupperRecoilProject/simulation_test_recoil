<#
只輸出「單一 commit」的內容到文字檔
- summary：提交摘要與檔案清單（--stat）
- focus：摘要與檔案清單，加上精簡 diff（-U N）
- full：摘要與完整 diff（-p）

相容 Windows PowerShell 5.1
輸出 UTF-8，避免中文亂碼
#>

[CmdletBinding()]
Param(
  [ValidateSet('summary','focus','full')]
  [string]$Mode = 'focus',

  # diff 上下文行數，僅在 focus 模式生效
  [int]$Context = 3,

  # 指定要輸出的 commit，預設 HEAD
  [string]$Commit = 'HEAD',

  # 文字輸出檔名
  [string]$Output = 'commit_export.txt',

  # 只看特定路徑或副檔名，例如 "src","*.py"
  [string[]]$Paths,

  # UTF-8 是否帶 BOM，預設不帶
  [switch]$Utf8Bom
)

try {
  $global:OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

# 先解析成完整 hash，順便驗證 commit 是否存在
$fullHash = & git rev-parse $Commit 2>$null
if ($LASTEXITCODE -ne 0 -or -not $fullHash) {
  Write-Error "找不到指定的 commit：$Commit"
  exit 1
}

# 組裝 git show 參數
$pretty = '---`ncommit %H`nAuthor: %an <%ae>`nDate: %ad`nSubject: %s`n'
$gitArgs = @('show', '--date=iso8601', '--no-color', "--pretty=format:$pretty")

switch ($Mode) {
  'summary' { $gitArgs += @('--stat') }
  'focus'   { $gitArgs += @('--stat', "-U$Context") }
  'full'    { $gitArgs += @('-p') }
}

$gitArgs += $fullHash

# 限定路徑放在最後
if ($Paths -and $Paths.Count -gt 0) {
  $gitArgs += '--'
  $gitArgs += $Paths
}

Write-Host "Running: git $($gitArgs -join ' ')" -ForegroundColor Cyan

if (Test-Path $Output) { Remove-Item $Output -Force }

# 串流寫入 UTF-8 檔案
& git @gitArgs | Out-File -FilePath $Output -Encoding UTF8

if ($LASTEXITCODE -ne 0) {
  Write-Warning "git 結束代碼 $LASTEXITCODE，請確認參數是否正確"
} else {
  Write-Host "匯出完成 → $Output" -ForegroundColor Green
}

if ($Mode -in @('summary','focus')) {
  Write-Host "說明：summary 與 focus 模式遇到二進制檔案會以 'Binary files differ' 標示" -ForegroundColor DarkGray
  Write-Host "若想看完整二進制差異，可改用 full 模式或另行加上格式轉文字流程" -ForegroundColor DarkGray
}

#怎麼用
#
#只輸出最新一筆，重點版（±3 行）
#
#.\test\export_commit_once.ps1 -Mode focus
#
#
#指定某筆 commit（貼短 hash 或完整 hash）
#      .\test\export_commit_once.ps1 -Mode focus -Commit 91d1d3b -Output one_commit.txt
#
#
#只要摘要與檔案清單（最精簡）
#
#.\test\export_commit_once.ps1 -Mode summary -Commit HEAD
#
#
#完整 diff（稽核用）
#
#.\test\export_commit_once.ps1 -Mode full -Commit HEAD
#
#
#只看某些路徑或副檔名
#
#.\test\export_commit_once.ps1 -Mode focus -Commit HEAD -Paths "src","*.xml"