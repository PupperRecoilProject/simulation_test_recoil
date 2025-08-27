<#
導出 Git 提交內容到單一文字檔，支援三種模式
- summary：提交摘要與檔案清單（--stat）
- focus：摘要與檔案清單，加上精簡 diff（-U N）
- full：摘要與完整 diff（-p）

額外功能
- 強制 UTF-8 輸出，避免中文日文韓文亂碼
- 常用過濾條件：作者 author、時間 since until、關鍵字 grep、排除合併 no-merges、指定路徑 path
- 自訂輸出檔名與 diff context 範圍

使用範例
  .\test\export_commits.ps1
  .\test\export_commits.ps1 -Mode summary -Since "2025-08-01" -Until "2025-08-27"
  .\test\export_commits.ps1 -Mode focus -Context 5 -Author "Weiyu"
  .\test\export_commits.ps1 -Mode full -Grep "policy","param_keys"
  .\test\export_commits.ps1 -Paths "src","configs\robot.yaml"
#>

[CmdletBinding()]
Param(
  [ValidateSet('summary','focus','full')]
  [string]$Mode = 'focus',

  # diff 內容的上下文行數，僅在 focus 模式生效
  [int]$Context = 3,

  # 文字輸出檔名
  [string]$Output = 'commits_export.txt',

  # 只看某位作者
  [string]$Author,

  # 只看某段時間
  [string]$Since,
  [string]$Until,

  # 關鍵字過濾（對 subject 與 body）
  [string[]]$Grep,

  # 排除合併提交
  [switch]$NoMerges,

  # 只看特定路徑或副檔名
  [string[]]$Paths,

  # 以 UTF-8 BOM 輸出，預設無 BOM
  [switch]$Utf8Bom
)

try {
  $global:OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

# 組裝 git log 參數（改用 $gitArgs，避免觸發自動變數 $args）
$pretty = '---`ncommit %H`nAuthor: %an <%ae>`nDate: %ad`nSubject: %s`n'
$gitArgs = @('log', '--date=iso8601', '--reverse', '--no-color', "--pretty=format:$pretty")

switch ($Mode) {
  'summary' { $gitArgs += @('--stat') }
  'focus'   { $gitArgs += @('--stat', "-U$Context") }
  'full'    { $gitArgs += @('-p') }
}

if ($Author) { $gitArgs += "--author=$Author" }
if ($Since)  { $gitArgs += "--since=$Since" }
if ($Until)  { $gitArgs += "--until=$Until" }

if ($NoMerges) { $gitArgs += '--no-merges' }

if ($Grep -and $Grep.Count -gt 0) {
  foreach ($g in $Grep) { $gitArgs += @('--grep', $g) }
  $gitArgs += '--all-match'
}

if ($Paths -and $Paths.Count -gt 0) {
  $gitArgs += '--'
  $gitArgs += $Paths
}

# 顯示即將執行的 git 命令
Write-Host "Running: git $($gitArgs -join ' ')" -ForegroundColor Cyan

# 若輸出檔存在先刪除
if (Test-Path $Output) { Remove-Item $Output -Force }

# 在 PowerShell 5.1 用管線串流寫檔，避免 .NET 5+ 的 ArgumentList 屬性相容性問題
& git @gitArgs | Out-File -FilePath $Output -Encoding UTF8

if ($LASTEXITCODE -ne 0) {
  Write-Warning "git 結束代碼 $LASTEXITCODE，請確認是否在 git repository 內或參數是否正確"
} else {
  Write-Host "匯出完成 → $Output" -ForegroundColor Green
}

if ($Mode -in @('summary','focus')) {
  Write-Host "說明：summary 與 focus 模式遇到二進制檔案會以 'Binary files differ' 標示" -ForegroundColor DarkGray
  Write-Host "若想檢視完整二進制差異內容，請改用 full 模式或加入專用轉換流程" -ForegroundColor DarkGray
}
