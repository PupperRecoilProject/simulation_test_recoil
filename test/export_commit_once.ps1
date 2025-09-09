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

  # 【修改】新增輸出資料夾參數，預設為 'output'
  [string]$OutputDir = 'output',

  # 【修改】註解更新：現在這只代表「檔名」
  [string]$Output = 'commit_export.txt',

  # 只看特定路徑或副檔名，例如 "src","*.py"
  [string[]]$Paths,

  # UTF-8 是否帶 BOM，預設不帶
  [switch]$Utf8Bom
)

try {
  $global:OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

# 【新增】組合完整輸出路徑，並確保資料夾存在
# 如果輸出資料夾不存在，就建立它
if (-not (Test-Path $OutputDir -PathType Container)) {
  Write-Host "建立輸出資料夾：$OutputDir" -ForegroundColor Yellow
  $null = New-Item -Path $OutputDir -ItemType Directory -Force
}
# 組合完整的檔案路徑
$fullOutputPath = Join-Path -Path $OutputDir -ChildPath $Output


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

# 【修改】使用新的完整路徑變數 $fullOutputPath
if (Test-Path $fullOutputPath) { Remove-Item $fullOutputPath -Force }

# 【修改】串流寫入 UTF-8 檔案到指定路徑
& git @gitArgs | Out-File -FilePath $fullOutputPath -Encoding UTF8

if ($LASTEXITCODE -ne 0) {
  Write-Warning "git 結束代碼 $LASTEXITCODE，請確認參數是否正確"
} else {
  # 【修改】顯示完整的輸出路徑
  Write-Host "匯出完成 → $fullOutputPath" -ForegroundColor Green
}

if ($Mode -in @('summary','focus')) {
  Write-Host "說明：summary 與 focus 模式遇到二進制檔案會以 'Binary files differ' 標示" -ForegroundColor DarkGray
  Write-Host "若想看完整二進制差異，可改用 full 模式或另行加上格式轉文字流程" -ForegroundColor DarkGray
}