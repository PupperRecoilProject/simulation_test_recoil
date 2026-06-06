# Quadruped Robot — 重啟計畫（跨 repo 協調）

> 本文件雖放在 `simulation_test_recoil`（高階端，當基地），但協調的是**整個工作區的兩個 repo**：
> - `simulation_test_recoil/` — 高階控制軟體 (Python / NiceGUI / MuJoCo / ONNX)
> - `pupper_recoil/`（`../../pupper_recoil`）— Teensy 4.0 即時控制韌體 (C++ / PlatformIO)
> - （訓練 repo 暫未下載，需要時再拉）
>
> 重啟基準 tag（兩 repo 皆有）：`restart-baseline-2026-06-03`
> 隨時可 `git checkout restart-baseline-2026-06-03` 回到重啟當天狀態。

## 工作流程
- `main` = 永遠可動的主線。
- **分支策略**：大 / 高風險 / 可能放棄的探索 → 開 `feat/xxx`、`fix/xxx` 分支做完再併回；
  小 / 安全（文件、config 微調）→ 直接在 main 上做。（單人開發，不為小改動付分支儀式成本）
- push 到 GitHub 前先確認。
- 每完成一項，在下表打勾。

## Phase 0 — 地基（純結構 / git，不碰邏輯）
- [x] 0.1 兩個 repo 打 `restart-baseline-2026-06-03` tag
- [x] 0.2 sim：dev4.13 併入 main，main 重回主線 (v4.14.3)
- [x] 0.3 建 `quadruped-robot.code-workspace`（雙 repo 一個視窗）
- [x] 0.4 建本計畫表與工作流程
- [x] 0.5 push 兩個 repo 的 main + tag 到 GitHub
- [x] 0.6 PROJECT_PLAN 移入 sim/docs/ 進版控

## Phase 1 — 跨 repo 介面契約文件
> 成果：`INTERFACE_CONTRACT.md`（同目錄）
- [x] 1.1 34 欄位資料契約（Teensy → PC：欄位 / 單位 / 座標系）
- [x] 1.2 指令協定（PC → Teensy：`move all`、`monitor freq/p/h`、回應規則）
- [x] 1.3 馬達編號 + 方向（編號兩端一致、命名差異、`correction_vector` 補償邏輯）
- [x] 1.4 連線參數（baud 921600）+ 已知疑點清單

## Phase 2+ — 之後再排
- [x] **roll/pitch 疑點調查** — 已用訓練端原始碼釐清：訓練 `get_pitch`=`-arcsin(up_vector[1])`(側向=roll)，韌體送 roll **一致**；歧異在 sim **模擬模式** `_get_current_pitch` 用了真 pitch 公式。詳見 INTERFACE_CONTRACT §4-1。
- [ ] **roll/pitch 後續**：(a) 實體機台確認 Teensy AHRS roll 正負號/軸向；(b) 修 sim 模擬模式 pitch 公式使與訓練一致；(c) 三方正名 pitch→roll。⚠️ (a) 需實體機器狗
- [ ] **角速度正負號驗證**（座標系已確認一致；剩軸對應/符號）— ⚠️ 需實體機器狗在場
- [ ] **（評估）訓練流程是否重做** — 見下節
- [x] 過時文件修正 — ✅ 已做(本機 commit 未 push)：fw README/platformio baud→921600、sim readme 移除 main.py/tennsy.md 過時引用
- [x] **`_execute_stop` self-join** — ✅ 釐清：現行碼(v4.14.0)已移除，舊報告過時；改加回歸測試鎖死。見 reports/FIX_hardware_handoff_2026-06-04.md(分支)
- [x] hardware 交接 try/finally 保證 resume_control（F2/T3b）— ✅ 已修於分支 `fix/hardware-stop-selfjoin`，待 review
- [x] 建立機器狗 Python 環境（T1）— ✅ env `pupper-sim` (py3.11) 已建；訓練 env 仍待建
- [~] **重構：headless 控制核心 + 薄客戶端** — 提案草稿完成(T9, `ARCHITECTURE_PROPOSAL.md`)；尚未動工
- [x] 建立測試 SOP（T9）— ✅ 草稿 `TEST_SOP.md`
- [x] feature inventory（T4）— ✅ `FEATURE_INVENTORY.md`

## 夜間自動任務成果（第一批，2026-06-04 凌晨，本機 commit 未 push）
- `reports/INVESTIGATION_2026-06-04.md`：recoil 是 +Y 側向力（解釋 roll）、command_4d=目標側傾(非 pitch)、訓練時 vx/vy/omega 恆為零（recoil 模型是站立吸力模型）、e2e_fixed 為子元重匯的不同權重、teensy 輸出問題線索。
- `reports/REVIEW_hardware_stability_2026-06-04.md`：靜態審查，F1 self-join bug（🔴）、F2 resume_control 洩漏風險。
- `ARCHITECTURE.md`：sim 架構 onboarding。

## 自動續跑成果（第二批，2026-06-04 03:11–06:00，循環喚醒逐項處理，本機 commit 未 push）
> 完整逐項紀錄見 `TASK_QUEUE.md`（各項已打勾 + 一句結果）。主任務 T1–T9 + T3b 全完成；備選 B1–B2 完成。
- **T1** env `pupper-sim`；**T2** sim 啟動 smoke（揪 cp950 編碼 bug C-S1）。
- **T3/T3b**（分支 `fix/hardware-stop-selfjoin`，唯一控制碼改動，**待 review/push**）：序列埠交接洩漏修復 + 4 回歸測試；釐清 F1 已修。
- **T4** `FEATURE_INVENTORY.md`；**T5** 將就設計 sweep(REFACTOR C2，含雙鍵盤系統)。
- **T6** `REVIEW_timing`(證實非單一時鐘；`_update_recoil_warning_timer` 疑無呼叫者)。
- **T7** `REVIEW_concurrency`(非重入鎖地雷)；**T8** `REVIEW_performance`(terrain_cache 洩漏=久跑超卡主嫌)。
- **T9** `ARCHITECTURE_PROPOSAL.md` + `TEST_SOP.md`。
- **B1** `GLOSSARY.md`；**B2** `INVESTIGATION_git_archaeology`(control_mode 是放棄的列舉遷移殘存物)。
- 循環喚醒 cron 已於 06:00 後自刪。**剩 B3–B8 未做**。
- 早上待辦：① review 分支 `fix/hardware-stop-selfjoin` 決定併/push；② 確認 `_update_recoil_warning_timer` 是否真未被呼叫(OPEN_THREADS A8)。

## 訓練流程評估（mujoco_playground_recoil，待決策）
現況：MuJoCo Playground（DeepMind/brax PPO）的 **fork**，pupper 環境改自 Go1。
- 觀察到的問題：notebook 為訓練入口（不易重現/版控）、多份「複製」檔與多個 joystick 變體
  （joystick / joystickwithgun / joysticks_sac / *-複製）、sensor/常數沿用 Go1 可能挾帶假設、已與上游 playground 分岔。
- 選項：A 保留並整理 fork（收斂成單一 env + 腳本化 + 文件化）｜B 重新基於最新上游 playground 乾淨移植 recoil env｜C 換框架（最大工程）。
- 建議：列為**戰略項**，待 make-it-right 整理告一段落再決；短期先靠本契約把 obs 定義鎖死即可。
- [ ] 過時文件修正（README baud 115200、platformio monitor_speed 460800、sim readme 指向已刪除的 tennsy.md 破連結）
- [ ] 版本號註解雜訊清理（`【v4.x.x 修改】`）— 改靠 git 歷史，不再行內標版本
- [ ] 硬體進出穩定性（「teensy 輸出問題」線頭）
- [ ] （待決）建 `CLAUDE.md`：收錄篩選後仍適用的開發/協作原則（見下）

## 開發原則（從舊手冊 v28.1 篩選後保留）
> 舊手冊是為「對話式、無版控、無記憶」的舊工作流寫的。git + 本計畫 + 記憶檔已取代多數手動儀式。
> **保留**（思維與架構）：
> - Make it work → right → fast（現處 "make it right"）
> - 基於事實的零信任除錯（證據鏈、對照實驗、追溯端到端數據流）
> - 多方案 + 說明 Why + 優劣分析
> - 三層 SSoT / 單向數據流 / 狀態驅動控制（準確描述現有架構）
> - 日誌等級語義（debug/info/warning/error，程式碼已遵循）
>
> **丟棄**（舊工作流補洞用，現已多餘）：
> - 行內 `【vX.X.X 修改】` 版本註解 → 改用 git commit / blame
> - 對話用 `[保留][新增][修改]` diff 標籤 → Claude Code 直接顯示 diff
> - 手冊版本綁對話編號的儀式 → 由 git + 記憶檔取代

## 已知疑點摘要（詳見 INTERFACE_CONTRACT.md §4）
1. **roll/pitch**：韌體第 9 欄送 roll 卻標 pitch；且 sim 模式用真 pitch、硬體模式用 roll，recoil 模型兩邊看到不同物理量。權威定義在訓練 repo 的 `joystick.py`。
2. **馬達方向**：韌體不翻轉，補償全在 sim `correction_vector`；手動 `move` 方向與 AI 相反。
3. **baud 不一致**：實際 921600，README/platformio.ini 記載過時。
