# 自動續跑任務佇列 (Task Queue)

> 入口見 `README.md`。此檔給「閒置觸發的循環喚醒」(cron 56e79201) 逐一處理。
> Harrison 不在時，我一停下就自動取下一個未開始任務來做。

## 自主執行規則（Harrison 不在時遵守）
1. **取最上面一個「未開始 `[ ]`」任務**執行；做完改 `[x]` 並在該行補一句結果/commit。
2. **本機 commit、不要 push。** push 一律等 Harrison 回來確認。
3. **不改控制路徑程式碼、不做有風險的變更**（除非該任務明確授權，如 #3）。
4. **絕不卡死**：遇阻塞/需 Harrison 決策/環境跑不動 → 標 `[BLOCKED]` 並寫原因，**跳下一個**。
5. 發現的「將就/理所當然/不漂亮」設計 → 一律記進 `REFACTOR_SCOPE.md`（一定要有紀錄）。
6. 過今天(6/4) 06:00 後，循環喚醒會自刪停止。
7. 每輪結束留一行進度。先做「主任務」，全部完成再做「備選池」。

## 主任務（依序）
- [ ] **T1 建 sim 環境**：用 `C:\Users\Harrison\miniconda3\Scripts\conda.exe` 建 env `pupper-sim`(Python 3.11)，pip 裝 sim 依賴(numpy onnxruntime PyYAML nicegui pygame pyserial Pillow scipy mujoco glfw)，做 import smoke test。更新 environment 記憶。卡住→BLOCKED。
- [ ] **T2 sim 啟動 smoke test**：能跑就啟動 `main_nicegui.py`(sim 模式,含超時保護)，記錄 error/warning →`reports/SMOKE_2026-06-04.md`。無 env→BLOCKED。
- [ ] **T3 退出硬體卡死 bug（授權碰控制碼，僅限分支）**：在分支 `fix/hardware-stop-selfjoin` 用 `src/mock` 寫重現 `_execute_stop` self-join 的特徵測試＋最小修復；有 env 就跑測試驗證。**不併 main、不 push**。無 env→寫好但標「未驗證」。
- [ ] **T4 Feature inventory**：通讀 sim，整理現有所有功能(模式/按鍵/UI/硬體)→`docs/FEATURE_INVENTORY.md`。
- [ ] **T5 「將就/理所當然」設計 sweep**：系統掃 sim，每條可疑設計(檔:行+為何+建議)記進 `REFACTOR_SCOPE.md`。
- [ ] **T6 時間/單一時鐘 專題**：靜態分析 sim/hw 時間機制→`reports/REVIEW_timing_2026-06-04.md`。
- [ ] **T7 防死鎖/併發 專題**：審查 Lock/Event/Queue/事件路徑列死鎖+競態→`reports/REVIEW_concurrency_2026-06-04.md`。
- [ ] **T8 效能/記憶體洩漏 靜態獵查**：找無上限成長的緩衝/狀態解釋「久跑超卡」→reports/REFACTOR_SCOPE。
- [ ] **T9 重構架構提案 + 測試 SOP 草稿**：headless 核心 API 草案、模組邊界、strangler-fig 遷移；測試 SOP + 介入 SOP →`docs/` 草稿。

## 備選池（主任務做完才做；同樣依序、同規則）
- [ ] **B1 術語正規化文件**：把軸向慣例、roll/pitch、關節命名(hip/abduction…)的權威名稱定死→`docs/GLOSSARY.md`，消除三方命名混淆。
- [ ] **B2 git 歷史考古：找被改壞的功能**：掃 sim git log 找「設計過又壞掉/被 revert」的功能線索→reports。純讀。
- [ ] **B3 死碼/未使用偵測（只報告、不刪）**：找未使用函式/孤兒檔(如舊 observation.py)→REFACTOR_SCOPE。**不自動刪除**。
- [ ] **B4 config.yaml 稽核**：逐鍵記錄用途、標出過時/未用鍵(如註解掉的 recipes、warmup_duration)→reports/REFACTOR_SCOPE。
- [ ] **B5 依賴稽核**：實際 import vs readme pip 清單，找缺漏/未用、建議版本釘選→reports。
- [ ] **B6 pytest 基礎架構**：建 pytest 設定/conftest/test 目錄/一個 trivial 測試(僅 infra，不大量寫測試)。附加性、安全。
- [ ] **B7 韌體 pupper_recoil 盤點+sweep**：指令集 inventory + smell sweep→reports(放 sim docs)。純讀。
- [ ] **B8 工作區根 README**：根目錄寫一份 README 說明三 repo + quickstart(目前根只有 CLAUDE.md+workspace)。

## 已完成 / 已封存
（完成的任務移到這裡保留紀錄）
