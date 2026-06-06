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
- [x] **T1 建 sim 環境**：✅ 完成。建好 env `pupper-sim`(python 3.11.15)，裝 numpy2.4.6/onnxruntime1.26/PyYAML6.0.3/nicegui3.12.1/pygame2.6.1/pyserial3.5/Pillow12.2/mujoco3.9/glfw2.10/pytest9.0，import smoke test 全過。**發現**：(a) README 清單的 `scipy` 全 repo 無 import→已剔除；(b) `tree_demo_server.py` 需 aiohttp/cv2/nanoowl 屬無關孤兒檔→未裝(記 B3)；(c) numpy 是 2.x，舊碼若按 1.x 寫恐有 breaking→T2 留意。已更新 environment 記憶。
- [x] **T2 sim 啟動 smoke test**：✅ 完成。sim 可完整啟動到 NiceGUI server(localhost:8080)，9 個 ONNX 全載入+預熱成功，numpy 2.x 在啟動+warmup 路徑無 breakage。**發現**：🔴S1 cp950 emoji print 編碼 bug(非 UTF-8 終端起不來，記 REFACTOR C-S1)；⚠️S2 NanoOwl 子程序硬編主入口+cv2 缺依賴(C-S2)；ℹ️S3 ORT 警告噪音。詳見 `reports/SMOKE_2026-06-04.md`。自動跑 sim 前須設 `PYTHONUTF8=1`。
- [x] **T3 退出硬體卡死 bug** ✅（程式碼在分支 `fix/hardware-stop-selfjoin`，未併 main、未 push）：**零信任修正**——F1（`_execute_stop` self-join）核對現行碼**已於 v4.14.0 移除**，舊報告過時。未動已正確的 `_execute_stop`，改加回歸測試鎖死「不 self-join」。詳見分支 commit `e9fbf02` 與 `reports/FIX_hardware_handoff_2026-06-04.md`（此報告檔在分支上）。
- [x] **T3b 修 F2 序列埠交接洩漏** ✅（同分支）：真兇＝交接洩漏（最吻合 Harrison 症狀）。改 `_execute_start` 用 `started_ok`+try/finally 保證任何例外都歸還控制權；改 `SerialCommunicator.close()` 暫停時也關埠。新增 `test/test_hardware_handoff.py` 4 測試全綠。**待 Harrison review 後再決定併 main/push**。順帶發現 `test_joystick.py` import 期 `exit()` 炸全套收集(記 REFACTOR D-1，此條也在分支)。
- [x] **T4 Feature inventory** ✅：通讀 sim 完成 → `docs/FEATURE_INVENTORY.md`（6 執行模式/控制模式、完整鍵盤+Xbox 綁定、NiceGUI 全面板、模擬/地形/懸浮/重置、PolicyManager 多模型混合、硬體狀態機+序列埠指令、架構基礎設施）。已加入 README 文件清單。
- [x] **T5 「將就/理所當然」設計 sweep** ✅：系統掃 sim → REFACTOR_SCOPE「C2. T5 sweep」8 條。重點：🔴C2-1 雙鍵盤系統(GLFW+NiceGUI 重疊綁定)、🟡C2-2 xbox 直接改 state(違反單向流)、C2-3 執行期改 config、C2-4 寬鬆 except 吞錯、C2-5 64 處 print(C-S1 根因)、⚪C2-6 274 處版本註解、C2-7 全域 nanoowl_process、C2-8 IMU 座標 TODO。
- [x] **T6 時間/單一時鐘 專題** ✅ → `reports/REVIEW_timing_2026-06-04.md`。核心：🔴**sim 非單一時鐘**(物理走模擬時間、AI 決策綁牆鐘，負載高時節奏漂移)；🔴**`_update_recoil_warning_timer` 疑似無呼叫者**(後座力倒數可能沒在跑→OPEN_THREADS A8)；MAX_FRAME_TIME 魔術數字重複、sleep(0.001) vs Windows 計時器解析度、control_freq 一值兼三職。將就項記 REFACTOR C3。
- [x] **T7 防死鎖/併發 專題** ✅ → `reports/REVIEW_concurrency_2026-06-04.md`。核心：🔴**state.lock 是非重入 Lock + helper「假定持鎖不自鎖」約定**(死鎖/競態地雷，建議改 RLock)；🟡 持鎖跨重活(hard_reset/mode_change)致 UI 凍結、事件回呼同步阻塞發布者執行緒(裝置連接 sleep)、recoil_warning_active/ai_step_times 鎖不一致。將就項記 REFACTOR C4。未見典型 publish-within-lock 死鎖(建議立 lint)。
- [x] **T8 效能/記憶體洩漏 靜態獵查** ✅ → `reports/REVIEW_performance_2026-06-04.md` + REFACTOR C5。核心：🔴**INFINITE 地形 `terrain_cache` 只增不刪**(持久性設計、無 eviction，走越久記憶體越大→「久跑超卡」最強嫌疑)；🟡 高頻物理步持鎖+大量 copy(GC/鎖競爭)、UI 10Hz 持鎖、data_capture_buffer 無硬上限。已確認 log/freq/obs deque、渲染上下文、事件訂閱皆有界良好(非洩漏)。附實測驗證建議(印 cache 長度+RSS)。
- [x] **T9 重構架構提案 + 測試 SOP 草稿** ✅ → `docs/ARCHITECTURE_PROPOSAL.md`(headless 核心+薄客戶端分層、模組邊界對照表[把每個問題編號對應到修掉它的模組]、strangler-fig 6 步遷移) + `docs/TEST_SOP.md`(測試金字塔、Layer A/B/C、需 Harrison 介入步驟、每次改動最小回歸清單)。已加入 README 清單。**主任務 T1–T9 全數完成。**

## 備選池（主任務做完才做；同樣依序、同規則）
- [x] **B1 術語正規化文件** ✅ → `docs/GLOSSARY.md`：關節命名(hip 兩邊不同義→用軸序/ID)、馬達編號(三方一致)+方向(只翻 sim)、roll/pitch(韌體送 roll 正確、建議正名 lean)、座標軸(Y 前後非標準)、command 佈局、控制模式、baud 921600、縮寫。已加入 README。
- [x] **B2 git 歷史考古** ✅ → `reports/INVESTIGATION_git_archaeology_2026-06-04.md`。最大發現：🔴 今日 `control_mode` 字串模式系統其實是一次「想遷移到 OperatingMode/ControlSubMode 乾淨列舉但失敗被回退」的殘存物(a8a723e 引入+legacy shim，現已全移除，周邊一串「加回 X 避免 crash」)。另：「多執行緒重構=完全穩定」宣稱被後續硬體修補打臉、物理/求解參數反覆 revert、pyserial 腳本刪又還原。
> ⬇️ B3–B8 已於 2026-06-07 改名搬入 T02 批次並全數完成（見下方 T02 段落）。此處保留對照、勿重跑。
- [x] **B3 死碼/未使用偵測** ✅ = **T02-1**（→ reports/AUDIT_2026-06-07.md）
- [x] **B4 config.yaml 稽核** ✅ = **T02-2**（→ AUDIT 報告 + REFACTOR C-CFG）
- [x] **B5 依賴稽核** ✅ = **T02-3**（→ AUDIT 報告 + REFACTOR C-DEP）
- [x] **B6 pytest 基礎架構** ✅ = **T02-4**（pytest.ini/conftest/test_smoke，13 passed）
- [x] **B7 韌體 pupper_recoil 盤點+sweep** ✅ = **T02-5**（→ reports/INVENTORY_firmware_2026-06-07.md）
- [x] **B8 工作區根 README** ✅ = **T02-6**（工作區根 README.md）

## T02 批次（2026-06-07 規劃，待 compact 後自動執行）
> 由 `TASK_PLANNER_2026-06-07.html` 規劃 + 第一輪重構討論授權。決策權威見 `REFACTOR_DECISIONS.md`。
> 規則同 T01：本機 commit 不 push；控制碼/刪碼一律隔離分支 + 加測試、不併 main；遇阻塞標記跳過；將就設計記 REFACTOR_SCOPE。
> **跑完每項要注記「需歸檔/記憶」的重點，確保可直接 compact。**

純讀 / 文件類（安全，依序）：
- [x] **T02-1 死碼/未使用偵測** ✅ → `reports/AUDIT_2026-06-07.md`。孤兒：tree_demo_server.py、_update_recoil_warning_timer、README 引用已不存在的 observation.py；test/ 混入多支非測試工具腳本(建議移 tools/)。只報告未刪。
- [x] **T02-2 config.yaml 逐鍵稽核** ✅ → AUDIT 報告。死鍵：被註解的 observation_recipes、warmup_duration=0 疑未用；command_scaling_factors 註解過時(寫3軸實為4值)；auto_inhibit 預設 true。記 REFACTOR C-CFG。
- [x] **T02-3 依賴稽核** ✅ → AUDIT 報告。scipy 多餘(剔除)、glfw 漏列、mujoco 應升核心、onnxruntime-directml 屬選配。建議建 requirements.txt(排 T03)。
- [x] **T02-4 pytest 基礎架構** ✅：`pytest.ini`+`conftest.py`+`test/test_smoke.py`；--ignore 隔離 test_joystick/test_teensy_connection(會炸收集，D-1)。全套 8→13 passed。
- [x] **T02-5 韌體 pupper_recoil 盤點+sweep** ✅ → `reports/INVENTORY_firmware_2026-06-07.md`。指令集/模組/baud921600 齊；氣味：除錯指令未隔離、homing_main 命名。純讀。
- [x] **T02-6 工作區根 README** ✅：根目錄 `README.md`(三 repo + quickstart + conda/PYTHONUTF8 + 跨 repo 陷阱)。註：根非 git repo，此檔未版控。
- [x] **T02-7 後座力斷線端到端深查報告** ✅ → `reports/INVESTIGATION_recoil_wiring_2026-06-07.md`。確認三路徑兩斷(timer 孤兒+switch 無 subscriber)，僅手動按鈕活；含修復方案。

授權的程式碼變更（原隔離分支，已 review 併 main）：
- [x] **T02-8 後座力接線修復** ✅ **已併 main**（commit f4e4237，撿碼跳 docs 雜訊）：timer 接回 run loop + runtime 旗標 + switch subscriber + 5 測試。
- [x] **T02-9 移除 NanoOwl 影像伺服器** ✅ **已併 main**（commit d4796a2）：刪 Popen/tree_demo_server.py/nanoowl_process/subprocess import。
- [x] **T02-10 後座力旗標改正向命名** ✅ **已併 main**（commit ba439e9）：`auto_inhibit`(true=抑制)→`auto_warning_enabled`(false=不啟用)，去雙重否定；預設 false=只保留手動。涵蓋 config/state/event/controller/ui/測試。

**T02 收尾（2026-06-07）**：全項完成且**全部已併 main**（兩分支已 review 併入、過期分支已刪）。皆本機 commit、未 push。

## T03 批次（2026-06-07 規劃）
> 聚焦走路/控制（後座力短期內多半停用）。重構決策權威見 `REFACTOR_DECISIONS.md`。
- [x] **T03-1 情境A terrain_cache 久跑實測** ✅ → `reports/TERRAIN_CACHE_2026-06-07.html` + `tools/measure_terrain_cache.py`。走 20km：地塊 25→20,020 線性無上限，但 heap 僅 +4.5MB → **卡頓主因非快取記憶體**（C5-1 降 🟡、新增 C5-4 嫌疑）。
  - [ ] T03-1b 情境B **平地**走路品質 baseline：需真實物理+policy 的 headless eval harness（現 run loop 與視窗耦合、headless 僅 mock 無真物理）→ 併入 P3「抽核心」一起設計，暫緩單獨做。
- [x] **T03-2 第一輪重構完整路線圖** ✅ → `reports/REFACTOR_ROADMAP_2026-06-07.html`（HTML+SVG）。✅ 定案：P1/P2 **串行**、terrain LRU **降 P3**、下一步 **P1 起步**。

## T04 批次 = P1 機械式清理（2026-06-07 規劃，今晚自動執行）
> **路線圖 P1**。原則：低風險、不動控制路徑邏輯、可回退。決策權威 `REFACTOR_DECISIONS.md`。
> **自動執行規則**（沿用上方「自主執行規則」，並強化）：
> - 取最上面 `[ ]` 任務做；本機 commit、**絕不 push**。
> - **加檔/改文件類在 main**；**動程式碼一律開分支 + 跑測試**；**刪除核心碼一律只列清單不動手**（刪除留 Harrison review）。
> - **防阻塞**：遇歧義/需決策/環境問題 → 標 `[BLOCKED]` 寫因、**跳下一個**，絕不卡死。
>   分支類若測試 3 次內修不綠 → `git restore` 該任務變更、標 `[BLOCKED]`、繼續下一個。
> - 每改一檔跑 `PYTHONUTF8=1 python -m pytest -q`（須維持綠）+ 受影響入口 `ast.parse`/import smoke。
> - 新發現的將就設計記 `REFACTOR_SCOPE.md`。
- [ ] **T04-1 建 requirements.txt**（main，加檔最安全）：依 T02-3 稽核——剔 scipy、補 glfw、列 pytest/Pillow/psutil(視用到否)；版本 pin 用 `pupper-sim` env 實際版本。不改任何程式碼。
- [ ] **T04-2 萃取版本歷史**（main，加檔安全）：把散落的 `【vX.X.X】` 行內註解 + git tag 萃取成 `docs/VERSION_HISTORY.md` 時間線。**只萃取、本批不刪註解**（刪除動到全 repo，留 P1 後續 review 步驟）。
- [ ] **T04-3 print→log 轉換**（分支 `chore/print-to-log`）：把 `print(...)` 機械式換成對應 log 等級（依 CLAUDE.md 語義：狀態變更=info、診斷=debug）。逐檔轉、逐檔測；全套 pytest 綠才提交。fallback：某檔轉後測試掛且修不動 → restore 該檔、記 BLOCKED、續其他檔。
- [ ] **T04-4 死碼保守處理**（分支 `chore/deadcode-tidy`）：只做**高信度可逆**動作——`test/` 內非測試腳本移 `tools/`、修 README 對已刪檔的失效引用。**核心程式碼疑似死碼只更新清單於 AUDIT 報告、不刪**。
- [ ] **T04-5 收尾歸檔**（main）：產 `reports/OVERNIGHT_SUMMARY_2026-06-08.html`（**字級加大、最小 ≥13px**）；更新 `restart-progress` 記憶；確認工作樹乾淨、列出當晚所有 commit 與待 review 分支；**不 push**。

## 已完成 / 已封存
（完成的任務移到這裡保留紀錄）
- T01 批次（2026-06-04）：主 T1–T9 + 備選 B1–B2 全完成；硬體修復 T3/T3b 已於 2026-06-07 併入 main（commit 478ee6c，未 push）。
