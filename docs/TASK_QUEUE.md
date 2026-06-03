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
