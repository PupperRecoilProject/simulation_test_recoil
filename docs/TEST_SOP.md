# 測試 SOP（DRAFT）— 讓專案穩定的可重複測試流程

> 草稿，待 Harrison 校對。入口見 `README.md`。
> 目的：建立「程式單元/整合 + 模擬 + GUI」三層測試的標準作業流程，並明確切分
> **AI 可全自動** vs **需 Harrison 介入** 的部分（後者寫成可操作步驟）。終極目標：AI 能自主驗證系統。

## 0. 環境前置（每次測試共用）
- conda env：`pupper-sim`（見 environment 記憶）。叫用：
  `& "C:\Users\Harrison\miniconda3\envs\pupper-sim\python.exe" ...`
- **務必設 `PYTHONUTF8=1`**（否則啟動因 cp950 崩潰，見 REFACTOR C-S1；修好前都要設）。
- 在 repo 根（`simulation_test_recoil/`）執行。

## 1. 測試金字塔（投資比重）
```
        ▲  少   E2E / GUI smoke (Playwright, web client)         ← 慢、脆、只測關鍵流程
       ╱ ╲     模擬整合測試 (啟動→走幾秒→斷言不崩/狀態合理)
      ╱   ╲    核心 API 單元/整合測試 (快、無 UI、主戰場)
     ╱─────╲   ← 重構後打「Control Core」；現在先打可隔離的純函式
```
重構後**主力打核心 API**（快、不需 UI/顯示器）；CLI/GUI 只配少量 smoke/e2e。

## 2. Layer A — 程式單元/整合測試（pytest）⟶ ✅ AI 全自動
**現況**：`test/` 已有 `test_serial_utils`、`test_teensy_connection`(需硬體→skip)、`test_hardware_handoff`(T3b 新增)。
**SOP**：
1. 跑全套：`PYTHONUTF8=1 python -m pytest test/ -q`
2. ⚠️ **已知壞檔**：`test/test_joystick.py` 在 import 期呼叫 `exit()`，會炸掉整個收集（REFACTOR D-1）。
   暫時用 `--ignore=test/test_joystick.py` 跑；修好前不要納入全套。
3. 綠燈＝通過；任何 red/error 需先排除再繼續。
**優先補的特徵測試**（重構護網，對應 ARCHITECTURE_PROPOSAL §3.1）：
   obs 組裝（ObservationManager）、policy 推論 I/O 形狀、模式切換狀態機、序列埠交接（已起頭）。
**建議基礎建設（B6）**：加 `pytest.ini`/`conftest.py`（統一 sys.path 與 PYTHONUTF8）、`test/` 分 `unit/`、`integration/`。

## 3. Layer B — 模擬整合 / smoke ⟶ ✅ AI 全自動（不需實體機）
**目的**：sim 能啟動、跑一段、狀態合理、不崩、不洩漏。
**SOP**：
1. 啟動 smoke（已驗證可行，見 reports/SMOKE_2026-06-04.md）：用一次性啟動器跑 `main_nicegui.py`，
   活過 N 秒視為啟動成功，再以 CTRL_BREAK 收尾；擷取啟動期 error/warning。
2. **記憶體/效能長跑驗證**（針對 C5-1 terrain_cache）：INFINITE 地形持續前進數分鐘，
   定期記錄 `len(terrain_manager.terrain_cache)` 與行程 RSS（psutil）；單調上升＝坐實洩漏。對照 FLAT 應平穩。
3. 重構後改打 Core API：`core.start_sim(); for _ in range(N): core.step(); assert core.get_state() 合理`，免開視窗、可在 CI 跑。
**注意**：目前啟動會連帶起 NanoOwl 子程序（cv2 缺→崩，無害但有噪音，C-S2）；測試判讀時忽略該 traceback。

## 4. Layer C — GUI / Web e2e ⟶ 🙋 半自動（UX 主觀需 Harrison；流程可自動）
- **可自動**：用 Playwright（或 Claude Preview/Chrome 工具）開 web client，點按鈕→斷言狀態/標籤變化（功能性 e2e）。
- **需 Harrison**：UX 主觀好不好用、版面、視覺舒適度 → **不自動**，等重構出 web client 後與 Harrison 一起評估
  （NiceGUI UX 評估明確需 Harrison 在場，見 REFACTOR E）。

## 5. 需 Harrison 介入的測試（寫成可操作步驟）🙋
> 凡涉及實體、主觀、不可逆者。每項給「我準備什麼 / 你做什麼 / 怎麼判定」。
1. **實體 Teensy / 機器狗連線**：
   - 我備：序列埠連線腳本 + 預期 34 欄位格式（INTERFACE_CONTRACT）。
   - 你做：插 USB、選對 COM 埠、回報是否收到遙測、馬達是否動。
   - 判定：欄位數/單位符合契約、無亂碼（baud 921600）。
2. **馬達方向 / IMU 正負號校驗**：
   - 我備：單軸測試指令序列 + 預期符號表。
   - 你做：實機觀察某軸動作方向、傾斜機身看 roll/pitch 數值符號。
   - 判定：對照 INTERFACE_CONTRACT §4（roll/pitch、角速度、加速度待實體確認的符號）。
3. **進出硬體模式穩定性**（驗證 T3/T3b 修復）：
   - 你做：進入硬體→啟用 AI→停止→再進入，循環數次，看是否卡死/連不上。
   - 判定：每次都能正常進出、序列埠不洩漏（修復後預期 OK）。
4. **push / 重大架構決策核可**：分支 review 後由你決定併 main / push。

## 6. 每次「改動後」最小回歸清單（重構期間遵守）
1. `pytest test/ --ignore=test/test_joystick.py -q` 綠。
2. sim 啟動 smoke 通過（Layer B 步驟1）。
3. 若動到硬體/控制路徑 → 走分支 + 對應特徵測試 + 待 Harrison review，**不直接併 main、不 push**。
4. 發現新「將就」設計 → 記 REFACTOR_SCOPE。

## 7. 與既有文件關係
- 問題/將就：`REFACTOR_SCOPE.md`；架構目標：`ARCHITECTURE_PROPOSAL.md`；
  功能對照：`FEATURE_INVENTORY.md`；契約：`INTERFACE_CONTRACT.md`；待挖坑：`OPEN_THREADS.md`。
