# 重構範圍與問題盤點（DRAFT，待 Harrison 校對）

> 來源：2026-06-04 與 Harrison 的討論整理。此為草稿，可能有誤解或遺漏，請校對。
> 核心定位：這是重啟，目標是「做出一個**完整可用**的機器狗系統」，**不接續過去的工作目標**。
> 重構受高度鼓勵。無時間壓力、興趣導向，協作者只有 Harrison + AI 助手。
> **核心目標：機制設計漂亮、功能人性化且完善。** 舊專案中糟糕的設計/隨意決定一律可丟，不被其影響。

## A. 大方向（已確認）
- 完整可用的機器狗系統；不接續舊目標。
- 重構 = 預設選項；以「漂亮的設計」為標準，不在舊作法上將就修補。
- 最終目標：**AI 助手能完全自主操作**（測試、驗證、操作）。若某元件（如現有 GUI）不利於自主操作，可整套替換。

## B. 訓練（暫緩，只盤點問題；不被舊作法影響）
- **座標系超怪**：Y 軸是前後（y+ 或 y- forward），非標準慣例，當時沒修。現在有機會改。
- 訓練組員無深度學習背景、方法不科學、無實驗紀錄 → 模型只是「湊到剛好能走」，無技術深度，可能有莫名決定。
- 決策：**訓練暫緩**；先盤點問題。日後可能整個推掉重寫（重啟可重新用，不必照舊）。
- 模型**可重用**，但**不要基於舊訓練流程下去改**。

## C. sim 重構（盤點「將就設計」重寫）
- 找出當時迫於時間的不完善/將就設計，逐一以漂亮設計重寫。
- **時間同步 / 時鐘機制不完善**：模擬模式理論上應為單一時鐘，但印象中有問題 → 查專案是否有線索。
- **防死鎖機制**：當時設計過，疑有未考慮到的情況 → 開專題檢查。
- **效能**：(a) 筆電跑模擬場景很慢；(b) 久跑會超卡需重開（疑似記憶體洩漏/緩衝無上限）；(c) 按鈕順序有 bug。
- **C-S1（🔴 啟動 bug，T2 實測）**：滿地 emoji `print("✅…")` 無編碼保護，Windows cp950 + 非 UTF-8
  stdout（pipe/寫檔）→ `UnicodeEncodeError` → `main()` try 捕捉後 `sys.exit("failed to initialise")`。
  任何非 UTF-8 終端（含 log 導向檔案/CI/自動化）都起不來。治本：改用既有 `log` logger 取代裸 `print`，
  或入口 `sys.stdout.reconfigure(encoding="utf-8")`/`PYTHONUTF8=1`。詳見 reports/SMOKE_2026-06-04.md S1。
- **C-S2（耦合，T2 實測）**：`main_nicegui.py::main()` 啟動時無條件 `Popen("tree_demo_server.py")`
  拉起 NanoOwl 影像伺服器子程序，無旗標可關；該檔 `import cv2` 缺依賴即崩。與機器狗核心無關的功能硬塞主入口。
  重構應移出主入口、改為可選外掛。詳見 SMOKE S2、孤兒檔見 B3。
- **C-S3（噪音）**：ONNX 預熱刷一堆 onnxruntime C++ 最佳化警告，可降 graph_optimization_level 或調 ORT log severity。

### C2. T5 系統性 sweep（2026-06-04 靜態掃描所得，按嚴重度排）
- **C2-1（🔴 架構重複｜雙鍵盤系統）**：同時存在兩套鍵盤輸入處理：
  `src/input_handlers/keyboard_input_handler.py`（GLFW，掛在 MuJoCo 視窗）與
  `src/controllers/global_keyboard_driver.py::GlobalKeyboardDriver`（NiceGUI 瀏覽器端，ui_controller.py:139 實例化）。
  兩者都綁 WASD/QE/IK/重置/FRW 等重疊鍵位 → 來源不明、行為可能不一致、維護成本翻倍。
  疑為「GLFW 視窗→瀏覽器」遷移的中間遺留。重構應收斂為單一輸入抽象（事件源 → 指令），是高價值清理點。
- **C2-2（🟡 單向數據流破例）**：`xbox_input_handler.py:50` 註解明載「暫時還需要直接修改 state，後續可以改為發布事件」
  → 直接寫 state 而非走 event_bus，違反專案「輸入→事件→Controller→改 state」原則。應改發事件。
- **C2-3（🟡 執行期變動 config）**：`keyboard_input_handler.py:51-56` 在執行期對 config dataclass 補寫
  `keyboard_pitch_step`（若 yaml 沒定義）。config 應為不可變 SSoT；缺鍵應在載入/驗證階段補預設，而非散落在 handler。
- **C2-4（🟡 寬鬆 except 吞錯）**：`keyboard_input_handler.py` 與 `global_keyboard_driver.py` 多處 `except Exception:`（甚至 pass）
  靜默吞掉所有錯誤（global_keyboard_driver 就有 ~9 處）→ 難除錯、可能遮蓋真實 bug。應收斂為具體例外型別 + 記 log。
- **C2-5（🟡 print 濫用｜C-S1 根因）**：src 內 **64 處頂層 `print(`**（含 emoji），未走 `log`。
  這正是 C-S1 cp950 崩潰的根因，也使日誌等級/導向失效。應統一改用 `log`。
- **C2-6（⚪ 版本註解雜訊）**：src 內 **274 處 `【vX.X.X 修改/新增】` 行內版本註解**（hardware_controller 就 76 處）。
  CLAUDE.md 已定調：版本歷史交給 git，不再行內標。屬大規模清理（建議重構時順手移除，勿單獨大改）。
- **C2-7（⚪ 全域可變狀態）**：`main_nicegui.py` 用 module-global `nanoowl_process` 持有子程序。與 C-S2 同根（NanoOwl 硬塞主入口），重構移出時一併處理。
- **C2-8（指向既有議題）**：`observation_manager.py:197` TODO（v5.0.0 狀態估算器）、`:240` TODO（需與 Teensy IMU
  實際座標 Y-fwd/X-right 校對）→ 與 B 節「非標準軸向慣例」同源，正名/校對時一起處理。`xbox_input_handler.py:67`
  手把第 4 軸 pitch 固定 0（未映射）。
- 備註：上述僅靜態掃描的代表性樣本，非窮舉；重構各模組時應再就地複查。

### C3. 時間 / 時鐘（T6 專題，詳見 reports/REVIEW_timing_2026-06-04.md）
- **🔴 C3-1 雙時鐘**：sim 物理走模擬時間累加器，AI 決策卻綁牆鐘 `current_time`，兩者不同源；
  負載高時物理被夾成慢動作而 AI 仍按牆鐘 → 每模擬步的 AI 次數漂移，脫離訓練固定 dt 假設。
  漂亮解：AI 決策改綁物理步數/`sim.data.time`，收斂為單一模擬時鐘。
- **🟡 C3-2**：`current_time` 在內層物理迴圈未重新取樣 → AI 決策節奏抖動（與 C3-1 同源）。
- **🟡 C3-3**：`MAX_FRAME_TIME=0.25` 魔術數字在 sim/hw 各寫死一份 → 應入 config 具名單一來源。
- **🟡 C3-4**：主迴圈 `time.sleep(0.001)` 未顧 Windows ~15.6ms 計時器解析度 → 迴圈頻率被壓低；
  應設計時器解析度或改事件驅動。
- **🟡 C3-5**：`control_freq` 一值身兼三職（sim AI 節奏 / hw 控制迴圈 / Teensy 遙測頻率）→ 拆具名參數。

### C4. 併發 / 鎖（T7 專題，詳見 reports/REVIEW_concurrency_2026-06-04.md）
- **🔴 C4-1 非重入中央鎖 + 隱性持鎖約定**：`state.lock` 是 `threading.Lock`(非 RLock)，而 helper(`set_control_mode`
  /`reset_control_state`/`clear_command`)「假定呼叫者已持鎖、自己不鎖」。→ helper 一旦被加回鎖即自我死鎖；
  呼叫者忘了鎖即靜默競態。建議改 RLock 或建立 `_locked` 私有方法分層。
- **🟡 C4-2 持中央鎖跨重活**：`hard_reset`(鎖內跑 10× mj_step + policy reset)、`_handle_mode_change`(鎖內 request_start)
  → 其他執行緒被擋，快速切模式/重置時 UI 凍結。應縮小臨界區。
- **🟡 C4-3 事件回呼同步阻塞發布者執行緒**：`on_device_connect_requested` 在回呼內跑阻塞的 `scan_and_connect`
  (含 sleep 0.5 + 埠掃描)，會卡住發布它的鍵盤/UI 執行緒。耗時裝置連接應背景化。
- **🟡 C4-4 `recoil_warning_active` 鎖不一致**：RecoilWarningController 回呼不加鎖直寫，sim 端加鎖寫 → 統一加鎖存取器。
- **🟡 C4-5 `ai_step_times` `_freq_lock` 保護不一致**：control_loop:348 append 未鎖、其他處有鎖（延續 F3）。統一。
- 建議全域 lint：稽核「`with state.lock` 區塊內是否有 `event_bus.publish`」（持鎖 publish → 回呼再取鎖＝死鎖）。

### C5. 效能 / 記憶體（T8 專題，詳見 reports/REVIEW_performance_2026-06-04.md）
- **🟡 C5-1 terrain_cache 無上限成長（T03-1 實測：成長屬實但非卡頓主因）**：INFINITE 地形的 `terrain_cache`
  (terrain_manager.py:74) 是「持久性」設計，走過的每個網格 tile 只增不刪，僅 reset 才 clear、無 eviction。
  **2026-06-07 實測**（`reports/TERRAIN_CACHE_2026-06-07.html`，走 20km）：地塊數 25→20,020 **線性無上限**確認，
  但 `TerrainTile` 極輕（~224B/塊），Python heap 僅 +4.5MB → **單純快取記憶體不是「久跑超卡」的合理主因**，
  降級為 🟡。仍建議上 **LRU 上界**（架構衛生）。⚠️ 卡頓真因改查 C5-4。
  解：窗口外 tile evict / LRU 上限，或改「種子重生」免快取整塊。
- **🔴 C5-4 每次網格滑動的固定重繪開銷（卡頓新嫌疑）**：`shift_grid_center`→`update_hfield` 每次滑動
  對 5×5 視窗**每塊都重跑地形 generator（25 次）**、重填 ~501×501 `full_hfield_data`、寫回
  `model.hfield_data` 並觸發物理/渲染同步。此固定成本與行走頻率相關，比快取大小更可能是「久跑超卡」來源。待真實物理+渲染情境量測佐證。
- **🟡 C5-2 data_capture_buffer 無硬上限**：捕獲中每幀 append，靠時長/手動停止結束；漏關即成長。加最大幀數保護。
- **🟡 C5-3（呼應 T7）高頻物理步持鎖 + 大量 `.copy()`**：GC churn + 鎖競爭，筆電上拖慢。縮小臨界區/重用緩衝/批次寫回。
- 已確認有界良好(非洩漏)：log_queue(500)、freq deque(100)、obs_histories(history_length)、
  rendering 的 scene/context/cam 重用、事件訂閱一次性。

### C6. T02 稽核補充（2026-06-07，詳見 reports/AUDIT_2026-06-07.md）
- **C-CFG config.yaml 將就項**：被註解掉的 `observation_recipes`(死 config)、`warmup_duration:0` 疑未用死鍵、
  `command_scaling_factors` 註解過時(寫「[vy,vx,wz]」3軸實為4值)、`auto_inhibit` 預設 true(自動預警預設關)。
- **C-DEP 依賴**：README pip 清單 scipy 多餘、glfw 漏列、mujoco 應升核心、onnxruntime-directml 屬選配；
  建議建 `requirements.txt`/`environment.yml` 取代散落 pip 行（排 T03）。
- **C-TEST test/ 目錄混雜**：多支非測試工具腳本(dump_project/project_overview/pyserial_*/verify_model_mode/test_teensy_connection)
  與真單元測試混放；建議分到 `tools/`，pytest 只掃真測試。`test_joystick.py` import 期 `exit()` 已用 --ignore 暫隔離(根治待移目錄)。

### ✅ 已於 T02 處理（記錄存證）
- **C-S2 / C2-7 NanoOwl 硬塞主入口** → 已移除（分支 `chore/remove-nanoowl`，待 review）。
- **後座力斷線**（timer 孤兒 + switch 無 subscriber）→ 已接回（分支 `fix/recoil-wiring`，待 review）。

## D. 測試 / 品質（從零建立，工程化）
- 當時幾乎沒寫 pytest → 必有運行 bug、以及「原本設計但被改壞」的功能。
- 目前無完整測試 SOP：需涵蓋 (1) 程式單元/整合測試 (2) GUI 介面測試 (3) 模擬測試。
- 由 AI 盡量包辦測試流程；需 Harrison 介入的部分要寫成**可操作的 SOP**（工程思維建立，Harrison 想學去他處用）。
- 自主操作為終極目標。

## E. GUI
- 現有 NiceGUI：功能內容 OK，但**運作不一定正常、不夠好用**。
- **退出硬體模式會卡死** → 已對應到 `hardware_controller._execute_stop` 的 self-join bug（REVIEW_hardware_stability F1，高度吻合）。
- 若不利 AI 自主操作，可整套換（需保留可被測試/自動操作的控制介面，例如 CLI/API 優先、UI 為薄層）。
- **NiceGUI UX 評估（瀏覽器截圖+問題清單）= 需 Harrison 在場一起做、重要、不自動執行。** 等重構方向定了再評估介面較有意義。

## F. 流程 / 雜項
- 先**記錄所有現有功能**（feature inventory），再重構、替換舊機制架構。
- 喚醒：已設 +2h/+3h（session-only，見討論）。
- 記憶：已是專案本地記憶（在專案 .claude 目錄下）。
- 不急著 compact，先充分研究討論。

## 目標架構決策（已採納，2026-06-04）
**分離原則：headless 控制核心 + 薄客戶端。** 把控制邏輯做成無介面的核心（乾淨 Python API），
介面只是可替換的客戶端。好處：可測試、AI 可完全自主操作、UI 隨時換皮。
- 核心：headless 控制 API（Python）。
- 開發/自動化/我操作 + 測試驅動：**CLI（建議 Typer）**。
- 即時 UI（之後挑）：FastAPI+WebSocket+單頁 HTML/JS（取代現有 NiceGUI），資料視覺化可用 rerun.io。
- 測試策略：單元/整合測試打**核心 API**（快、不需 UI）；CLI/GUI 只配少量 smoke/e2e（Playwright）。
- 現有 NiceGUI：API 出來後降級為可替換客戶端，不滿意直接換。

## 自主測試範圍（我能做 vs 需 Harrison）
- ✅ 可全自主：pytest、啟動 sim 做 smoke test、瀏覽器自動化驅動 web UI、**用 `src/mock` 假硬體跑「進出硬體模式」**（故能自主重現/驗證「退出硬體卡死」）、靜態分析、效能/記憶體 profiling。
- 🙋 需 Harrison：真 Teensy/機器狗實測、馬達方向/IMU 正負號、插拔 USB、主觀 UX 拍板、push 核可、重大架構決策。
- 前提：要我自主跑 sim 測試，需先建好裝有 sim 依賴的 Python 環境（見 environment 記憶；待辦）。

## 待 Harrison 補充 / 我注意到的缺口
- 你訊息中有兩處「還有」後面似乎沒接內容，可能有想講漏掉的，請補。
- recoil 模型 OOD 問題（餵非零運動指令給站立模型）也屬「將就/需釐清」。
- roll/pitch 命名與 Y-前後 的怪座標是同一套「非標準軸向慣例」的不同面向，重構時應一起正規化。
