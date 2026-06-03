# 防死鎖 / 併發 專題審查 — 2026-06-04（T7，靜態分析）

> 對象：`state.py`(中央鎖)、`event_system.py`、`simulation_controller.py`、`hardware_controller.py`、
> `serial_communicator.py`、`recoil_warning_controller.py`。只做靜態 review。
> 回應 Harrison 疑點：「當時設計過防死鎖機制，但懷疑有沒顧慮到的東西」。
> 信心：🔴高 / 🟡中 / ⚪低。皆建議實測（壓力/長跑/快速切模式）復現。

## 執行緒清單（誰在跑）
1. **NiceGUI/uvicorn**（主程序 + async）：UI 按鈕 → `event_bus.publish(...)`（在 UI 執行緒上同步跑回呼）。
2. **SimulationController.run**（sim 執行緒）：主迴圈，大量 `with state.lock`。
3. **HardwareController.control_thread**：硬體狀態機 + AI step；`_execute_*` 取 state.lock。
4. **HardwareController.read_thread**：讀序列埠 → parse。
5. **SerialCommunicator.read_thread**：HUMAN 模式下讀序列埠（與 #4 用暫停握手交接）。
6. **XboxInputHandler thread**：輪詢手把 → publish。
7. **GLFW 鍵盤回呼**：在 sim 視窗事件執行緒上 publish。
鎖：`state.lock`(中央) ＋ `EventSystem._lock` ＋ `HardwareController._freq_lock` ＋ SerialCommunicator 的兩個 Event 旗標。

## 🔴 X-1 中央鎖是非重入 `threading.Lock` + 「呼叫者須持鎖」的隱性約定
- `state.py:88`：`lock: threading.Lock`（**非 RLock**）。
- 同時，`set_control_mode`/`reset_control_state`/`clear_command` 等 helper **故意不自己鎖**，
  註解明載「假定總在持有鎖的上下文中被呼叫」（state.py:293）。
- 這是雙面地雷：
  - (a) 若某 helper 哪天被加回 `with self.lock:` → **同執行緒重入非重入鎖＝立即自我死鎖**。
  - (b) 若某呼叫者忘了先持鎖就呼叫 helper → **無保護的資料競態**（不會報錯，靜默壞資料）。
- 現況靠人工紀律維持，極脆弱。建議：改 `threading.RLock()`，或建立「公開方法自鎖、私有 `_xxx_locked` 才假定持鎖」的清楚分層。

## 🟡 X-2 持中央鎖跨重量級操作 → 其他執行緒卡頓（疑似「卡」感來源之一）
- `hard_reset()`：在 `with state.lock` 內跑 `for _ in range(10): mujoco.mj_step(...)` + `policy_manager.reset()`。
- `_handle_mode_change()`：在 `with state.lock` 內呼叫 `hardware_controller.request_start()`（啟動執行緒 + 入列）。
- 期間任何其他執行緒（UI 刷新、輸入回呼、控制執行緒 `_execute_start` 的 `with state.lock`）都被擋住。
- 後果：快速切模式 / 重置時，UI 與輸入會短暫凍結。漂亮做法：縮小臨界區，重活在鎖外做，鎖只保護狀態讀寫。

## 🟡 X-3 事件回呼在「發布者執行緒」上同步執行 → 慢回呼阻塞來源執行緒
- `event_system.publish`：複製訂閱列表後在鎖外**同步**逐一呼叫（設計本身 OK，避免持 event 鎖跑回呼）。
- 但回呼是同步的：`on_device_connect_requested` → `serial_comm.scan_and_connect()`（內含 `time.sleep(0.5)` + 埠掃描，**阻塞**）。
- 若此事件由 GLFW 鍵盤執行緒(L 鍵)或 UI 執行緒發布，該執行緒會被阻塞數百 ms ~ 數秒。
- 建議：耗時的「裝置連接/掃描」改丟背景工作（執行緒/工作佇列），回呼只觸發、不阻塞。

## 🟡 X-4 `recoil_warning_active` 鎖使用不一致
- `RecoilWarningController.on_trigger/on_reset`（事件回呼，在發布者執行緒）**直接寫** `self.state.recoil_warning_active`，**不加 state.lock**。
- 而 sim 端 `_update_recoil_warning_timer` 在 `with state.lock` 內寫同一旗標（雖然該函式疑似沒被呼叫，見 T6-6）。
- 同一狀態有的路徑鎖、有的不鎖 → 競態。布林賴 GIL 大多無害，但屬不一致設計。建議統一經 state 的加鎖存取器。

## 🟡 X-5 `ai_step_times` 的 `_freq_lock` 保護不一致
- `_update_frequencies`（line ~389 `popleft`）與另一處 append（line ~684）在 `_freq_lock` 內；
  但 `_control_loop` line 348 的 `self.ai_step_times.append(...)` **未加 `_freq_lock`**。
- 多為同一 control_thread 存取（風險低），但顯示用讀取可能跨執行緒。建議所有存取統一經 `_freq_lock`，消除不一致。
  （延續夜間審查 F3。）

## ✅ 設計上做對的地方
- 旗標交接：`run()` 在 `with state.lock` 內「複製請求旗標 + 立即清除」，再於鎖外執行對應動作 → 正確的請求/消費模式。
- `event_system.publish` 在鎖外跑回呼、先複製列表 → 避免回呼內 (un)subscribe 造成迭代錯誤，也不持 event 鎖跑回呼。
- 主要通知事件（如 `_handle_mode_change` 的 `EVENT_MODE_CHANGED`）在**鎖外**發布 → 未見「持 state.lock 時 publish 給又取 state.lock 的回呼」這條典型死鎖（仍建議立為 lint 規則）。
- 序列埠交接用紅綠燈握手（relinquish/resume），且 T3b 已補 try/finally 保證歸還。

## 未證實但需注意（建議實測復現）
- **快速連點模式切換 / 同時硬體啟停**：X-1+X-2 疊加下，sim 執行緒長持鎖 vs 控制執行緒 `_execute_start` 等鎖，
  雖非死鎖但可能長時間互卡，觀感像當掉。建議壓力測試。
- **publish-within-lock 全面稽核**：本次只稽核主要 controller；建議全域掃一次「`with state.lock` 區塊內是否有 `event_bus.publish`」。

## 建議（待 Harrison 拍板，皆未改碼）
1. `state.lock` → `RLock`，或建立 `_locked` 私有方法分層（解 X-1）。
2. 縮小 `hard_reset`/`_handle_mode_change` 的臨界區（解 X-2）。
3. 耗時裝置連接改背景化（解 X-3）。
4. 統一 `recoil_warning_active`、`ai_step_times` 的加鎖存取（解 X-4/X-5）。
> 將就項已彙整至 REFACTOR_SCOPE「C4. 併發/鎖」。
