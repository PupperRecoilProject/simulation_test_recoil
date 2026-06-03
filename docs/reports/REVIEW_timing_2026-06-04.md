# 時間 / 單一時鐘 專題審查 — 2026-06-04（T6，靜態分析）

> 對象：`simulation_controller.py::run`、`hardware_controller.py::_control_loop`、`config.py` 時間欄位、
> `simulation.py`。只做靜態 review（未跑長時間實測）。回應 Harrison 疑點：「模擬模式理論上應為單一時鐘，但印象中有問題」。
> 信心：🔴高 / 🟡中 / ⚪低。

## 結論先講
模擬模式**不是單一時鐘**：物理走「模擬時間累加器」，但 AI 決策節奏卻綁「真實牆鐘」，兩者來源不同步。
負載一高（筆電），物理被 `MAX_FRAME_TIME` 夾住變慢動作，AI 決策卻仍按牆鐘 → 每模擬秒的 AI 次數會漂移，
這正是「印象中時鐘有問題」的根。下方逐點。

## 系統現況：三條獨立時間節奏
1. **物理**：`physics_accumulator += 真實 frame_time`，以固定 `physics_timestep` 償還（標準固定步長 game-loop，✅ 正確）。
2. **AI 決策**：`if current_time >= next_logic_update_time: 決策; next_logic_update_time += logic_interval`，
   其中 `logic_interval = 1/control_freq`，`current_time` 取自牆鐘。**← 與物理不同源**。
3. **渲染**：`next_render_update_time` 亦走牆鐘（解耦渲染，✅ 合理）。
另：硬體模式由 `HardwareController._control_loop` 自帶**第四條** accumulator 迴圈（同 `control_freq`），與 SimController 並行。

## 問題清單
### 🔴 T6-1 雙時鐘：物理(sim 時間) vs AI(牆鐘)不同源
- 物理推進量＝累加的真實時間 / `physics_timestep`；AI 觸發＝牆鐘是否跨過 `next_logic_update_time`。
- 正常負載下兩者大致對齊；但一旦跟不上即時，物理被 `MAX_FRAME_TIME=0.25s` 夾住（每輪最多推 0.25s 物理），
  模擬進入「慢動作」，而 AI 決策仍照牆鐘節奏 → **每模擬步的 AI 決策數不再恆定**，控制行為與訓練時的固定 dt 假設脫節。
- 漂亮做法：AI 決策也綁模擬時間（用 `sim.data.time` 或物理步數計數），讓「每 N 個 physics step 跑一次 AI」恆定，
  使整個模擬成為**單一（模擬）時鐘**；牆鐘只用於渲染/即時節流。

### 🟡 T6-2 `current_time` 在內層物理迴圈未更新（時間取樣點問題）
- `current_time` 只在外層迴圈頂端取一次（步驟1）。內層 `while physics_accumulator >= physics_timestep`
  可能一輪跑多個物理步，但比較 AI 觸發時用的是**同一個過期 `current_time`**，而 `next_logic_update_time` 持續累加。
- 後果：AI 決策在「落後牆鐘」時可能於同一外層輪內連續觸發（追趕），或在快速輪內被抑制 → 決策節奏抖動、不均勻。
- 與 T6-1 同源；若改用模擬時間/步數計數即一併解決。

### 🟡 T6-3 `MAX_FRAME_TIME = 0.25` 魔術數字、兩處重複
- `simulation_controller.py:130` 與 `hardware_controller.py:316` 各寫死一份「防螺旋死亡」上限。
- 屬將就：應移入 config 並命名（如 `max_frame_time`），單一來源。記 REFACTOR。

### 🟡 T6-4 `time.sleep(0.001)` 與 Windows 計時器解析度
- 主迴圈尾端 `time.sleep(0.001)`。Windows 預設計時器解析度約 15.6ms，未調 `timeBeginPeriod` 時
  1ms sleep 實際可能睡 ~15ms → 主迴圈最高頻率被壓低，連帶影響渲染/AI 取樣的牆鐘精度（放大 T6-1/T6-2）。
- 漂亮做法：明確設定計時器解析度，或改用事件/條件變數驅動而非忙等 + 短睡。

### 🟡 T6-5 `control_freq` 一個旋鈕身兼三職（語義過載）
- 同一 `control_freq` 同時決定：(a) sim AI 決策間隔 `logic_interval`；(b) `HardwareController` 控制迴圈節奏；
  (c) Teensy 遙測頻率 `monitor freq`。三者物理意義不同卻共用一值，調一處會牽動三處。
- 漂亮做法：拆成 `ai_control_freq` / `telemetry_freq` 等具名參數（即使預設相同），語義清楚、可獨立調。

### 🔴 T6-6 `_update_recoil_warning_timer()` 疑似孤兒（後座力倒數沒在跑？）
- 此函式（`simulation_controller.py:208`，內含唯一的 `recoil_timer -= control_dt` 倒數邏輯）**全 repo 無任何呼叫者**。
- `recoil_timer` 只在 `hard_reset` 被重設，但沒有任何主迴圈路徑在遞減它（除非 `recoil_warning_controller`
  另有獨立計時——需核對該檔）。若確為未呼叫，則 sim 的隨機後座力事件根本不會觸發 → 是「設計過但被改壞/接線斷掉」的典型。
- 行動：**需 Harrison/實測確認**。已記 OPEN_THREADS。注意它假設「每次呼叫＝過了 control_dt」（又一個隱含時鐘假設）。

## 對「模擬很慢 / 久跑超卡」的關聯
- 「很慢」：T6-1 的慢動作夾制 + T6-4 的 sleep 解析度，會讓筆電上模擬感覺拖慢（而非掉幀）。
- 「久跑超卡」較可能是記憶體/緩衝問題（留待 T8），時間機制本身未見無上限累積。

## 建議（待 Harrison 拍板，皆未改碼）
1. 把 sim 收斂為**單一模擬時鐘**：AI 決策改綁物理步數 / `sim.data.time`（解 T6-1、T6-2）。
2. `MAX_FRAME_TIME`、計時器解析度、`control_freq` 三職 → 進 config 並具名（T6-3/4/5）。
3. 查清 `_update_recoil_warning_timer` 是否真的沒被呼叫（T6-6）。
