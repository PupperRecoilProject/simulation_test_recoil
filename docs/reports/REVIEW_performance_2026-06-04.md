# 效能 / 記憶體洩漏 靜態獵查 — 2026-06-04（T8，靜態分析）

> 對象：`terrain_manager.py`、`ui_controller.py`、`rendering.py`、`policy.py`、`logger.py`、freq buffers。
> 目標：找「無上限成長的緩衝/狀態」以解釋 Harrison 回報的「久跑會超卡、需重開」。只做靜態 review。
> 信心：🔴高 / 🟡中 / ⚪低。

## 🔴 P1 最強嫌疑：INFINITE 地形 `terrain_cache` 只增不刪（記憶體隨行走距離無上限成長）
- `terrain_manager.py:74` `self.terrain_cache: Dict[(int,int), TerrainTile] = {}`，
  註解明載「儲存**所有**生成過的地塊，確保地形的持久性」。
- 滑動窗口 `update()`→`shift_grid_center()`→`_get_or_create_tile()`（line 236-251）只會**新增** tile 進 cache；
  全 repo 對 cache 僅有 `.clear()`（hard reset / `regenerate_terrain_and_adjust_robot` / `reset`），
  **無任何逐塊 eviction（grep 無 pop/del）**。
- 後果：在 INFINITE 模式持續行走，機器人經過的**每一個**新網格座標都永久保留一個 `TerrainTile`
  （內含 `tile_resolution²` 的 heightfield 陣列）。走得越久 → cache 條目與記憶體單調成長，且 dict 越大查找越慢。
  **這與「久跑超卡、需重開」高度吻合**（尤其長時間連續走 INFINITE 地形）。
- 漂亮做法：把「持久性」改為**有界 LRU / 滑動窗口外的 tile 直接淘汰**（離開窗口範圍即 evict，或設 cache 上限）。
  若需「回到原地地形一致」，可改存「種子 + 程序生成」而非快取整塊資料（重算成本換記憶體）。
- ⚠️ 屬效能/記憶體核心問題，已記 REFACTOR_SCOPE C5-1。建議用實測佐證（見下「驗證建議」）。

## 🟡 P2 高頻物理迴圈的鎖 + 複製造成 GC/競爭壓力（非洩漏，但拖慢）
- `_single_physics_step` 每個物理步都 `with state.lock` 並對多個陣列 `.copy()`（torso_quat/vel/joint pos/vel/accel…），
  接著 `observation_manager.update_all_observations()`。物理步頻率 = 1/physics_timestep（可能數百 Hz）。
- 後果：大量短命 numpy 物件 → GC churn；且每步搶中央鎖 → 與 UI/輸入執行緒競爭（呼應 T7 C4-2）。
- 非記憶體洩漏（物件會回收），但會讓筆電上「感覺慢/卡」。漂亮做法：批次寫回、縮小鎖臨界區、重用緩衝陣列。

## 🟡 P3 UI 10Hz 刷新持中央鎖
- `ui_controller.py:257` `ui.timer(0.1, self.update_ui_elements)`；`update_ui_elements` 開頭 `with self.state.lock` 取快照。
- 元件更新本身**正確**（`label_widget.set_text(...)` 原地更新預建 label，**非重建元件** → 無 DOM/元件洩漏 ✅）。
- 但每 0.1s 在 NiceGUI 執行緒搶中央鎖，與 sim 執行緒（尤其 hard_reset 持鎖跑重活）競爭 → 偶發卡頓。屬 T7 C4-2 同源。

## 🟡 P4 `data_capture_buffer` 捕獲期間成長（漏關即無上限）
- `policy.py`：開始捕獲時 `clear()`，捕獲中每幀 `append(current_frame_data)`（line 308）。
- 由「時長」或「手動停止」結束。若使用者開了捕獲忘了停（或 duration 設很大），buffer 會持續成長。
- 屬預期內的暫存，但缺硬上限。建議加最大幀數保護 + 達上限自動停並警告。記 C5-2。

## ✅ 已確認「有界、良好」的緩衝（非洩漏）
- `logger.py:6` `log_queue = deque(maxlen=500)` ✅
- `hardware_controller` `data_received_times` / `ai_step_times` `deque(maxlen=100)` ✅
- `policy.py` `obs_histories[name] = deque(maxlen=history_length)` ✅
- `rendering.py`：`sim.scene/context/cam` 為一次性建立並重用；每幀只造小 `MjrRect` 與 `np.array2string` 字串（瞬時、可回收）→ 無渲染上下文洩漏 ✅
- 事件訂閱：各 controller 在 `__init__` 訂閱一次（非每幀/每次切換重複訂閱）→ 無訂閱者洩漏 ✅
  （EventSystem.subscribe 亦去重，line 77）。

## 驗證建議（待 Harrison / 可後續自動化）
1. **P1 實測**：以 PYTHONUTF8=1 啟動 sim，進 INFINITE 地形持續前進數分鐘，
   每隔一段時間印 `len(terrain_manager.terrain_cache)` 與行程 RSS（psutil）。預期：兩者隨時間單調上升＝坐實 P1。
2. 對照組：FLAT/SINGLE 地形長跑，cache 不應成長 → RSS 平穩，即可把「超卡」歸因於 P1。

## 建議（待拍板，皆未改碼）
1. **P1（最高優先）**：terrain_cache 改有界（窗口外 evict 或 LRU 上限），或改「種子重生」免快取整塊。
2. P2/P3：縮小中央鎖臨界區、重用緩衝、批次寫回（與 T7 重構一起做）。
3. P4：data_capture_buffer 加最大幀數保護。
> 將就/設計項已彙整至 REFACTOR_SCOPE「C5. 效能/記憶體」。
