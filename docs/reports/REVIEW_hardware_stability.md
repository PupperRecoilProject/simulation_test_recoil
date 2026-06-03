# 硬體穩定性靜態審查 — 2026-06-04（夜間自動任務）

> 對象：`src/controllers/hardware_controller.py`、`src/hardware/serial_communicator.py`、`src/hardware/teensy_api.py`。
> 只做靜態 code review（未執行、未改碼）。聚焦「進出硬體閃退/卡死」與序列埠交接。
> 信心標註：🔴高 / 🟡中 / ⚪低。各項都建議實機/實測復現確認。

## 🔴 F1. `_execute_stop` 對自己的執行緒呼叫 join → RuntimeError（最可疑）
- `_execute_stop()` 是在 `_control_loop()` 內被呼叫的（命令迴圈 `HWCommand.STOP` 分支），
  也就是它**執行在 `self.control_thread` 這條執行緒上**。
- `_execute_stop()` 結尾（v4.13.10-w）有：
  ```python
  self._is_running_event.clear()
  if self.control_thread and self.control_thread.is_alive():
      self.control_thread.join(timeout=0.5)   # ← 當前執行緒 join 自己
  ```
- Python 對「執行緒 join 自己」會丟 `RuntimeError: cannot join current thread`。
- 命令迴圈未對 `_execute_stop()` 包 try/except → 例外往上拋出 `_control_loop`，**控制執行緒以未捕捉例外結束**，
  且其後的清理 log（"硬體控制與讀取執行緒已停止"）不會執行。
- 影響：STOP 大致仍會生效（`_is_running_event` 已 clear、`_set_internal_state(STOPPED)` 在 join 之前已執行），
  但以「丟例外」方式收場，**極可能是「進出硬體不穩/卡死」的來源之一**。
- 建議（待確認後再改）：移除這段 self-join，或改成
  `if threading.current_thread() is not self.control_thread: self.control_thread.join(...)`。
  read_thread 的 join 沒問題（不同執行緒）。

## 🟡 F2. 交接 handshake 若未成對 `resume_control` 會卡住序列埠
- `relinquish_control()`（SerialCommunicator）升起 `_pause_requested`、等 `_is_paused_flag`；
  `resume_control()` 才放下。兩者必須成對。
- `_execute_start` 的各錯誤路徑都有 `resume_control()`（良好）；但若在持有控制權期間發生
  **非預期例外**（未被現有 except 覆蓋），可能漏呼叫 `resume_control()`。
- 後果：SerialCommunicator 讀取執行緒永久暫停；且 `SerialCommunicator.close()` 在
  `_pause_requested` 為 set 時**直接 return 不關埠**（line 161-163）→ 序列埠洩漏，下次連線失敗。
- 建議：在 `_execute_start`/`_execute_stop` 用 try/finally 確保 `resume_control()` 必被呼叫。

## 🟡 F3. 頻率監控的鎖不一致 + deque pop 用法
- `data_received_times` 的存取有 `_freq_lock` 保護；但 `ai_step_times` 在 `_update_frequencies()`
  用 `self.ai_step_times.pop(0)` **未加鎖**（且 `pop(0)` 對 deque 是 O(n)，應用 `popleft()`）。
- 與 `_perform_ai_step` 中 `self.ai_step_times.append(...)` 可能跨幀競爭（同執行緒，風險低）。
- 建議：統一用 `_freq_lock` + `popleft()`。⚪ 影響小，屬整潔性。

## 🟡 F4. 「AI 啟用但 link 非 VERIFIED」會靜默不送指令
- `TeensyAPI.send_motor_commands` 在 `hardware_link_status != VERIFIED` 時**回傳 True 但不送**（靜默）。
- 設計上是「預設安全」（啟動後預設 MUTED），但 UI 若顯示「AI 作用中」而馬達不動，使用者易誤判。
- 建議：UI 明確區分 MUTED/VERIFIED 狀態（可能 ui_controller 已處理，待核）。屬 UX，非 bug。

## ✅ 設計上做得好的地方
- 安全熔斷：`_read_from_port`/`_perform_ai_step` 遇 SerialException → 設 `CONNECTION_LOST` + `FAILED` +
  clear running event，且 `request_start` 拒絕從 `CONNECTION_LOST` 重啟（須重連）。
- 預設安全：啟動成功後預設 `MUTED` 且 AI 不自動跑，待 UI 明確啟用。
- 固定時間步長控制迴圈（time_accumulator + MAX_FRAME_TIME 防螺旋死亡），與 SimulationController 對稱。

## 與「teensy 輸出問題」的關聯
F1 的 self-join 例外發生在「進出硬體」的 STOP 路徑，與該 commit（修進出閃退）情境吻合，
是「交接後殘留不穩」的高度可疑點。建議優先以「進入硬體→啟用→停止→再進入」的循環復現，
搭配觀察控制執行緒是否因 RuntimeError 靜默死亡。

## 建議修復順序（待你 review，皆未動碼）
1. F1（self-join）— 最可能直接解決卡死，改動小。
2. F2（try/finally 保證 resume_control）— 防序列埠洩漏。
3. F3/F4 — 整潔性與 UX。
