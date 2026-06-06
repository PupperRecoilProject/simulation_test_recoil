# 硬體進出 / 序列埠交接修復 — 2026-06-04（T3 + T3b，分支 `fix/hardware-stop-selfjoin`）

> 本機 commit、未 push、未併 main。等 Harrison review。環境：`pupper-sim`（見 environment 記憶）。
> 原則：基於事實的零信任除錯——不盲信我自己 6/4 的舊審查報告，先對照現行碼。

## 重大事實修正：F1（`_execute_stop` self-join）在現行碼**已不存在**
- 舊報告 `REVIEW_hardware_stability_2026-06-04.md` 把「最可疑」指向 `_execute_stop` 內對自身執行緒 `join()`。
- **核對現行 main(v4.14.3)：該 self-join 已於 v4.14.0 移除**（`hardware_controller.py::_execute_stop`
  原處留有註解「移除了會導致死鎖的執行緒 self-join 邏輯」）。現行 `_execute_stop` 結尾只 `_set_internal_state(STOPPED)`，無 join。
- 因此「退出硬體卡死」的根因**不是** F1。我未對已正確的 `_execute_stop` 做任何修改，改為加**回歸測試**鎖死此正確行為。
- 另查 `shutdown()` 內的 `control_thread.join()/read_thread.join()`（line 217-220）：呼叫者僅
  `main_nicegui.cleanup_resources`（NiceGUI 關閉執行緒）與 UI→`request_stop()`（走命令佇列，非 join），
  **都不在控制執行緒上**，故目前不是 self-join。未改它，避免無謂風險。

## 真兇（高度吻合 Harrison 親身症狀「退出硬體後下次連不上」）= F2 交接洩漏
兩個成對缺陷會造成序列埠永久洩漏：
1. **`_execute_start` 的 `resume_control` 不保證成對**：取得控制權（`relinquish_control`）後，
   原 `except` 只接 `serial.SerialException`。若中途拋**非預期例外**，`resume_control()` 不被呼叫 →
   `SerialCommunicator._pause_requested` 永久 set → 讀取執行緒永久暫停。
2. **`SerialCommunicator.close()` 在暫停時直接 return 不關埠**（舊 line 161-163）→ 序列埠不關閉 →
   下次連線失敗。與 (1) 疊加即「退出硬體後連不上、需重開」。

## 本次修改（最小、安全、行為保留）
- `src/controllers/hardware_controller.py::_execute_start`
  - 取得控制權後整段包進 `try/except(SerialException)/except(Exception)/finally`，以 `started_ok` 旗標控制：
    **只要啟動未成功（含非預期例外），`finally` 必呼叫一次 `resume_control()` 並清掉 `self.ser`**；
    成功時則保留控制權（不 resume），維持原語意。移除了散落各失敗分支的重複 `resume_control()`，集中到 finally。
- `src/hardware/serial_communicator.py::close()`
  - 暫停狀態不再「直接 return 跳過關閉」，改為**先清除暫停旗標再正常關閉**（明確呼叫 close() 即代表要關）。
    根除「因未歸還控制權→埠永不關閉」的洩漏。

## 新增回歸測試 `test/test_hardware_handoff.py`（4 條，全綠）
1. `test_execute_stop_does_not_self_join`：在被指定為 control_thread 的執行緒上跑 `_execute_stop`，
   應正常完成、不丟 RuntimeError、狀態到 STOPPED（防有人重新引入 self-join）。
2. `test_execute_start_resumes_control_on_unexpected_exception`：啟動中拋 ValueError（非 SerialException），
   仍 `resume_control()` 恰一次、狀態 FAILED。
3. `test_execute_start_keeps_control_on_success`：成功啟動**不可** resume_control（保留獨佔）。
4. `test_close_releases_port_even_when_paused`：暫停下 close() 仍關埠並清旗標。

驗證：`pytest test/test_hardware_handoff.py` → 4 passed。
全套（排除壞檔 test_joystick）→ 6 passed, 1 skipped（teensy 需實體機）。

## 順帶發現（已記 REFACTOR_SCOPE）
- **`test/test_joystick.py` 第 18 行 import 期呼叫 `exit()`** → SystemExit 直接炸掉整個 pytest 收集
  （`no tests ran` + INTERNALERROR）。任何全套 `pytest test/` 都會被它擋死。屬必須處理的測試基礎建設債（記 D / B6）。

## 仍需 Harrison / 實機確認
- 本修復消除了「交接洩漏→連不上」這條最吻合的路徑，但「退出硬體卡死」是否還有其他成因（如 UI 執行緒等待、
  Teensy 端行為），需以實機「進入→啟用→停止→再進入」循環復現確認。
- 是否併入 main / push 由 Harrison 決定。
