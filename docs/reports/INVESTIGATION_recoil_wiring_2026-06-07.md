# 後座力（Firearm Recoil Warning, FRW）斷線端到端深查 — 2026-06-07（T02-7）

> 對象：UI 面板 → 事件 → 控制器 → state → ObservationManager 整條接線。回應「後座力有開關，但好像沒在動」。
> 結論先講：**自動後座力與其開關全斷，目前只有「手動觸發/重置」按鈕能動。** 純讀分析。

## 現況：三條路徑，兩條斷

### ✅ 路徑 A — 手動觸發/重置（唯一還活著的）
`ui_controller._create_recoil_warning_panel` 的「手動觸發/手動重置」按鈕
→ publish `EVENT_FIREARM_RECOIL_WARNING_TRIGGER/RESET_REQUESTED`
→ `RecoilWarningController.on_trigger/on_reset`（有 subscribe）
→ 直接寫 `state.recoil_warning_active = True/False`
→ `observation_manager.py:285` 讀此旗標餵給模型 `firearm_recoil_warming` 觀測。
**完整可動。**

### 🔴 路徑 B — 自動後座力循環（斷：函式無呼叫者）
`simulation_controller._update_recoil_warning_timer()`（含唯一的 `recoil_timer -= control_dt` 倒數 +
隨機 2.5~10s 間隔 + 0.15s 預警 + `*** RECOIL EVENT ***`）**全 repo 無任何呼叫者**。
- `run()` 主迴圈（line 168-180）只呼叫 `_perform_ai_decision()` 與 `_single_physics_step()`，**從未呼叫計時器**。
- `recoil_timer` 只在 `hard_reset`(line 479) 被重設，沒有任何路徑遞減它。
- ⇒ **自動後座力事件永遠不會發生。**

### 🔴 路徑 C — 「抑制自動預警」switch（斷：事件無 subscriber）
`ui_controller.py:646` 的 switch → publish `EVENT_FRW_AUTO_INHIBIT_SET/CLEAR`
→ **全 repo 無任何人 subscribe 這兩個事件**（grep `subscribe(EVENT_FRW_AUTO_INHIBIT` 空）。
- ⇒ 撥動 switch 在**執行期完全無作用**。`auto_inhibit` 只在路徑 B 的函式裡從 **config 檔**讀一次（而路徑 B 本身又沒被呼叫）。

## 雪上加霜：config 預設也關著
`config.yaml:247` `firearm_recoil_warming.auto_inhibit: true`，註解明載「true = 關閉自動預警，只保留手動」。
→ 即使把路徑 B 接回去，**目前 config 值也會讓它只走 inhibit 分支**（不觸發預警、只空轉計時器）。

## 應然（從程式碼意圖推斷）
1. 主迴圈每個 AI 決策 tick（= 一個 `control_dt`）應呼叫一次 `_update_recoil_warning_timer()` → 自動後座力循環運轉。
2. switch 應能在執行期切換 auto_inhibit（接 subscriber → 改一個 runtime 旗標，而非只讀 config）。
3. config `auto_inhibit` 應只當「初始值」，runtime 由 switch 覆寫。

## 修復方案（T02-8 已實作於分支 `fix/recoil-wiring`）
- **B 接回**：在 `run()` 的 AI 決策區塊，`_perform_ai_decision()` 旁呼叫 `_update_recoil_warning_timer()`
  （兩者同節奏，符合計時器「每次呼叫＝過了 control_dt」的假設）。
- **C 接回**：新增 `state.frw_auto_inhibit`（以 config 初始化）；新增 subscriber 處理
  `EVENT_FRW_AUTO_INHIBIT_SET/CLEAR` 改寫此旗標；計時器改讀 `state.frw_auto_inhibit` 而非 config。
- 加測試：計時器會遞減/觸發/重置、inhibit 旗標切換生效。
- ⚠️ config 預設 `auto_inhibit: true` 是否改為 false（讓自動後座力預設開）→ **留給 Harrison 決定**（不在程式修復內動）。

## 待 Harrison 拍板
1. 自動後座力**預設要開還是關**？（改 config `auto_inhibit`）
2. 隨機間隔 2.5~10s / 預警 0.15s 這些數字是否合理？（目前是寫死的魔術數字，可移入 config）
