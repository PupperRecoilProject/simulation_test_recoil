# git 歷史考古 — 找「設計過又壞掉/被回退」的功能（2026-06-04，B2，純讀）

> 對象：sim repo git log（346 commits）。目的：找被改壞、半途而廢、或反覆來回的功能線索，供重構參考。
> 方法：grep commit 訊息 revert/還原/壞/移除/暫時/disable + 追關鍵檔案時序。皆未改碼。

## 🔴 A1 放棄的模式系統遷移：OperatingMode/ControlSubMode → 退回 control_mode 字串
- `a8a723e fix: update sim controller for new modes`：訊息明載「migrate to **OperatingMode/ControlSubMode** 架構,
  移除舊 control_mode 依賴」＋「add **legacy control_mode property** to SimulationState 以相容舊模組」。
- **但現行碼 grep `OperatingMode/ControlSubMode` 完全無匹配** → 這套新列舉架構**後來被整個移除/回退**，
  最終存活的是「過渡相容用」的 `control_mode` 字串（即今日 FEATURE_INVENTORY §2 那組字串模式）。
- 周邊 commit 透露混亂：`8644028 深度重構 state 與 hardware controller` → `466356a 加回 SimulationState 便利方法`
  → `7091c3c fix: 加回 terrain 狀態 to avoid crash`（多個「加回 X 避免 crash」＝重構當下把功能改壞再補回）。
- 多來自 `codex/*` 分支（自動化/外部產生），與 Harrison 描述「混沌、無紀律」相符。
- **重構啟示**：模式系統曾想做成乾淨列舉（方向對），但執行失敗被回退。重構時應**重做成乾淨的 enum 狀態機**
  （ARCHITECTURE_PROPOSAL 的 core/input + state），並從這次失敗學到「要先有測試護網再動」。

## 🟡 A2 「多執行緒重構＝完全穩定」的宣稱後來被打臉
- `71cf47c feat(core): 完成多執行緒重構，達成系統完全穩定`（粗體宣稱）。
- 但其後仍有大量硬體「進出閃退/AI 暫停卡死/離開卡死」修補（OPEN_THREADS item 2，+251 行），
  且本次 T3b 才修掉序列埠交接洩漏 → **「完全穩定」是過早樂觀**。提醒：穩定性需測試佐證，不靠宣稱。

## 🟡 A3 物理/地形參數反覆來回（tuning churn）
- `9f71228 移除地形平坦化濾波器 restore crisp pyramid`、`bf8e7b4 恢復求解器設定…revert solver iterations and timestep`、
  `2c09d90 整合 cascade config 與動態 control dt`、PR#11 `update-solver-parameters`。
- 求解器/timestep/地形濾波被改了又還原 → 物理參數缺乏單一可信來源與紀錄。重構時應把物理/求解參數集中進 config 並記錄理由。

## 🟡 A4 啟動序列移除 'stop' 指令（已知，列此存證）
- `777acdc 移除硬體模式的 stop 指令`：對應 `_execute_start` 內 v4.14.3 註解（為從站姿無縫進入硬體、避免掉電癱軟）。
  屬刻意設計，非 bug；但前提是「Teensy 已在已知安全狀態」，重構時需明確化此假設。

## 🟡 A5 legacy 腳本反覆增刪
- `fd1c44f remove legacy main & pyserial scripts`、`998155a pyserial 還原`、`dc3321e 刪除pdf`、`9302918 刪除 .pyc`。
- pyserial 腳本被刪又還原 → 邊界不清。配合 B3 死碼偵測一併釐清哪些 test/ 腳本仍有用。

## 小結
最值得記住的是 **A1**：今天的 `control_mode` 字串模式系統，其實是一次「想做成乾淨列舉但失敗被回退」的殘存物。
這正是「將就設計」的活標本——重構模式系統時要避免重蹈覆轍（先測試、再動、一次做乾淨）。
已將 A1/A2 線索回連 REFACTOR_SCOPE 與 OPEN_THREADS。
