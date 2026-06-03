# docs/ 文件地圖（恢復脈絡的單一入口）

> **任何 session / compact / 喚醒後，先讀這份。** 它指向所有其他文件，避免讀錯或漏讀。
> 本專案是四足機器狗重啟，協作者僅 Harrison + AI 助手。核心目標：機制漂亮、功能人性化且完善。

## 讀取順序（恢復脈絡）
1. **本檔**（文件地圖）
2. `PROJECT_PLAN.md` — 計畫表、工作流程、進度（最新狀態看這）
3. `TASK_QUEUE.md` — 自動續跑的任務佇列 + 自主執行規則
4. `REFACTOR_SCOPE.md` — 重構範圍與問題盤點（含「將就/理所當然」設計清單）
5. 需要細節時再讀：`INTERFACE_CONTRACT.md`、`ARCHITECTURE.md`、`OPEN_THREADS.md`、`reports/`
6. 跨 session 記憶（自動載入，compact 不影響）：
   `C:\Users\Harrison\.claude\projects\D--Harrison-harrison-working-code-quadruped-robot\memory\`

## 文件清單與職責（無歧義、各司其職）
| 檔案 | 職責 | 類型 |
|------|------|------|
| `README.md`（本檔） | 文件地圖 / 入口 / 命名規範 | 活文件 |
| `PROJECT_PLAN.md` | 計畫、工作流程、進度勾選、訓練流程評估 | 活文件 |
| `TASK_QUEUE.md` | 自動續跑任務佇列 + 自主執行規則 | 活文件 |
| `REFACTOR_SCOPE.md` | 重構範圍、問題盤點、將就設計清單 | 活文件 |
| `INTERFACE_CONTRACT.md` | PC↔Teensy 序列埠契約（34欄位/指令/馬達/疑點） | 活文件 |
| `ARCHITECTURE.md` | sim 架構 onboarding | 活文件 |
| `OPEN_THREADS.md` | 未展開線索 / 調查待辦 | 活文件 |
| `reports/<TYPE>_<topic>_<YYYY-MM-DD>.md` | 時間點快照（調查/審查） | 快照 |

## 命名規範（避免版本歧義）
- **活文件**：固定檔名、**不加日期**，持續更新；版本歷史交給 git（`git log`/`git blame`）。
- **快照報告**：放 `reports/`，**檔名加日期** `YYYY-MM-DD`，型別前綴大寫：
  `INVESTIGATION_*`(調查)、`REVIEW_*`(審查)。一旦寫出代表「當時的觀察」，不回頭改，要更新就出新日期檔。
- 一律放 `simulation_test_recoil/docs/`（sim 是高階協調基地，跨 repo 文件也放這）。

## 交叉引用原則
- 每份活文件開頭都應指回本地圖；重大結論在「該主題的權威文件」只寫一次，其他地方用連結，不複製。
  （權威：契約→INTERFACE_CONTRACT；計畫→PROJECT_PLAN；問題清單→REFACTOR_SCOPE；線索→OPEN_THREADS。）
