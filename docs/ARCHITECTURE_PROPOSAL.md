# 重構架構提案（DRAFT）— headless 控制核心 + 薄客戶端

> 草稿，待 Harrison 校對。入口見 `README.md`。
> 這份綜合 T4–T8 的盤點/審查結論，提出目標架構與**漸進式（strangler-fig）遷移**路線。
> 前置決策（已採納，見 REFACTOR_SCOPE「目標架構決策」）：**headless 控制核心 + 可替換薄客戶端**。
> 本文不改任何程式碼，只給設計與步驟。

## 0. 為什麼要重構（一句話動機，對應已發現問題）
現況把「控制邏輯／狀態／時間／UI／硬體」糾纏在一起，導致：非 UTF-8 起不來(C-S1)、雙鍵盤系統(C2-1)、
非單一時鐘(C3-1)、非重入中央鎖地雷(C4-1)、terrain_cache 洩漏(C5-1)。這些都源於**缺乏清楚的層邊界**。
目標：把「核心」抽成乾淨、可測試、無 UI 依賴的 Python API，讓 UI/CLI/測試都只是它的客戶端。

## 1. 目標分層
```
            ┌─────────────────────────────────────────────┐
            │            Clients (薄、可替換)               │
            │  CLI(Typer)  │  Web(FastAPI+WS+HTML)  │ 測試 │
            └───────────────┬─────────────────────────────┘
                            │  只透過 Core API（函式呼叫 / 事件）
            ┌───────────────▼─────────────────────────────┐
            │                Control Core (headless)        │
            │  ┌─────────┐ ┌──────────┐ ┌───────────────┐  │
            │  │ Domain  │ │ Policy   │ │ Clock/Scheduler│  │
            │  │ State   │ │ Manager  │ │ (單一時鐘)     │  │
            │  └─────────┘ └──────────┘ └───────────────┘  │
            │  ┌──────────────┐  ┌──────────────────────┐  │
            │  │ SimBackend    │  │ HardwareBackend       │  │
            │  │ (MuJoCo)      │  │ (Teensy serial)       │  │
            │  └──────────────┘  └──────────────────────┘  │
            └───────────────────────────────────────────────┘
```
- **Control Core**：純 Python，無 nicegui / glfw / 視窗依賴。提供明確 API（如 `core.start_sim()`,
  `core.set_command(vx,vy,wz,...)`, `core.select_policy(name)`, `core.step()`, `core.get_observation()`,
  `core.attach_hardware(port)` …）。可被 import、被測試、被任何客戶端驅動。
- **Backends**：`SimBackend`(MuJoCo) 與 `HardwareBackend`(Teensy) 實作同一介面（`reset/step/read_state/apply_ctrl`），
  讓「sim 與硬體共用 AI 路徑」變成乾淨的多型，而非現在散落的 if/else 與 `raw_*` 共用慣例。
- **Clients**：CLI 優先（利於我自動測試/操作）、Web 次之（取代 NiceGUI）。UI 只呼叫 Core API + 訂閱通知。

## 2. 模組邊界與單一職責（對照現有檔案）
| 目標模組 | 取代/吸收現有 | 關鍵職責 | 修掉的問題 |
|---|---|---|---|
| `core/state`（純資料 + 加鎖存取器） | `state.py` | SSoT，**RLock 或 `_locked` 分層** | C4-1 非重入鎖地雷 |
| `core/clock` | run() 內的時間邏輯 | **單一模擬時鐘**：AI 綁物理步數 | C3-1/2 雙時鐘 |
| `core/input`（單一輸入抽象：事件源→指令） | keyboard_input_handler + global_keyboard_driver | 一套綁定，多來源(GLFW/web/xbox)轉同一指令事件 | C2-1 雙鍵盤、C2-2 直接改 state |
| `core/policy` | policy.py | 模型載入/混合/推論 | （大致可留，介面化） |
| `backend/sim` | simulation.py + simulation_controller 物理部分 | MuJoCo step/render 分離 | 時間/鎖耦合 |
| `backend/hardware` | hardware_controller + serial + teensy_api | 序列埠交接、狀態機 | 已修交接洩漏(T3b) |
| `io/logging` | logger.py + 所有 `print` | 全面 `log`，UTF-8 安全 | C-S1、C2-5 |
| `clients/cli`、`clients/web` | main_nicegui + ui_controller | 薄客戶端 | NiceGUI 耦合、NanoOwl 硬塞(C-S2) |

## 3. Strangler-fig 漸進遷移（不大爆炸重寫，全程 main 可動）
> 原則：每步都在綠色測試後合併；舊路徑與新核心並存，逐段「絞殺」舊碼。
1. **建特徵測試護網**（先做）：對現有可觀測行為補 characterization tests（已起頭：`test/test_hardware_handoff.py`）。
   先涵蓋 obs 組裝、policy 推論 I/O、模式切換狀態機、序列埠交接。→ 重構靠它防回歸。
2. **抽 Core API 外殼**：新建 `core/` 介面層，內部**先委派**給現有 controllers（adapter）。客戶端改打 Core API。
3. **逐模組內遷**：依「副作用小→大」順序把邏輯搬進 core 並切乾淨依賴：
   先 logging(C-S1) → 輸入抽象(C2-1) → 單一時鐘(C3) → 鎖分層(C4) → terrain 有界(C5) → backend 介面化。
4. **加 CLI 客戶端**：用 Typer 包 Core API（`pup sim start`, `pup policy use X`, `pup hw connect COMx` …）。
   讓我能**全自動**驅動測試與操作（呼應終極目標）。
5. **Web 客戶端替換 NiceGUI**：FastAPI+WebSocket+單頁；資料視覺化可評估 rerun.io。NanoOwl 改為可選外掛、移出主入口(C-S2)。
6. **退役舊碼**：當某段舊 controller 不再被任何客戶端直接使用，刪除（B3 死碼流程，先報告再刪）。

## 4. 風險與守則
- 控制路徑（硬體）改動風險高 → 一律走分支 + 特徵測試 + 待 Harrison review（沿用本次 T3/T3b 流程）。
- 每步可獨立驗證、可回退；不在無測試護網下動控制碼。
- 「漂亮」優先於「相容舊習」：舊的將就設計（REFACTOR_SCOPE 全表）在對應步驟一次清掉，不再將就修補。

## 5. 與既有文件的關係
- 問題清單權威：`REFACTOR_SCOPE.md`（本提案的「修掉的問題」全部對應其編號）。
- 介面契約：`INTERFACE_CONTRACT.md`（backend/hardware 與 Teensy 的 34 欄位/指令不變）。
- 功能不遺漏：`FEATURE_INVENTORY.md`（遷移時逐項對照，確保新架構覆蓋舊功能）。
- 測試做法：`TEST_SOP.md`（配套草稿）。
