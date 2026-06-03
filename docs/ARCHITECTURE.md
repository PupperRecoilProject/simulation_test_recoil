# simulation_test_recoil 架構總覽

> 高階控制軟體的 onboarding 文件。目的：讓人（或 AI）快速理解資料怎麼流、各模組職責。
> 對應版本：main (v4.14.3)。最後更新 2026-06-04。

## 一句話
一套 Python 程式，同時驅動 **MuJoCo 模擬**與**實體 Teensy 硬體**；中間用 **ONNX 策略模型**決策。
模擬與硬體共用同一條 AI 推論路徑——靠「中央狀態 + 標準化觀測」把資料源抽象掉。

## 入口
- **`main_nicegui.py`** — 唯一入口。NiceGUI 網頁 UI、多執行緒。支援 `--no-sim`（無頭硬體模式，用 `src/mock/`）。
- （`main.py` 輕量 GLFW 版已於 v4.14.2 刪除。）

## 核心解耦三件套（`src/core/`）
1. **`event_system.py`** — 執行緒安全的事件匯流排（單例 `event_bus`）。模組間只透過 publish/subscribe 溝通，
   事件名稱集中定義（`EVENT_*`）。請求類（UI→邏輯）、通知類（邏輯→UI）、資料類（tick）。
2. **`state.py` `SimulationState`** — **應用級單一真相來源 (SSoT)**。
   - 訂閱 tick/command/mode 事件被動更新；持有所有 `raw_*`（原始感測器）與 `std_obs`（標準化觀測）。
   - 也持有各模組參考（`*_ref`）作為全域上下文。所有共享寫入都用 `self.lock`。
3. **`logger.py`** — 等級語義：debug(高頻診斷,預設關)/info(關鍵狀態)/warning(可恢復)/error(失敗)。

## 資料流（單向）
```
輸入(鍵盤/搖桿/UI) → 發布事件 → Controller 處理 → 更新中央 State → 渲染 UI
                                          ↓
   資料源(模擬 或 硬體) 寫 state.raw_* → ObservationManager 算 std_obs → PolicyManager 讀 std_obs 推論 → 動作
```
**關鍵**：PolicyManager 從 `state.std_obs` 讀資料，**不知道**來源是模擬還是硬體 → sim-to-real 的乾淨抽象。

## 控制器（`src/controllers/`）
- **`simulation_controller.py`** — 模擬主迴圈驅動者。推進 MuJoCo、寫 `raw_*`、發 tick、套用控制。
  狀態機由 `state.control_mode` 等旗標驅動。
- **`hardware_controller.py`** — 硬體主迴圈驅動者（執行緒 + 狀態機 STOPPED/STARTING/RUNNING/STOPPING/FAILED）。
  讀 Teensy 序列埠（34 欄位）寫 `raw_*`、跑 AI、送 `move all`。含安全熔斷與頻率監控。
  ⚠️ 穩定性疑點見 `reports/REVIEW_hardware_stability_2026-06-04.md`。
- **`ui_controller.py`** — NiceGUI 介面邏輯；UI 呈現層 SSoT 是其 `_label_descriptors`。
- **`recoil_warning_controller.py`** — 後座力預警旗標（手動 trigger/reset）。
- **`global_keyboard_driver.py`** — 全域鍵盤。

## 觀測與策略（`src/simulation/observation_manager.py`、`src/hardware/policy.py`）
- **ObservationManager** — 觀測層 SSoT (`ALL_OBS_DIMS`)。每幀把 `raw_*` 依各模型「配方(recipe)」
  算成 `std_obs`。模式感知：硬體模式直接信任 Teensy 的重力/角速度/pitch；模擬模式用四元數計算。
- **PolicyManager** — 載入 `config.yaml` 所有 ONNX 模型，各自維護觀測歷史(deque)，
  支援兩模型間**平滑線性融合**切換。模型+配方由 config 定義。

## 控制模式（`state.control_mode`）
`WALKING`(AI) / `FLOATING`(懸浮) / `JOINT_TEST` / `MANUAL_CTRL` / `HARDWARE_MODE` / `SERIAL_MODE`。

## 模擬專屬（`src/simulation/`）
`simulation.py`(MuJoCo 介面) / `rendering.py`(疊層) / `terrain_manager.py`(地形) / `floating_controller.py`。
資源在 `assets/`（XML 場景 + STL mesh），模型在 `models/`。

## 硬體專屬（`src/hardware/`）
`serial_communicator.py`(連線/讀取執行緒/控制權交接) / `teensy_api.py`(指令協定封裝) / `xbox_controller.py`。
跨 repo 介面細節見 `INTERFACE_CONTRACT.md`。

## 設定
`config.yaml` 是調校中心：ONNX 模型庫 + 各模型觀測配方、PID/級聯增益、地形參數、後座力、馬達校正向量。

## 想改東西時看哪裡
| 要改 | 看 |
|------|----|
| 加觀測元件 | `observation_manager.py` 的 `ALL_OBS_DIMS` + generator；config recipe |
| 加模型 | `config.yaml` `onnx_models` |
| 改控制流程/模式 | 對應 controller + `state.control_mode` |
| 改 UI | `ui_controller.py` |
| 改 Teensy 溝通 | `teensy_api.py` + `INTERFACE_CONTRACT.md` |
| 加事件 | `event_system.py` 定義 `EVENT_*` |
