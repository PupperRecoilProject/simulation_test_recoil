# 韌體 pupper_recoil 盤點 + smell sweep — 2026-06-07（T02-5，純讀）

> 對象：`../../pupper_recoil`（Teensy 4.0 / PlatformIO / C++）。Harrison 主責 repo。
> 目的：指令集 inventory + 設計氣味掃描，供跨 repo 理解與重構參考。序列埠 34 欄位契約權威仍在 INTERFACE_CONTRACT。

## 建置環境（platformio.ini）
- board `teensy40`，framework `arduino`，**`monitor_speed = 921600`**（與 INTERFACE_CONTRACT 一致；README/舊記載 115200 過時）。
- lib_deps：`SparkFun LSM6DSO`(IMU 6軸)、`ArduinoJson@7`、`BasicLinearAlgebra@5`。

## 模組結構（include/ + src/）
| 模組 | 職責 |
|---|---|
| `RobotController` | 頂層協調（狀態、迴圈） |
| `MotorController` | 12 馬達控制（DJI C610 / CAN，見 lib/DJIC610Controller） |
| `CommandHandler` | 序列埠指令解析與分派 |
| `TelemetrySystem` | 遙測輸出（對應 PC 端 34 欄位串流） |
| `AHRS` / `LSM6DSO_SPI` | 姿態估算（Mahony）+ IMU SPI 驅動 |
| `homing_main.cpp` | 主進入點（注意：檔名 homing，疑為歸零/主程式合一） |
| lib `DJIC610Controller` | DJI C610 馬達 + FlexCAN_T4 CAN 匯流排 |
| lib `MahonyAHRS` | Mahony 姿態濾波 |

## 指令集 inventory（CommandHandler.cpp 解析）
頂層 action：`move` / `stand` / `zero` / `set` / `get`(=`status`) / `monitor` / `focus` / `cal` / `stop` / `reboot` / `raw` / `test`
- **move 目標**：`m<id>`(單馬達) / `g<name>`(群組) / `gl<id>`(腿群組 leg) + `<h> <u> <l>`(三軸角)
- **monitor 子指令**：`freq <hz>`(遙測頻率) / `focus m<id>` / `stop`
- **raw**：`raw m<id> <mA>`（直接給電流，除錯用）
- **test**：`test wiggle m<id>`（單馬達擺動測試）
- 對應 PC 端：`move all` 批量、`monitor freq/p/h` 等（INTERFACE_CONTRACT §2 指令協定權威）。

## 設計氣味（sweep，供參考；韌體決策權在 Harrison）
- 🟡 主檔名 `homing_main.cpp`：歸零邏輯與主程式可能混在一起，命名不直觀。
- 🟡 關節命名：韌體用 hip/upper/lower（h/u/l），與 sim 的 Abduction/Hip/Knee 不同義 → 跨 repo 溝通用軸序/ID（GLOSSARY 已記）。
- 🟡 `raw`/`test wiggle` 等除錯指令與正常控制指令同層 → 實機操作時需小心誤觸（無模式隔離）。
- ℹ️ 馬達方向不在韌體翻轉，全靠 sim `correction_vector` 補償；手動 `move` 方向與 AI 相反（INTERFACE_CONTRACT §3 已記）。
- ℹ️ AHRS 送出的是 roll（sim 端標 pitch）→ 已知歧異，待實體驗證正負號（INTERFACE_CONTRACT §4）。

## 結論
韌體結構清晰、模組分工明確（優於 sim 端的混沌）。指令集完整且涵蓋單馬達/群組/除錯。
主要待辦屬「跨 repo 命名一致性」與「除錯指令隔離」，非急迫。深入重構待實體機台在場時再做。
