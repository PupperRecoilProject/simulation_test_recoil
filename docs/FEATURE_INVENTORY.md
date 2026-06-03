# 功能盤點（Feature Inventory）— simulation_test_recoil

> 入口見 `README.md`。本檔＝重構前「現有功能」的完整快照式清單（活文件，隨功能變更更新）。
> 目的：重構/替換舊機制前，先把「現在到底有哪些功能」記清楚，避免重寫時遺漏。
> 權威來源：程式碼實測（T2 啟動）＋靜態通讀（keyboard/xbox handler、ui_controller、state、hardware_controller）。
> 行為正確性不在本檔背書（部分功能可能已被改壞，見 REFACTOR_SCOPE / OPEN_THREADS）。

## 1. 執行模式（啟動旗標）
| 啟動方式 | 說明 |
|---|---|
| `python main_nicegui.py` | 預設：MuJoCo 模擬 + NiceGUI 網頁控制台（port 8080） |
| `python main_nicegui.py --no-sim` | 無頭模式：用 `src/mock` 假元件取代模擬（MockSimulation/MockObservationManager…），供無模擬除錯 |

啟動時會**無條件**另起 `tree_demo_server.py`（NanoOwl 影像伺服器）子程序——缺 cv2/nanoowl 即崩，但不影響主程式（見 REFACTOR C-S2）。

## 2. 控制模式（`state.control_mode`，狀態機）
鍵盤/UI 切換；彼此互斥。預設 `WALKING`。
| 模式 | 進入 | 功能 |
|---|---|---|
| `WALKING` | 預設 / UI「走路」/ 鍵盤模式退出 | AI 策略走路，吃 4D 指令 `[vy, vx, wz, pitch]` |
| `FLOATING` | UI「懸浮」/ 鍵盤 T 或 F 切 float 旗標 | 機器人被懸浮控制器固定，用於觀察/調參 |
| `HARDWARE_MODE` | UI「硬體」(需序列埠已連) / 鍵盤 H (需 hw 已啟動) | 接管實體 Teensy，AI 控制實體馬達 |
| `JOINT_TEST` | 鍵盤 G / UI「關節測試」 | 逐關節加偏移測試（不送運動指令），`[`/`]` 選關節、↑↓ ±0.1、C 歸零 |
| `MANUAL_CTRL` | UI「手動控制」 | 離散步進手動下指令；含調參、float 切換、序列埠子模式入口 |
| `SERIAL_MODE` | 鍵盤 `` ` ``(GRAVE) | 直接打字送序列埠指令（Enter 送出、Backspace 刪除、`` ` `` 返回） |

## 3. 鍵盤綁定（`keyboard_input_handler.py`，指令向量 `[vy, vx, wz, pitch]`）
**通用 / 預設模式**
- `G`→JOINT_TEST、`H`→HARDWARE_MODE、`T`→切懸浮旗標、`M`/`Tab`→切 UI 頁
- `L`→連接 Xbox、`O`→切換地形、`N`→單步、`1~9/0`→選策略索引 0~9
**MANUAL_CTRL 模式**
- 移動：`W/S`→vx、`A/D`→vy、`Q/E`→wz、`I/K`→pitch、`C`→清零（僅 PRESS 各加一步）
- `F`→切懸浮、`` ` ``→SERIAL_MODE、`Esc`→關閉、`R`→硬重置、`X`→軟重置
- `Y`→重生地形(僅無限地形)、`P`→存高度場 PNG、`Tab`→切頁
- 調參：`[`/`]`選參數、↑↓調值（保留 REPEAT）
**JOINT_TEST 模式**：`G`退出、`[`/`]`選關節、↑↓±0.1、`C`歸零
**SERIAL_MODE**：可列印字元入緩衝、Enter 送出、Backspace 刪、`` ` ``返回

## 4. Xbox 手把（`xbox_input_handler.py`）
- 左搖桿 X→vy、左搖桿 Y→vx(反向)、右搖桿 X→wz；pitch 固定 0（手把不調 pitch）
- 靈敏度來自 `config.gamepad_sensitivity`
- `Select`→硬重置、`L1`/`R1`→切調參項、（方向鍵/另一軸）→調參值

## 5. NiceGUI 網頁 UI（`ui_controller.py`，port 8080）
**主控制面板**：模式按鈕（走路/懸浮/硬體[綁定序列埠連線]/關節測試/手動控制）、暫停切換、步進(N)、軟/硬重置。
**資料捕獲**：指定時長捕獲 + 手動停止並存檔（`policy_manager.start/stop_data_capture`）。
**策略 & 地形選擇器**：下拉選 ONNX 策略、下拉選地形。
**調參滑桿**：kp / kd / action_scale / bias。
**裝置面板**：啟用馬達開關、AI 啟用/停用開關(發 `EVENT_HARDWARE_AI_TOGGLE_REQUESTED`)、連接序列埠(U)、連接搖桿(J)、退出程式。
**搖桿面板**：顯示 + 清除命令。
**關節控制面板**：懸浮開關、選關節、滑桿(±π)、±0.1/歸零。
**核心數據儀表板**：向量網格顯示（gyro/gravity/accel/joint…）。
**即時視覺 (NanoOwl)**：prompt 辨識輸入 + 影像（依賴 tree_demo_server）。
**ONNX 觀察向量顯示**：即時 obs 向量。
**互動式控制台**：序列埠輸入框 + 發送；快捷鈕 `cal`/`stop`/`stand`/`zero`、`monitor h`/`monitor p`、`monitor freq <N>`(下拉選頻率)。
**操作手冊**：`~` 按鈕顯示 help。

## 6. 模擬功能
- **地形**（`terrain_manager.py`）：`INFINITE`(無限程序生成，可重生/存 PNG) 與 `FLAT`，鍵盤 O 或 UI 切換；
  生成參數見 `config.TerrainGenerationConfig`（sine/steps/noise/pyramid…）。
- **懸浮控制器**（`floating_controller.py`）：固定式 mocap 懸浮（target_height + 姿態 PD）。
- **重置**：軟重置（保持地形/位置）、硬重置（完整重置）。
- **暫停 / 單步**：可暫停物理並逐幀步進（N）。
- **後座力預警**（`recoil_warning_controller.py` + `firearm_recoil_warming` obs）：開火前預警，配合 recoil 模型維持平衡。

## 7. AI 策略（`policy.py` PolicyManager）
- 多 ONNX 模型載入 + 預熱；數字鍵/UI 切換；切換時平滑混合（`policy_transition_duration`）。
- 各模型有「配方(recipe)」決定 obs 組成與歷史長度（base dim 48 或 51；含/不含 `current_pitch`/`commands_4d`/`firearm_recoil_warming`）。
- T2 實測載入的 9 個模型：stable_walk، agile_model، new_high_level، new_e2e،
  fire_on_recoil_51، fire_on_recoil_51_fixed، sim2real_v1_flat_0911، sim2real_v1_rough_0911، sim2real_v2_0917。
- 資料捕獲：可錄推論輸入/輸出存 CSV。

## 8. 硬體控制（`hardware_controller.py` + `teensy_api.py` + `serial_communicator.py`）
- 狀態機：STOPPED→STARTING→RUNNING→STOPPING；錯誤→FAILED；斷線→CONNECTION_LOST(熔斷，需重連)。
- 啟動序列：握手取得序列埠控制權 → `monitor freq <control_freq>` → `monitor p`(POLICY_STREAM)；
  成功後預設 **MUTED + AI 關閉**（預設安全，待 UI 啟用）。
- AI 控制迴圈：固定時間步長（time_accumulator + MAX_FRAME_TIME 防螺旋死亡）；
  送馬達指令前做 Sim↔HW 方向校準（`correction_vector`，見 INTERFACE_CONTRACT）。
- 序列埠指令（經 TeensyAPI / 互動台）：`move all`、`monitor freq/p/h`、`stop`、`cal`、`stand`、`zero`。
- 連線參數 baud 921600（見 INTERFACE_CONTRACT）。
- 頻率監控：資料接收頻率 / AI 決策頻率（deque + `_freq_lock`）。

## 9. 架構基礎設施
- **事件匯流排**（`event_system.py`）：單例 `event_bus`，發布/訂閱解耦（見 ARCHITECTURE）。
- **中央狀態**（`state.py` `SimulationState`）：應用級 SSoT；含 `raw_*`(sim/hw 共用 AI 路徑)、各種旗標與引用。
- **ObservationManager**：觀測層 SSoT（`ALL_OBS_DIMS`）。
- **ORT provider**（`ort_provider.py`）：選 onnxruntime execution provider（T2 實測選 CPU）。
- **Logger**（`logger.py`）：debug/info/warning/error 分級。

## 10. 已知「不完整 / 待釐清」功能（指向其他文件，不在此重複）
- NanoOwl 視覺 = 與機器狗核心無關、硬編在主入口（REFACTOR C-S2）。
- roll/pitch 與 Y-前後 非標準軸向（INTERFACE_CONTRACT §4 / REFACTOR B）。
- recoil 模型 OOD（站立模型餵非零運動指令，REFACTOR 待釐清）。
- 啟動需 UTF-8 否則崩（REFACTOR C-S1）。
- 退出硬體/交接洩漏（已修於分支 fix/hardware-stop-selfjoin，待 review）。
