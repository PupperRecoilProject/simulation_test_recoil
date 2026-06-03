# 未展開的線索 / 調查待辦 (Open Threads)

> 記錄「有線索但還沒深入查」的項目，避免 compact / 換 session 後遺失。
> 已釐清的結論進 INTERFACE_CONTRACT.md；計畫進度進 PROJECT_PLAN.md；本檔專放「待挖的坑」。
> 最後更新：2026-06-04。

## A. 未展開的線索（read-only 可查，多數不需實體機）
> 📋 2026-06-04 夜間自動任務已調查 1/3/4/5：見 `reports/INVESTIGATION_2026-06-04.md`。
> 硬體穩定性(項2)見 `reports/REVIEW_hardware_stability_2026-06-04.md`（發現疑似 self-join bug）。
1. **「teensy 輸出問題」** — sim `dev4.13` 最後一個 commit 留言，從未查明是什麼。
   線索：可能與 roll/pitch、或遙測在高頻下的穩定性/掉幀有關。
2. **硬體進出穩定性** — `hardware_controller.py` 有 +251 行在修「進出硬體閃退 / AI 暫停卡死 / 離開後卡死」。
   序列埠控制權交接 + 執行緒收斂可能仍脆弱，未實測。值得靜態審查 `_execute_start/_execute_stop/shutdown` 的鎖與 join 邏輯。
3. **e2e vs core+normalizer** — 部署用 `*_e2e_*.onnx`（正規化已 fuse 進圖）；但 sim 另有
   `pupper_ppo_policy_e2e_101580800_fixed.onnx`，"fixed" 到底修了什麼未知。可比對兩個 onnx 的差異。
4. **recoil warning 機制** — `recoil_warning_controller.py` 只有 24 行；sim 用 `recoil_timer/recoil_interval` 模擬預警。
   真實硬體上「開火預警」如何觸發？有無實體扳機/訊號源？尚未釐清端到端路徑。
5. **command_4d 第 4 維語義** — 名為「target pitch」，但既然 pitch 實為側向(roll)，這維到底命令什麼？
   需對訓練 `joystickwithgun.py` 的 `sample_command` / command_config 核對（a/b 是 4 維）。
6. **訓練變體取捨** — `joystickwithgun.py`(現役 recoil) vs `joysticks_sac.py`(SAC) vs `*-複製`。
   哪個是現役、SAC 路線是否值得評估。
7. **地形模型對應** — sim 新增 `0911_FlatTerrain/RoughTerrain.onnx`(sim2real_v1) 與訓練的
   `scene_mjx.xml`(flat) / `hfield_mjx.xml`(rough) 的對應關係。

## B. 已識別、刻意延後（多數待實體機）
- **sim 模擬模式 `_get_current_pitch` 修正**：改用側向分量對齊訓練（訓練 = `-arcsin(up_vector[1])`）。
  *先別改*，待實體確認 Teensy AHRS roll 正負號後一起做（否則可能改錯方向）。
- **過時文件修正**：fw README baud 115200、platformio `monitor_speed=460800`（實際 921600）；
  sim `readme.md` 指向已刪除的 `tennsy.md` 破連結。低風險、隨時可做。
- **版本號註解雜訊清理**：移除程式碼裡的 `【vX.X.X 修改】` 行內標籤，改靠 git。

## C. 必須實體機器狗在場才能做
- Teensy AHRS **roll 正負號 / 軸向** 對 MuJoCo `up_vector[1]` 的驗證。
- **角速度**正負號/軸對應（座標系已確認一致）。
- **加速度**正負號/軸（單位 m/s² 已一致；當時「對應過」但細節未留存）。
- 序列埠交接/熔斷在真硬體上的穩定性實測。

## 關鍵已知結論（防遺失，詳見 INTERFACE_CONTRACT.md）
- roll/pitch：訓練 `current_pitch` 實為側向(roll)；**韌體送 roll 是對的**，歧異在 sim 模擬模式 pitch 公式。
- 馬達順序：訓練/韌體/sim 三方一致（FR,FL,RR,RL × hip,upper,lower）。
- 部署 recoil 模型 = `pupper_ppo_policy_e2e_101580800.onnx`，來自 `joystickwithgun.py`，PPO ~1.016 億步。
