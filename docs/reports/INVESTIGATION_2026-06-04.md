# 線索調查報告 — 2026-06-04（夜間自動任務）

> 針對 OPEN_THREADS A 區的 read-only 調查。結論記於此，未改任何程式碼。

## 1. command_4d 第 4 維語義 + recoil 模型本質（重大發現）

訓練 `joystickwithgun.py` 的 `command_config`：
```python
a=[0.0, 0.0, 0.0, 0.5],   # 註解標 #Vx,Vy,Omega（只標了前3個）
b=[0.9, 0.25, 0.5, 0.7],
# sample_command: cmd = uniform(shape=4)*2a - a  → 取值 [-a, a]
```

**關鍵**：振幅 `a` 的前三維（vx, vy, omega）= **0** → 訓練時運動指令**恆為零**；
只有**第 4 維**在 `[-0.5, 0.5]` 變動。

推論：
- **recoil 模型 (`fire_on_recoil_51`) 不是運動模型，而是「原地站立 + 吸收後座力」模型。**
- 在 sim/硬體餵非零的 vx/vy/omega 給它 = **超出訓練分布**（行為未定義）。
- 第 4 維（sim 稱 "target pitch"）= **目標側傾/lean，已確認**。
  訓練 `_reward_tracking_pitch`：`pitch_error = (command[3] - current_pitch)²`，
  而 `current_pitch` = `-arcsin(up_vector[1])` = 側向(roll)。
  → **cmd[3] 是「目標側傾(roll)」的追蹤目標**（±0.5 rad），不是俯仰。
  sim/硬體把它標成 "target pitch" 同樣是 roll/pitch 命名混淆的延伸。

## 2. recoil 機制（解開 roll/pitch 的物理原因）

訓練 `firearm_recoil` config：
```python
enable=True, interval_range=[50,200], warning_duration=3, duration=0.2,
direction=[0.0, 1.0, 0.0],          # +Y 局部座標 = 側向！
force_scale_range=[4.1, 4.3]
```

- **後座力施加在 +Y（局部側向）** → 把狗往側邊推 → 產生的是 **roll（側傾）**。
- 這就是為何 `current_pitch` 觀測取 `up_vector[1]`（側向分量）= roll：模型要感知並抵抗側向後座力。
- **roll/pitch 命名混淆 + 物理機制 至此完全自洽**。

### 真實硬體上的 recoil warning 如何觸發？
- `recoil_warning_controller.py`（24 行）：只有**手動** trigger/reset（事件
  `EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED`），由 UI 發起。
- `config.yaml` `firearm_recoil_warming.auto_inhibit: true` → **自動預警關閉，只留手動**。
- 結論：**硬體上沒有接實體擊發感測器**；warning bit 是人工按鈕給策略的「即將開火」提示。
  真實後座力來自實體槍，warning 只是預告旗標。sim 端則是用 `recoil_timer/interval` 模擬整個物理後座力。

## 3. e2e vs e2e_fixed 模型

| 檔 | 大小 | md5 |
|----|------|-----|
| `pupper_ppo_policy_e2e_101580800.onnx` | 787348 | ef4abd… |
| `..._101580800_fixed.onnx` | 787348 | 8a9cf6… |

- 大小相同、**md5 不同** → 是**不同權重的模型**（非重複檔）。
- 來源：sim git 有 commit `27cd2b2 匯入新onnx {子元0908:修復}` → 組員「子元」在 09/08 重匯的修正版。
- 「fixed 修了什麼」**無法從二進位判斷**，需 netron 看圖差異或問子元/查訓練 notebook。
- 實務建議：兩個都留著，A/B 在 sim 裡跑比較行為差異即可（PolicyManager 支援多模型平滑切換）。

## 4. 「teensy 輸出問題」線頭

- 出處：sim commit `3e61709 修復進出硬體閃退問題 增加檢查確實交接 目前問題teensy輸出問題`。
- 解讀：在修好「進出硬體閃退 + 確認控制權交接」之後，**殘留**一個 Teensy 輸出（遙測）問題。
  關聯 commit `3c71229 log 的 teensy 回應`。
- 與硬體穩定性高度相關，詳見 `REVIEW_hardware_stability_2026-06-04.md`（其中發現一個 stop 流程的疑似 self-join bug，
  可能就是交接後不穩的來源之一）。

## 待後續（需訓練 notebook 或實體機）
- cmd[3] 的 reward 用途（`tracking_pitch`）→ 確認第 4 維到底命令什麼物理量。
- e2e_fixed 的實際差異。
- recoil 模型在非零運動指令下的 OOD 行為（可在 sim 實驗）。
