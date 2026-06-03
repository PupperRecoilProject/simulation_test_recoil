# PC ↔ Teensy 介面契約 (Interface Contract)

> 本文件定義高階控制軟體 (`simulation_test_recoil`, Python) 與即時控制韌體
> (`pupper_recoil`, Teensy 4.0 C++) 之間，透過序列埠溝通的完整契約。
> 這是兩個 repo 之間唯一的整合介面，任何一邊改動都必須同步本文件。
>
> 最後核對：2026-06-03（對應 sim `main` v4.14.3 / fw `main`）。

---

## 0. 連線參數

| 項目 | 值 | 來源 |
|------|----|----|
| Baud rate | **921600** | fw `src/homing_main.cpp:52` `Serial.begin(921600)`；sim 端一致 |
| 換行 | `\n` (LF) | 指令與資料幀皆以 `\n` 結尾 |
| 編碼 | UTF-8 (ignore errors) | sim `_read_from_port` 用 `decode('utf-8', errors='ignore')` |

> ⚠️ **過時記載**：fw `README.md` 寫 115200、`platformio.ini` `monitor_speed=460800`，
> 兩者都**不是**實際運行值。實際以 `Serial.begin(921600)` 為準。（待 Phase 2 修正）

---

## 1. 資料契約 Teensy → PC（Policy Stream, 34 欄位）

由韌體 `monitor p` 指令切換進入。每幀一行 CSV，**34 個浮點數**，逗號分隔，`\n` 結尾，無標頭。

| 欄位 index | 內容 | 維度 | 單位 | 座標系 |
|-----------|------|------|------|--------|
| 0–2   | 角速度 (gyro) | 3 | rad/s | 機身 (body) |
| 3–5   | 重力向量 | 3 | g (正規化) | 機身 |
| 6–8   | 加速度計 | 3 | m/s² | 機身 |
| 9     | ⚠️ pitch（見疑點1） | 1 | rad | — |
| 10–21 | 關節角度 ×12 | 12 | rad | 見 §3 馬達順序 |
| 22–33 | 關節角速度 ×12 | 12 | rad/s | 見 §3 馬達順序 |

**產生端**：`pupper_recoil/src/TelemetrySystem.cpp::printAsPolicyStream`
**解析端**：`simulation_test_recoil/src/controllers/hardware_controller.py::parse_policy_stream`
```python
raw_torso_angular_velocity = data_vec[0:3]
raw_gravity_vector         = data_vec[3:6]
raw_accelerometer          = data_vec[6:9]
raw_pitch_rad              = data_vec[9]
raw_joint_positions        = data_vec[10:22]
raw_joint_velocities       = data_vec[22:34]
```

> 解析端硬性要求恰好 34 欄，否則丟棄整幀並警告。
> 韌體單位轉換在送出時完成：gyro 由 dps→rad/s、accel 由 g→m/s²、角度由 deg→rad。

### 系統訊息（非資料幀）
凡是以 `[` 開頭的行（如 `[CMD]`、`[OK]`、`[ERROR]`）會被解析端當系統訊息處理、不進資料流。

---

## 2. 指令協定 PC → Teensy

封裝於 `simulation_test_recoil/src/hardware/teensy_api.py`。指令為純文字 + `\n`，不分大小寫。

| 指令 | 用途 | 回應協定 |
|------|------|---------|
| `stop` | 停止所有馬達、進 IDLE | NONE（送出即可） |
| `monitor p` | 切換到 Policy Stream 模式 | NONE |
| `monitor h` | 切回人類可讀模式 | NONE |
| `monitor freq <hz>` | 設定遙測頻率 | **OK**（等待 `[OK]`，預設 timeout 1s） |
| `move all <a0> … <a11>` | 一次設定 12 關節目標角度 (rad) | NONE |

**啟動序列**（`hardware_controller._execute_start`）：`stop` → `monitor freq <control_freq>` → `monitor p`。
**停止序列**（`_execute_stop`）：`stop` → `monitor h`。

> 安全守衛：`send_motor_commands` 只有在 `hardware_link_status == VERIFIED` 時才真正送出
> `move all`；MUTED/UNVERIFIED 下靜默丟棄但回傳成功（避免上層誤判）。

### 韌體完整指令集（手動除錯用，非 AI 流程）
`stand` / `zero` / `cal` / `move m<id> <rad>` / `move g<h|u|l> <rad>` /
`move gl<0-3> <h> <u> <l>` / `set <target> <p> <v>` / `get` / `reset` /
`monitor (h|c|d|p)` / `focus` / `reboot` / `raw m<id> <mA>` / `test wiggle m<id>`。
詳見 `pupper_recoil/README.md`。

---

## 3. 馬達編號與方向

### 編號（兩端一致 ✅）
| ID | 腿 | 關節 |
|----|----|----|
| 0,1,2 | Front-Right (FR) | hip, upper, lower |
| 3,4,5 | Front-Left (FL) | hip, upper, lower |
| 6,7,8 | Rear-Right (RR) | hip, upper, lower |
| 9,10,11 | Rear-Left (RL) | hip, upper, lower |

- 韌體來源：`pupper_recoil/src/RobotController.cpp` `manual_calibration_pose_rad` 註解。
- sim 來源：`config.yaml` `default_pose` 與 `sim2real_motor_calibration` 註解。

### ⚠️ 命名差異（易混）
| 物理關節 | sim 用詞 | 韌體用詞 |
|---------|---------|---------|
| 第1軸 | Abduction | hip |
| 第2軸 | Hip | upper |
| 第3軸 | Knee | lower |

> 「hip」這個字兩邊指**不同**關節！溝通與讀碼時務必以「第1/2/3軸」或 ID 為準。

### 方向補償（分散兩處，需小心交互）
- **韌體**：校準姿態 `manual_calibration_pose_rad` 把左腿 (FL, RL) 整組鏡像為負值。
- **sim**：`config.yaml` `sim2real_motor_calibration.correction_vector`
  `= [1,-1,-1, -1,-1,-1, 1,-1,-1, -1,-1,-1]`
  - hip：右腿 +1 / 左腿 -1（左側鏡像）
  - upper/lower：全部 -1（ONNX 正向 vs Teensy 期望正向，整體反向）

> 韌體本身的 `move` 指令**不做** ONNX→Teensy 的方向翻轉；翻轉全靠 sim 端 `correction_vector`。
> 因此手動下 `move all` 測試時的方向，會與 AI 跑的方向不同——這是常見混淆點。

---

## 4. 已知疑點（待釐清，見 PROJECT_PLAN.md Phase 2）

1. **roll/pitch — 已用訓練端原始碼釐清（2026-06-03）**

   **訓練端權威定義**（`mujoco_playground_recoil` `.../locomotion/pupper/base.py::get_pitch`）：
   ```python
   def get_pitch(self, data):
       up_vector = self.get_upvector(data)
       return -jp.arcsin(up_vector[1])   # 取上向量的 Y(側向)分量
   ```
   IMU sensor 沿用 Go1 慣例（X=前, Y=左, Z=上），故 `up_vector[1]`（側向分量）
   數學上算的是 **roll（側傾）**，只是被命名為 "pitch"。recoil 模型 (`joystickwithgun.py`)
   的 `current_pitch` 觀測學的就是這個側傾量。

   **三方比對：**
   | 來源 | 第 9 欄/`current_pitch` 實際量 | 對訓練 |
   |------|------|------|
   | 訓練（權威） | `-arcsin(up_vector[1])` 側向 = roll | — |
   | 韌體 policy stream 第 9 欄 | 送 `roll` | ✓ 一致 |
   | sim 模擬模式 `_get_current_pitch` | `arcsin(-R[2,0])` 前向 = 真 pitch | ✗ 不一致 |

   **結論（暫不改碼，待實體驗證）：**
   - 上機（硬體模式）送 roll **很可能是對的**；歧異點其實在 **sim 模擬模式的 pitch 公式**
     （用前向分量算真 pitch，但模型是用側向 roll 訓練的）。原本以為韌體錯，方向反了。
   - 仍需實體機台確認：Teensy AHRS 的 roll 正負號/軸向是否與 MuJoCo `up_vector[1]` 一致（取決 IMU 安裝方向）。
   - 「pitch」一名在三方都誤導，未來建議統一正名為 roll/側傾。

2. **方向雙重補償風險**
   韌體校準鏡像左腿 + sim correction_vector 也處理左腿，兩者是否會疊加/抵消需上機核對。

3. **加速度座標系（歷史上已對應過）**
   韌體註解標明送的是「未經座標系修正的原始 IMU 加速度」。據開發者回憶，
   加速度（欄位 6–8）的座標系**當時已在訓練端與執行端之間對應/校對過**，確認兩邊不同並做了轉換。
   （Teensy IMU 可能是 Y-fwd / X-right。）

4. **角速度座標系（座標系已確認一致，正負號待實體驗證）**
   訓練 `get_gyro` 用 `GYRO_SENSOR` = IMU local/body frame；韌體欄位 0–2 也送 body frame
   → **座標系一致 ✓**。剩餘僅軸對應/正負號需實體機台確認（當時未驗證過）。

### 訓練端權威來源（已 clone）
模型由組員以 **MuJoCo Playground** fork 訓練：`mujoco_playground_recoil`（已 clone `dev-v2.3.2` 分支於工作區根目錄）。
- recoil 環境：`mujoco_playground/_src/locomotion/pupper/joystickwithgun.py`（含 `firearm_recoil_warning`）
- sensor getter / 軸定義：同目錄 `base.py`、`pupper_constants.py`
- obs `state` 順序（`_get_obs`）：gyro(3), gravity(3), accel(3), current_pitch(1),
  joint_pos-default(12), joint_vel(12), last_act(12), command(4), recoil_warning(1)
  → 與 sim `fire_on_recoil_51` recipe 完全對應 ✓
- 部署 recoil 模型 `pupper_ppo_policy_e2e_101580800.onnx`：PPO ~1.016 億步、e2e，經 `convert_tf.py` 轉 ONNX。
觀測順序/單位/座標系慣例的**最終權威在此**。

---

## 維護
- 改動序列埠格式、欄位順序、指令、馬達映射、方向時，**先更新本文件**再改碼。
- 根目錄 `PROJECT_PLAN.md` 為跨 repo 協調樞紐，會連回本文件。
