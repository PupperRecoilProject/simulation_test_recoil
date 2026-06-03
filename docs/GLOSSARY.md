# 術語表 / 命名正規化（GLOSSARY）

> 入口見 `README.md`。本檔＝**跨三方（訓練 / sim 上位機 / Teensy 韌體）命名的權威對照**，
> 目的：消除「同字不同義、同義不同字」造成的長期混淆。溝通時一律以本表的「建議統一說法」為準。
> 契約細節（欄位/單位/符號）的權威仍是 `INTERFACE_CONTRACT.md`；本檔只管「名字」。

## 1. 關節命名（最大陷阱：「hip」兩邊不同義）
每條腿 3 個關節，由近端到遠端（身體→腳尖）：
| 軸序 | 訓練 / sim 上位機 | Teensy 韌體 | 物理意義 | **建議統一說法** |
|---|---|---|---|---|
| 第 1 軸 | Abduction（外展） | hip | 大腿左右擺（內收/外展） | **Abduction（外展）/ 第1軸** |
| 第 2 軸 | Hip（髖） | upper | 大腿前後擺（俯仰） | **Hip-pitch（髖俯仰）/ 第2軸** |
| 第 3 軸 | Knee（膝） | lower | 小腿彎曲 | **Knee（膝）/ 第3軸** |
- ⚠️ **「hip」在 sim 指第2軸、在韌體指第1軸** → 口語講「hip」必歧義。**一律用「第1/2/3軸」或馬達 ID 溝通。**
- 來源：CLAUDE.md 跨 repo 陷阱、INTERFACE_CONTRACT。

## 2. 馬達編號順序（三方一致 ✅）
12 馬達固定順序：**FR(0,1,2) → FL(3,4,5) → RR(6,7,8) → RL(9,10,11)**，每腿 ×（第1軸,第2軸,第3軸）。
- FR=右前, FL=左前, RR=右後, RL=左後。訓練/韌體/sim 三方一致（已確認）。

## 3. 馬達方向（翻轉只在 sim 端）
- 韌體**不翻轉**方向；方向補償全在 sim 的 `correction_vector`（`config.sim2real_motor_calibration`）。
- 後果：**手動下韌體 `move` 指令的方向，會與 AI（經 sim 校準）相反**。除錯手動控制時要記得。
- 來源：CLAUDE.md、hardware_controller `_apply_motor_direction_calibration`。

## 4. 姿態角：roll / pitch（已釐清的歷史誤名）
| 名稱出處 | 字面 | **實際物理量** |
|---|---|---|
| 訓練 `get_pitch` = `-arcsin(up_vector[1])` | pitch | **側向傾斜＝roll** |
| Teensy 第 9 欄（`printAsPolicyStream`）送 `roll` | 標 pitch | **roll（與訓練一致 ✅）** |
| sim **模擬模式** `_get_current_pitch` = `arcsin(-R[2,0])` | pitch | **真正的 pitch（← 歧異點）** |
- 結論：**韌體送 roll 是對的**（與訓練的「current_pitch 實為 roll」一致）；歧異在 sim 模擬模式用了真 pitch 公式。
- **建議統一說法**：訓練/契約那個量一律叫 **`lean`（側傾/roll）**，避免再用誤導性的「pitch」。
  正名與修 sim 公式待實體機確認 Teensy AHRS roll 正負號後一起做（見 OPEN_THREADS B、INTERFACE_CONTRACT §4）。

## 5. 座標軸慣例（非標準，重構時待正名）
- **Y 軸＝前後**（y+ 或 y- forward 待定）、X 軸＝左右 —— 非一般「X-forward」慣例，當時沒改。
- **後座力方向 = +Y 側向**（recoil `direction=[0,1,0]`）；recoil 模型訓練時 vx/vy/omega 恆為 0（站立吸力模型）。
- IMU 座標：observation_manager TODO 註明需與 Teensy「Y-fwd, X-right」校對（C2-8）。
- **建議**：重構時把座標慣例正規化為標準（或至少在單一處明確定義並全程一致），與 roll/pitch 正名一起做。

## 6. 指令向量（command）
- sim 鍵盤/手把端佈局：**`command = [vy, vx, wz, pitch]`**（keyboard_input_handler 明載；注意 vy 在前）。
- 第 4 維 `pitch` 實際命令的是**目標側傾(lean/roll)**，非真 pitch（呼應 §4）；recoil 模型用 `commands_4d`。
- ⚠️ 「commands_3d」(48 維模型) vs 「commands_4d」(51 維 recoil 模型) 取決於模型配方（見 FEATURE_INVENTORY §7）。

## 7. 控制模式名稱（sim 內部字串，互斥狀態機）
`WALKING` / `FLOATING` / `HARDWARE_MODE` / `JOINT_TEST` / `MANUAL_CTRL` / `SERIAL_MODE`（見 FEATURE_INVENTORY §2）。

## 8. 連線參數
- **baud = 921600**（權威值）。README/platformio.ini 舊載 115200/460800 **已過時，勿信**（部分已修，見 PROJECT_PLAN）。

## 9. 縮寫
- **FRW** = Firearm Recoil Warning（開火預警）。
- **obs** = observation（觀測向量）；**SSoT** = Single Source of Truth；**EP** = (ONNX) Execution Provider。
- **e2e onnx** = 正規化已 fuse 進圖的端到端模型（部署用）。

---
> 命名有疑義時：關節→§1（用軸序/ID）、姿態角→§4（用 lean）、軸向→§5。契約數值請回 `INTERFACE_CONTRACT.md`。
