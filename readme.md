## 開發環境安裝指南 (Windows)

本指南將引導你建立一個乾淨、獨立的 Python 虛擬環境。指南分為兩部分：**核心環境** (適用於所有使用者) 和**可選的物理模擬環境** (適用於需要模擬功能的使用者)。

---

### 第一章：核心環境安裝 (基礎功能)

完成此章節後，你將能夠運行專案的基礎功能 (例如，在 `--no-sim` 模式下)。

#### 第一步：建立並啟用虛擬環境

為避免與系統全局 Python 套件產生衝突，我們為專案建立一個獨立的虛擬環境。

1.  **開啟終端機**：
    在專案的根目錄下，開啟 PowerShell 或命令提示字元 (CMD)。

2.  **建立虛擬環境**：
    執行以下指令，這將會在專案目錄下建立一個名為 `.venv` 的資料夾。
    ```powershell
    python -m venv .venv
    ```

3.  **啟用虛擬環境**：
    啟用後，你的終端機提示符前方會出現 `(.venv)` 字樣。
    ```powershell
    .venv\Scripts\activate
    ```

> **黃金準則**：啟用虛擬環境後，請**永遠使用 `python -m pip`** 來執行安裝或管理套件，以確保操作對象是當前的虛擬環境。

#### 第二步：安裝核心套件

這些是運行專案核心功能與使用者介面的基礎套件。請執行以下指令一次性安裝：

```powershell
python -m pip install onnxruntime-directml numpy scipy PyYAML Pillow pyserial nicegui pygame
```

#### 第三步：驗證 ONNX Runtime 環境

安裝完成後，執行以下指令來確認 ONNX Runtime 是否能正確使用你的 GPU (Intel/NVIDIA/AMD 皆可)。

```powershell
python -c "import onnxruntime as ort; print('ONNX Runtime Version:', ort.__version__); print('Available Providers:', ort.get_available_providers())"
```

**期望輸出：**
```
ONNX Runtime Version: 1.18.0 (或其他版本)
Available Providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
```
看到 `DmlExecutionProvider`，代表你的 GPU 加速已為核心推論功能準備就緒。

**至此，你已可以運行專案的無模擬 (`--no-sim`) 模式。**

---

### 第二章：可選：安裝 MuJoCo 物理模擬環境

若你需要運行物理模擬、訓練或與虛擬環境互動，請繼續此章節。

此安裝的核心是 `JAX` 函式庫，它負責物理計算。你需要根據你的電腦硬體，選擇 **CPU** 或 **NVIDIA GPU** 其中一條安裝路徑。

<br>

<details>
<summary><b> 👉 路徑 A：NVIDIA GPU 加速環境 (推薦，適用於配備 NVIDIA 顯示卡的桌機)</b></summary>

此路徑將利用你的 NVIDIA GPU 進行高效能物理模擬。

**前置準備：安裝 NVIDIA CUDA Toolkit**
1.  在終端機執行 `nvidia-smi`，確認右上角顯示的 `CUDA Version` (例如 `12.x`)。
2.  前往 [NVIDIA CUDA Toolkit 存檔頁面](https://developer.nvidia.com/cuda-toolkit-archive) 下載並安裝與之對應的版本 (例如 CUDA Toolkit 12.4)。

**安裝步驟**
1.  **Clone `mujoco_playground` 專案 (若尚未執行)**
    ```powershell
    git clone https://github.com/google-deepmind/mujoco_playground.git
    cd mujoco_playground
    ```
2.  **安裝 JAX (CUDA 版本)**
    ```powershell
    python -m pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
3.  **驗證 JAX GPU 後端**
    ```powershell
    python -c "import jax; print('JAX default backend:', jax.default_backend())"
    ```
    *   **期望輸出：** `JAX default backend: gpu`

4.  **安裝 MuJoCo Playground (核心依賴)**
    ```powershell
    python -m pip install -e .
    ```
5.  **最終驗證**
    ```powershell
    python -c "import mujoco_playground"
    ```
    若無錯誤訊息，即表示安裝成功。

</details>

<br>

<details>
<summary><b> 👉 路徑 B：CPU 通用環境 (適用於所有筆電及無 NVIDIA 顯示卡的電腦)</b></summary>

此路徑將使用 CPU 進行物理模擬，相容性最好。

**安裝步驟**
1.  **Clone `mujoco_playground` 專案 (若尚未執行)**
    ```powershell
    git clone https://github.com/google-deepmind/mujoco_playground.git
    cd mujoco_playground
    ```
2.  **安裝 JAX (CPU 版本)**
    ```powershell
    python -m pip install jax jaxlib
    ```
3.  **驗證 JAX CPU 後端**
    ```powershell
    python -c "import jax; print('JAX default backend:', jax.default_backend())"
    ```    *   **期望輸出：** `JAX default backend: cpu`

4.  **安裝 MuJoCo Playground (核心依賴)**
    ```powershell
    python -m pip install -e .
    ```
5.  **最終驗證**
    ```powershell
    python -c "import mujoco_playground"
    ```
    若無錯誤訊息，即表示安裝成功。

</details>

---

### 第三章：常見問題排解 (FAQ)

<details>
<summary><b>Q1: 虛擬環境啟用後，執行 `python -m pip` 提示 `No module named pip`</b></summary>

**A:** 這表示你的虛擬環境已損壞。請依照以下步驟重建：
   1.  **停用環境**: `deactivate`
   2.  **刪除舊資料夾**: `Remove-Item .venv -Recurse -Force` (在 PowerShell 中)
   3.  **回到本指南第一章**，重新建立並啟用一個全新的虛擬環境。

</details>

<details>
<summary><b>Q2: 安裝 `mujoco_playground` 時，出現 `Microsoft Visual C++ 14.0 or greater is required` 錯誤</b></summary>

**A:** 這是因為你嘗試安裝了包含 C++ 原始碼的**開發者工具** (例如 `pytype`)，但系統缺少 C++ 編譯器。
*   **最佳解法：** 確認你在安裝 `mujoco_playground` 時使用的是 `python -m pip install -e .` 而**非** `python -m pip install -e ".[all]"`。前者只會安裝運行所必需的核心依賴，完美繞過此問題。
*   **根本解法：** 前往 [Visual Studio 官網](https://visualstudio.microsoft.com/visual-cpp-build-tools/)下載並安裝 "Build Tools for Visual Studio"，並在安裝選項中勾選「使用 C++ 的桌面開發」。

</details>

<details>
<summary><b>Q3: 我有 NVIDIA 顯示卡並安裝了 CUDA，但 JAX 驗證後端仍顯示 `cpu`</b></summary>

**A:** 這通常是 JAX 官方暫時沒有提供適用於你目前 Python/Windows 版本的預編譯 CUDA 套件所致。
*   **建議：** 先接受 CPU 後端並完成安裝，確保專案可以運行。`mujoco_playground` 的多數功能在 CPU 上依然表現良好。
*   **進階方案：** 考慮使用 [WSL2 (Windows Subsystem for Linux)](https://learn.microsoft.com/zh-tw/windows/wsl/install)，在 Windows 內建的 Linux 環境中進行開發，這是 JAX 官方支援度最高的平台。

</details>


## 專案目錄結構與模組化原則
```
simulation_test_recoil/
├── assets/                     # 靜態資源 (Static Assets)
├── models/                     # AI 模型文件 (AI Model Files)
├── pdf/                        # 參考文檔 (Reference Documents)
├── src/                        # 核心原始碼 (Core Source Code)
│   ├── core/                   # 核心通用模組 (Core Common Modules)
│   ├── controllers/            # 主要控制器與邏輯協調 (Main Controllers & Logic Orchestration)
│   ├── simulation/             # 模擬環境相關 (Simulation Environment Specific)
│   ├── hardware/               # 硬體交互與底層AI推理 (Hardware Interaction & Low-level AI Inference)
│   ├── input_handlers/         # 用戶輸入處理 (User Input Handling)
│   ├── utils/                  # 通用工具函式 (General Utility Functions)
│   └── mock/                   # 模擬/測試用模組 (Mock/Test Modules)
├── test/                       # 測試與輔助腳本 (Tests & Auxiliary Scripts)
├── output/                     # 生成的輸出文件 (Generated Output Files)
├── .gitignore                  # Git 忽略文件 (Git Ignore File)
├── config.yaml                 # 應用程式主配置 (Main Application Configuration)
├── project_overview_config.yaml # 專案概覽工具配置 (Project Overview Tool Configuration)
├── main.py                     # CLI 主入口 (CLI Main Entry Point)
├── main_nicegui.py             # NiceGUI UI 主入口 (NiceGUI UI Main Entry Point)
├── readme.md                   # 專案說明 (Project Readme)
└── tennsy.md                   # Teensy 韌體文檔 (Teensy Firmware Documentation)
```

**目錄說明與分類原則：**

*   **`assets/`**:
    *   **原則**: 存放模擬器使用的所有靜態資源文件，例如 MuJoCo 模型所需的 3D 網格 (`.stl`)、紋理圖檔 (`.png`) 和場景定義文件 (`.xml`)。這些文件通常在運行時被載入，但不包含任何可執行邏輯。
*   **`models/`**:
    *   **原則**: 專門用於存放預訓練的 AI 模型文件，例如 ONNX 格式的策略模型 (`.onnx`, `.ort`)。這些是應用程式的數據資產，而非程式碼。
*   **`pdf/`**:
    *   **原則**: 存放專案相關的參考文檔，如 MuJoCo 的官方文檔等。
*   **`src/`**:
    *   **原則**: 本專案所有核心 Python 原始碼的根目錄。所有可執行邏輯的模組都應置於此，並在此目錄內進一步細分。這確保了源碼與其他類型的文件清晰分離。
    *   **`src/core/`**:
        *   **原則**: 存放應用程式最底層、最通用、被多個高層模組共同依賴且變動頻率較低的模組。它們是整個系統的基石。
        *   **內容**: 全局配置 (`config.py`)、中央狀態管理 (`state.py`)、異步事件匯流排 (`event_system.py`)、日誌系統 (`logger.py`)。
    *   **`src/controllers/`**:
        *   **原則**: 存放協調應用程式高層邏輯和業務流程的「指揮官」模組。它們負責監聽事件、更新中央狀態、調用低層服務，並確保各子系統的同步運行。
        *   **內容**: 模擬主控制器 (`simulation_controller.py`)、硬體主控制器 (`hardware_controller.py`)、UI 邏輯控制器 (`ui_controller.py`)。
    *   **`src/simulation/`**:
        *   **原則**: 存放與 MuJoCo 物理模擬環境直接相關的模組。這些模組通常需要直接訪問 MuJoCo 的 API 或模型數據。
        *   **內容**: MuJoCo 模擬器接口 (`simulation.py`)、3D 渲染與除錯疊層 (`rendering.py`)、地形管理 (`terrain_manager.py`)、懸浮控制器 (`floating_controller.py`)、以及將被新版取代的舊觀察器 (`observation.py`)。
    *   **`src/hardware/`**:
        *   **原則**: 存放與外部硬體設備（如 Teensy、Xbox 搖桿）進行低層通信和 AI 推理邏輯的模組。它們負責將物理世界或感測器的數據轉換為程式碼可處理的格式，或將程式碼指令轉換為硬體可執行的命令。
        *   **內容**: AI 策略管理 (`policy.py`)、序列埠通信 (`serial_communicator.py`)、Xbox 搖桿接口 (`xbox_controller.py`)。
    *   **`src/input_handlers/`**:
        *   **原則**: 專門處理用戶輸入（鍵盤、搖桿）的模組。它們的職責是將原始輸入事件翻譯為應用程式內部的標準命令或請求事件。
        *   **內容**: 鍵盤輸入處理 (`keyboard_input_handler.py`)、Xbox 搖桿輸入處理 (`xbox_input_handler.py`)。
    *   **`src/utils/`**:
        *   **原則**: 存放跨多個模塊使用的通用輔助函數或小型工具類。這些模塊通常不包含複雜的業務邏輯，而是提供可重用的功能。
        *   **內容**: 序列埠選擇工具 (`serial_utils.py`)。
    *   **`src/mock/`**:
        *   **原則**: 存放用於測試或無頭模式下使用的「模擬」或「假」的模組實現。它們提供與真實模組相同的接口，但內部邏輯是簡化或模擬的。
        *   **內容**: 模擬 MuJoCo 環境和各類控制器 (`mock_simulation.py`)。
*   **`test/`**:
    *   **原則**: 存放所有自動化測試腳本 (`pytest` 測試) 和獨立的輔助工具/演示腳本。這些文件不屬於應用程式的核心運行邏輯，但對開發和驗證至關重要。
    *   **內容**: 專案概覽生成工具 (`project_overview.py`)、序列埠控制台工具 (`test_pyserial_console.py`)、以及各種測試用例。
*   **`output/`**:
    *   **原則**: 專門存放腳本在運行時生成的各種輸出文件。這些文件通常不應被版本控制，因此被添加到 `.gitignore` 中。
    *   **內容**: 專案概覽報告 (`project_overview_*.md/.txt`)、地形快照 (`terrain_snapshot_*.png`)。
*   **根目錄文件**:
    *   **原則**: 存放專案的頂層配置、主要入口點和通用文檔。力求保持根目錄的簡潔和高層次概覽。
    *   **內容**: Git 忽略文件 (`.gitignore`)、應用程式主要配置 (`config.yaml`)、專案概覽工具配置 (`project_overview_config.yaml`)、主入口腳本 (`main.py`, `main_nicegui.py`)、專案總說明 (`readme.md`)、Teensy 韌體文檔 (`tennsy.md`)。