前置安裝：你需要安裝 PyYAML。在你的終端機中執行：
pip install onnxruntime numpy
pip install PyYAML
pip install mujoco glfw //這個看看環境有沒有載，我不是用這個
pip install inputs //搖桿用的
pip install pygame
pip install pyserial
pip install Pillow
pip install scipy  //運動學 hardware 需用

pip install numpy onnxruntime PyYAML nicegui pygame pyserial Pillow scipy

### 專案目錄結構與模組化原則
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