# -*- coding: utf-8 -*-
"""
===============================================================================
ORT Provider 工具使用說明（3 個環境版本）
===============================================================================
本模組負責在不同平台自動選擇 ONNX Runtime Execution Provider（EP）。
已提供：
- create_session(model_path, sess_options=None, provider_options=None)
- get_selected_providers()
- diag_print(prefix="[ORT]")

本說明分為三個「環境版本」對應你的三台機器。

------------------------------------------------------------------------------
版本 A：Windows 筆電（i7-11th，無獨顯）
------------------------------------------------------------------------------
目標：能跑 ONNX 推論，若可用 iGPU 走 DML，否則自動回落 CPU

安裝建議
- 優先安裝 onnxruntime-directml 以啟用 DmlExecutionProvider
- 若環境無法使用 DML，再改用一般 onnxruntime（只會有 CPUExecutionProvider）

驗證
  python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
期望輸出之一
  ['DmlExecutionProvider', 'CPUExecutionProvider']   或
  ['CPUExecutionProvider']

程式端不需改，create_session() 會自動選到 DML 或 CPU

------------------------------------------------------------------------------
版本 B：Windows 筆電（i7-11th + RTX 3070，固定用 DML）
------------------------------------------------------------------------------
目標：不安裝 CUDA 與 cuDNN，也能使用 GPU 加速

安裝建議
- 僅安裝 onnxruntime-directml
- 不要同時安裝 onnxruntime-gpu，以免 ORT 嘗試載入 CUDA DLL 後報錯

驗證
  python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
期望輸出
  ['DmlExecutionProvider', 'CPUExecutionProvider']

程式端不需改，create_session() 會選到 DML

備註
- 如果未來改想用 CUDA，需要完整安裝 CUDA 12.x 與 cuDNN 9.x 並將其 bin 加入 PATH
- 二擇一環境較單純。DML 省事，CUDA 速度通常更好

------------------------------------------------------------------------------
版本 C：Jetson Orin Nano（Linux aarch64）
------------------------------------------------------------------------------
目標：優先使用 TensorRT 或 CUDA 推論

安裝建議
- 安裝 onnxruntime-gpu（與 JetPack 的 CUDA/cuDNN 對應）
- 若 ORT build 支援 TensorRT EP，會自動出現在 available providers

驗證
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
期望輸出之一
  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']  或
  ['CUDAExecutionProvider', 'CPUExecutionProvider']

程式端不需改，create_session() 會依序偏好 TensorRT → CUDA → CPU

------------------------------------------------------------------------------
通用呼叫方式（三版本皆適用）
------------------------------------------------------------------------------
from src.core.ort_provider import create_session, diag_print
diag_print("[ORT][boot]")                           # 可選，啟動時列出環境與 EP
sess = create_session("models/your_model.onnx")     # 自動選擇最佳 EP

如需 SessionOptions
import onnxruntime as ort
from src.core.ort_provider import create_session
opt = ort.SessionOptions()
# opt.intra_op_num_threads = 1
# opt.enable_mem_pattern = False
sess = create_session("models/your_model.onnx", sess_options=opt)

------------------------------------------------------------------------------
偏好順序與可調整說明
------------------------------------------------------------------------------
預設偏好
- Jetson/Linux aarch64：TensorRT → CUDA → CPU
- Windows：CUDA → DML → CPU
- 其他：CUDA → CPU

如果你想讓 Windows 永遠優先 DML，可在 _preferred_by_platform() 中將
  return [EP_CUDA, EP_DML, EP_CPU]
改成
  return [EP_DML, EP_CUDA, EP_CPU]
一般情況不必修改，只要在該機器安裝對應的 ORT 發行版即可達到預期選擇

------------------------------------------------------------------------------
診斷與自我檢查
------------------------------------------------------------------------------
在程式任何啟動點加入：
  from src.core.ort_provider import diag_print
  diag_print("[ORT]")

會輸出：
  [ORT] Python: <python 路徑>
  [ORT] onnxruntime: <版本>
  [ORT] available: <系統偵測到的 EP 列表>
  [ORT] selected: <本模組實際選用的 EP 順序>
  [ORT] module: <onnxruntime 套件路徑>

可快速辨認是否誤用到另一個 Python 環境或另一個 ORT 變體

------------------------------------------------------------------------------
常見問題與排除
------------------------------------------------------------------------------
1) Windows 上看到 CUDA 錯誤，訊息含 cublasLt64_12.dll 或 cuDNN 相關
   - 原因：環境裝了 onnxruntime-gpu，但未安裝或未設好 CUDA 12.x 與 cuDNN 9.x
   - 作法：要嘛補齊 CUDA 與 cuDNN，要嘛改用 onnxruntime-directml 並移除 onnxruntime-gpu

2) available 出現 AzureExecutionProvider，但沒有 DmlExecutionProvider
   - 原因：載入到另一個 ORT 變體或另一個 Python 環境
   - 作法：統一環境，移除多餘 ORT 變體，只保留 onnxruntime-directml 或 onnxruntime-gpu 或 onnxruntime
           用 diag_print() 檢查 ort.__file__ 與 sys.executable 是否為預期路徑

3) 看不到 DmlExecutionProvider
   - 確認安裝的是 onnxruntime-directml
   - 確認使用的 Python 解譯器與安裝時一致
   - Windows 顯示卡與驅動需支援 DirectX 12。若不支援，會退回 CPU

4) Jetson 上看不到 TensorrtExecutionProvider
   - 可能是 ORT build 未啟用 TensorRT EP。只要有 CUDA EP 仍可使用
   - 如需 TRT，請使用支援 TRT 的 ORT 版本或依官方文件重新安裝

5) 不同終端出現不同 providers
   - 代表同機器使用了不同 Python 環境或 venv。請統一啟動方式並重啟所有 Python 進程

6) 在啟用了虛擬環境後，永遠使用 python -m pip 來安裝或管理套件，而不是直接使用 pip。

------------------------------------------------------------------------------
維護備註
------------------------------------------------------------------------------
- 本模組會快取已挑選的 providers，避免重複探測
- provider_options 預設為空。若需指定 CUDA device_id 或啟用 TRT FP16，可在 _maybe_provider_options() 中調整
- 不建議同一環境同時安裝 onnxruntime-gpu 與 onnxruntime-directml
"""

"""
【v4.10.3 新增】ONNX Runtime Provider 共用工具（跨平台自動選擇）
- Windows + NVIDIA：優先 CUDA，其次 DML，最後 CPU
- Windows + 其他 GPU / iGPU：優先 DML，最後 CPU
- Jetson / Linux + NVIDIA：優先 TensorRT，再來 CUDA，最後 CPU
- 其餘環境：CPU

用法：
from src.core.ort_provider import create_session, get_selected_providers, diag_print

sess = create_session("path/to/model.onnx")
print("Using:", get_selected_providers())
"""

from typing import List, Optional, Tuple
import os
import platform
import onnxruntime as ort

# 可能出現的 EP 名稱（是否可用以 get_available_providers() 為準）
EP_TRT   = "TensorrtExecutionProvider"
EP_CUDA  = "CUDAExecutionProvider"
EP_DML   = "DmlExecutionProvider"
EP_CPU   = "CPUExecutionProvider"

# 依平台預設偏好順序
def _preferred_by_platform() -> List[str]:
    sys = platform.system().lower()
    # 簡單推論：Jetson 多為 Linux + aarch64
    is_aarch64 = platform.machine().lower() in ("aarch64", "arm64")
    if sys == "linux" and is_aarch64:
        # Jetson/Orin/Nano：TensorRT → CUDA → CPU
        return [EP_TRT, EP_CUDA, EP_CPU]
    if sys == "windows":
        # Windows：如果裝了 CUDA 版 ORT，CUDA 通常最快；否則走 DML
        return [EP_CUDA, EP_DML, EP_CPU]
    # 其他（如一般 Linux x86_64 無 GPU）
    return [EP_CUDA, EP_CPU]

_selected_cache: List[str] = []

def get_selected_providers() -> List[str]:
    """
    回傳經由「可用 EP ∩ 偏好順序」挑選後的結果。
    會快取本次程序的結果，以避免重覆探測。
    """
    global _selected_cache
    if _selected_cache:
        return _selected_cache

    available = set(ort.get_available_providers())
    preferred = _preferred_by_platform()

    selected = [p for p in preferred if p in available]
    if not selected:
        # 安全保底
        selected = [EP_CPU]

    _selected_cache = selected
    return selected

def _maybe_provider_options(providers: List[str]) -> Optional[List[dict]]:
    """
    可在此針對特定 EP 放入 provider options（多半可留空）。
    例：CUDA 可指定 device_id；DML 通常不需要。
    """
    opts: List[dict] = []
    for p in providers:
        if p == EP_CUDA:
            # 若需指定 GPU，可改為 {"device_id": 0}
            opts.append({})
        elif p == EP_TRT:
            # TensorRT 常見可調參數（依實際支援與需求調整）
            # opts.append({"trt_fp16_enable": True})
            opts.append({})
        else:
            opts.append({})
    return opts if any(opts) else None

def create_session(
    model_path: str,
    sess_options: Optional[ort.SessionOptions] = None,
    provider_options: Optional[List[dict]] = None,
    strict_provider_len: bool = False,
) -> ort.InferenceSession:
    """
    建立 InferenceSession，依平台自動選擇 EP。
    - provider_options 若未提供，會自動給一組與 providers 對齊的空 options。
    - strict_provider_len=True 時，若 provider_options 長度與 providers 不符會拋例外；
      預設 False：將忽略傳入的 provider_options 以避免誤用而失敗。
    """
    providers = get_selected_providers()

    # 檔案存在性基本檢查
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX 模型不存在：{model_path}")

    # 預設 options
    if provider_options is None:
        provider_options = _maybe_provider_options(providers)

    # 長度保護
    if provider_options and len(provider_options) != len(providers):
        if strict_provider_len:
            raise ValueError("provider_options 長度需與 providers 對齊")
        provider_options = None

    # 建立 Session
    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )

def diag_info() -> Tuple[str, str, List[str], str]:
    """
    回傳偵錯資訊：(python_exe, ort_version, available_providers, ort_file)
    """
    import sys
    return (
        sys.executable,
        ort.__version__,
        list(ort.get_available_providers()),
        ort.__file__,
    )

def diag_print(prefix: str = "[ORT]") -> None:
    py, ver, av, path = diag_info()
    print(f"{prefix} Python: {py}")
    print(f"{prefix} onnxruntime: {ver}")
    print(f"{prefix} available: {av}")
    print(f"{prefix} selected: {get_selected_providers()}")
    print(f"{prefix} module: {path}")
