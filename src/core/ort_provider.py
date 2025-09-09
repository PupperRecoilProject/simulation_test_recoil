# -*- coding: utf-8 -*-
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
