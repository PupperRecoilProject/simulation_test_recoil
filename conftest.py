"""pytest 根設定（T02-4）。

放在 repo 根目錄：(1) 把 rootdir 釘在這裡，讓測試能 `from src...` 匯入；
(2) 確保 src 在 sys.path 上（即使從別處呼叫 pytest）。
詳見 docs/TEST_SOP.md。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
