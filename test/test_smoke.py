"""最小冒煙測試（T02-4）：確認 pytest 基礎設施可運作、核心模組可匯入。

不測行為，只證明「測試框架本身與匯入路徑是通的」。真正的單元測試另立檔案。
"""


def test_pytest_infra_alive():
    assert True


def test_core_imports():
    # 純匯入檢查：抓到「裝飾器/語法/相依」層級的 breakage。
    from src.core import event_system, state, config  # noqa: F401
    from src.core.event_system import event_bus  # noqa: F401
