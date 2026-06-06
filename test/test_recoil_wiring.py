"""後座力（FRW）接線回歸測試。

鎖死本次修復：
1. 「自動預警循環」switch 事件會即時切換 runtime 旗標 state.frw_auto_warning_enabled。
2. RecoilWarningController 以 config.auto_warning_enabled 初始化該旗標。
3. _update_recoil_warning_timer 會遞減計時器、跨過預警門檻時觸發、歸零時重置並抽新間隔。
4. auto_warning_enabled=False 時自動預警被完全停用（不會 active）。

採輕量 fake state（SimpleNamespace + 真 RLock），不啟動完整模擬執行緒。
"""
import threading
from types import SimpleNamespace

from src.core.event_system import (
    event_bus,
    EVENT_FRW_AUTO_WARNING_ENABLE,
    EVENT_FRW_AUTO_WARNING_DISABLE,
)
from src.controllers.recoil_warning_controller import RecoilWarningController
from src.controllers.simulation_controller import SimulationController


def _make_state(cfg_enabled=False, runtime_enabled=False, recoil_timer=5.0, control_dt=0.02):
    return SimpleNamespace(
        lock=threading.RLock(),
        recoil_timer=recoil_timer,
        recoil_interval=5.0,
        recoil_warning_active=False,
        frw_auto_warning_enabled=runtime_enabled,
        config=SimpleNamespace(
            control_dt=control_dt,
            firearm_recoil_warming=SimpleNamespace(auto_warning_enabled=cfg_enabled),
        ),
    )


def _tick(state):
    """以 fake self 呼叫計時器邏輯（self 需有 .state 與 .config）。"""
    fake_self = SimpleNamespace(state=state, config=state.config)
    SimulationController._update_recoil_warning_timer(fake_self)


def test_switch_events_toggle_runtime_flag():
    state = _make_state(cfg_enabled=False)
    RecoilWarningController(state)
    assert state.frw_auto_warning_enabled is False  # 初始 = config

    event_bus.publish(EVENT_FRW_AUTO_WARNING_ENABLE)
    assert state.frw_auto_warning_enabled is True

    event_bus.publish(EVENT_FRW_AUTO_WARNING_DISABLE)
    assert state.frw_auto_warning_enabled is False


def test_init_reads_config_auto_warning_enabled():
    state = _make_state(cfg_enabled=True)
    RecoilWarningController(state)
    assert state.frw_auto_warning_enabled is True


def test_timer_decrements_and_warns_near_zero():
    # 計時器接近 0（<=0.15 預警門檻）→ 推進一次後應觸發預警（需啟用自動預警）
    state = _make_state(runtime_enabled=True, recoil_timer=0.10, control_dt=0.02)
    _tick(state)
    assert state.recoil_timer < 0.10           # 有遞減
    assert state.recoil_warning_active is True  # 跨過預警門檻


def test_timer_resets_and_picks_new_interval_at_zero():
    state = _make_state(runtime_enabled=True, recoil_timer=0.01, control_dt=0.02)
    _tick(state)  # 0.01 - 0.02 = -0.01 <= 0 → 事件發生並重置
    assert state.recoil_warning_active is False
    assert 2.5 <= state.recoil_interval <= 10.0
    assert state.recoil_timer == state.recoil_interval


def test_auto_warning_disabled_suppresses_warning():
    # 停用時即使計時器跨過門檻也不可 active
    state = _make_state(runtime_enabled=False, recoil_timer=0.10, control_dt=0.02)
    _tick(state)
    assert state.recoil_warning_active is False
