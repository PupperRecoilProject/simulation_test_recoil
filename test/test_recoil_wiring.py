"""後座力（FRW）接線回歸測試（T02-8）。

鎖死本次修復：
1. 「抑制自動預警」switch 事件會即時切換 runtime 旗標 state.frw_auto_inhibit。
2. RecoilWarningController 以 config.auto_inhibit 初始化該旗標。
3. _update_recoil_warning_timer 會遞減計時器、跨過預警門檻時觸發、歸零時重置並抽新間隔。
4. auto_inhibit=True 時自動預警被完全抑制（不會 active）。

採輕量 fake state（SimpleNamespace + 真 RLock），不啟動完整模擬執行緒。
"""
import threading
from types import SimpleNamespace

from src.core.event_system import (
    event_bus,
    EVENT_FRW_AUTO_INHIBIT_SET,
    EVENT_FRW_AUTO_INHIBIT_CLEAR,
)
from src.controllers.recoil_warning_controller import RecoilWarningController
from src.controllers.simulation_controller import SimulationController


def _make_state(cfg_inhibit=False, recoil_timer=5.0, control_dt=0.02):
    return SimpleNamespace(
        lock=threading.RLock(),
        recoil_timer=recoil_timer,
        recoil_interval=5.0,
        recoil_warning_active=False,
        frw_auto_inhibit=False,
        config=SimpleNamespace(
            control_dt=control_dt,
            firearm_recoil_warming=SimpleNamespace(auto_inhibit=cfg_inhibit),
        ),
    )


def _tick(state):
    """以 fake self 呼叫計時器邏輯（self 需有 .state 與 .config）。"""
    fake_self = SimpleNamespace(state=state, config=state.config)
    SimulationController._update_recoil_warning_timer(fake_self)


def test_switch_events_toggle_runtime_flag():
    state = _make_state(cfg_inhibit=False)
    RecoilWarningController(state)
    assert state.frw_auto_inhibit is False  # 初始 = config

    event_bus.publish(EVENT_FRW_AUTO_INHIBIT_SET)
    assert state.frw_auto_inhibit is True

    event_bus.publish(EVENT_FRW_AUTO_INHIBIT_CLEAR)
    assert state.frw_auto_inhibit is False


def test_init_reads_config_auto_inhibit():
    state = _make_state(cfg_inhibit=True)
    RecoilWarningController(state)
    assert state.frw_auto_inhibit is True


def test_timer_decrements_and_warns_near_zero():
    # 計時器接近 0（<=0.15 預警門檻）→ 推進一次後應觸發預警
    state = _make_state(cfg_inhibit=False, recoil_timer=0.10, control_dt=0.02)
    _tick(state)
    assert state.recoil_timer < 0.10           # 有遞減
    assert state.recoil_warning_active is True  # 跨過預警門檻


def test_timer_resets_and_picks_new_interval_at_zero():
    state = _make_state(cfg_inhibit=False, recoil_timer=0.01, control_dt=0.02)
    _tick(state)  # 0.01 - 0.02 = -0.01 <= 0 → 事件發生並重置
    assert state.recoil_warning_active is False
    assert 2.5 <= state.recoil_interval <= 10.0
    assert state.recoil_timer == state.recoil_interval


def test_auto_inhibit_suppresses_warning():
    # 抑制時即使計時器跨過門檻也不可 active
    state = _make_state(cfg_inhibit=False, recoil_timer=0.10, control_dt=0.02)
    state.frw_auto_inhibit = True
    _tick(state)
    assert state.recoil_warning_active is False
