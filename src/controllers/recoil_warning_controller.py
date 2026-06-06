from src.core.event_system import (
    event_bus,
    EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED,
    EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED,
    EVENT_FRW_AUTO_INHIBIT_SET,
    EVENT_FRW_AUTO_INHIBIT_CLEAR,
)
from src.core.logger import log

class RecoilWarningController:
    def __init__(self, state_ref):
        self.state = state_ref
        # 以 config 的 auto_inhibit 作為 runtime 旗標初始值（之後由 UI switch 即時覆寫）。
        self.state.frw_auto_inhibit = self._read_config_auto_inhibit()
        event_bus.subscribe(EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED, self.on_trigger)
        event_bus.subscribe(EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED, self.on_reset)
        # 接回斷線的開關：switch 發的事件原本無人處理，導致撥動無效。
        event_bus.subscribe(EVENT_FRW_AUTO_INHIBIT_SET, self.on_auto_inhibit_set)
        event_bus.subscribe(EVENT_FRW_AUTO_INHIBIT_CLEAR, self.on_auto_inhibit_clear)
        log.info("✅ RecoilWarningController 初始化完成")

    def _read_config_auto_inhibit(self) -> bool:
        frw_cfg = getattr(getattr(self.state, 'config', None), 'firearm_recoil_warming', None)
        if isinstance(frw_cfg, dict):
            return bool(frw_cfg.get('auto_inhibit', False))
        if frw_cfg is not None:
            return bool(getattr(frw_cfg, 'auto_inhibit', False))
        return False

    def on_trigger(self, *_a, **_kw):
        self.state.recoil_warning_active = True
        log.critical("🔥 手動觸發 Firearm Recoil Warning")

    def on_reset(self, *_a, **_kw):
        if not self.state.recoil_warning_active:
            log.info("FRW 目前未觸發，略過 reset")
            return
        self.state.recoil_warning_active = False
        log.info("🟢 手動重置 Firearm Recoil Warning")

    def on_auto_inhibit_set(self, *_a, **_kw):
        self.state.frw_auto_inhibit = True
        log.info("FRW 自動預警：抑制 ON（只保留手動）")

    def on_auto_inhibit_clear(self, *_a, **_kw):
        self.state.frw_auto_inhibit = False
        log.info("FRW 自動預警：抑制 OFF（自動循環啟用）")
