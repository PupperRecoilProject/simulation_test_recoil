from src.core.event_system import (
    event_bus,
    EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED,
    EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED,
)
from src.core.logger import log

class RecoilWarningController:
    def __init__(self, state_ref):
        self.state = state_ref
        event_bus.subscribe(EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED, self.on_trigger)
        event_bus.subscribe(EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED, self.on_reset)
        log.info("✅ RecoilWarningController 初始化完成")

    def on_trigger(self, *_a, **_kw):
        self.state.recoil_warning_active = True
        log.critical("🔥 手動觸發 Firearm Recoil Warning")

    def on_reset(self, *_a, **_kw):
        if not self.state.recoil_warning_active:
            log.info("FRW 目前未觸發，略過 reset")
            return
        self.state.recoil_warning_active = False
        log.info("🟢 手動重置 Firearm Recoil Warning")
