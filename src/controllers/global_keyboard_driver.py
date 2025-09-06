# -*- coding: utf-8 -*-
"""
GlobalKeyboardDriver — 離散步進 & 完整熱鍵
- 每次 keydown 就對指令向量加/減一步；keyup 不變
- 忽略輸入焦點
- 補齊 UI 熱鍵：Space/N/U/J/K
指令向量: [vy, vx, wz, pitch]；鍵位：W/S→vx，A/D→vy，Q/E→wz，I/K→pitch
"""
from __future__ import annotations
from typing import Optional
import time
import numpy as np

from nicegui import ui
from nicegui.events import KeyEventArguments

from src.core.event_system import (
    event_bus,
    EVENT_COMMAND_UPDATED,
    EVENT_INPUT_MODE_CHANGE_REQUESTED,
    EVENT_SIMULATION_RESET_REQUESTED,
    EVENT_MANUAL_FLOAT_TOGGLED,
    EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED,
    EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED,
    EVENT_DEVICE_CONNECT_REQUESTED,
    EVENT_HARDWARE_AI_TOGGLE_REQUESTED,
)
from src.core.state import SimulationState


class GlobalKeyboardDriver:
    def __init__(
        self,
        state: SimulationState,
        key_speed: Optional[float] = None,
        pitch_step: Optional[float] = None,
        timeout_sec: int = 30,
    ) -> None:
        self.state = state
        base_step = float(getattr(self.state.config, 'keyboard_velocity_adjust_step', 0.30))
        self.key_speed: float = float(key_speed if key_speed is not None else base_step)
        default_pitch = max(0.02, base_step * 0.5)
        self.pitch_step: float = float(
            pitch_step if pitch_step is not None else getattr(self.state.config, 'keyboard_pitch_step', default_pitch)
        )
        self.pitch_limit = None
        if hasattr(self.state.config, 'pitch_limit'):
            try:
                self.pitch_limit = float(self.state.config.pitch_limit)
            except Exception:
                self.pitch_limit = None

        self.timeout_sec = int(timeout_sec)
        self._last_ts: float | None = None
        self._active: bool = True

        # 當前指令（持續累加）
        self._vec = np.zeros(4, dtype=float)  # [vy, vx, wz, pitch]

        # 橫幅
        self._banner = ui.label().classes(
            'fixed top-0 left-0 w-full bg-red-600 text-white text-center p-2 hidden z-50'
        )

        # 全域鍵盤（忽略輸入元件）
        self._kb = ui.keyboard(
            active=True,
            repeating=True,
            ignore=['input', 'textarea', 'select', '[contenteditable]'],
        )
        self._kb.on_key(self._on_key)

        ui.timer(5.0, self._check_timeout)
        self._try_subscribe_input_mode()
        self._show_banner()

    # ---------- 對外 ----------
    def enable(self) -> None:
        self._active = True
        self._kb.active = True
        self._show_banner()

    def disable(self) -> None:
        self._active = False
        self._kb.active = False
        self._show_banner(hide=True)

    def on_input_mode_change(self, mode: str) -> None:
        m = (mode or '').upper()
        # 非獨占：保持鍵盤一直啟用，避免與 VJOY / Gamepad / Hardware 互斥
        # 如果未來你真的需要獨占，再把這段改回去即可。
        self.enable()
        
    # ---------- 內部 ----------
    def _try_subscribe_input_mode(self) -> None:
        try:
            event_bus.subscribe(
                EVENT_INPUT_MODE_CHANGE_REQUESTED,
                lambda mode=None, **_: self.on_input_mode_change(mode or '')
            )
        except Exception:
            pass

    def _show_banner(self, hide: bool = False) -> None:
        if hide:
            self._banner.classes(add='hidden')
        else:
            self._banner.text = '🚨 鍵盤駕駛：W/S 前後，A/D 左右平移，Q/E 旋轉，I/K 俯仰，C 清零；Space 播放/暫停，N 單步，U 串列、J 搖桿、K 硬體AI，R 重置，Enter/Esc FRW，? 說明'
            self._banner.classes(remove='hidden')

    def _check_timeout(self) -> None:
        if not self._active or self._last_ts is None:
            return
        if time.time() - self._last_ts > self.timeout_sec:
            self.disable()
            ui.notify('鍵盤駕駛逾時自動關閉')

    def _publish(self) -> None:
        if self.pitch_limit is not None:
            self._vec[3] = float(np.clip(self._vec[3], -self.pitch_limit, self.pitch_limit))
        event_bus.publish(EVENT_INPUT_MODE_CHANGE_REQUESTED, mode='KEYBOARD')
        event_bus.publish(EVENT_COMMAND_UPDATED, command=self._vec.copy())

    def _clear(self) -> None:
        self._vec[:] = 0.0
        event_bus.publish(EVENT_COMMAND_UPDATED, command=np.zeros(4, dtype=float))

    def _step(self, keyname: str) -> None:
        """依鍵名做一步累加/遞減。"""
        if keyname == 'w':   self._vec[1] += self.key_speed           # vx+
        elif keyname == 's': self._vec[1] -= self.key_speed           # vx-
        elif keyname == 'a': self._vec[0] += self.key_speed           # vy+
        elif keyname == 'd': self._vec[0] -= self.key_speed           # vy-
        elif keyname == 'q': self._vec[2] += self.key_speed           # wz+
        elif keyname == 'e': self._vec[2] -= self.key_speed           # wz-
        elif keyname == 'i': self._vec[3] += self.pitch_step          # pitch+
        elif keyname == 'k': self._vec[3] -= self.pitch_step          # pitch-

    def _toggle_pause(self) -> None:
        try:
            with self.state.lock:
                self.state.single_step_mode = not self.state.single_step_mode
                if not self.state.single_step_mode:
                    self.state.execute_one_step = False
        except Exception:
            # 如果沒有 lock 或屬性，安靜略過
            pass
    def _is_help_hotkey(self, e: KeyEventArguments) -> bool:
        """偵測 '?'：支援 '?', '？', 'question', 'questionmark'，
        或 Shift + '/' / 'slash'（兼容不同瀏覽器/鍵盤回報）。"""
        name = (e.key.name or '').lower()

        # 直接回報問號（含全形）
        if name in {'?', '？', 'question', 'questionmark'}:
            return True

        # 可能回報成 '/' 或 'slash'，這時需要 Shift
        shift = bool(getattr(e.key, 'shift', False))
        try:
            # 某些版本會把 Shift 放在 e.modifiers.shift
            shift = shift or bool(getattr(e, 'modifiers', None) and getattr(e.modifiers, 'shift', False))
        except Exception:
            pass

        return shift and name in {'/', 'slash'}



    def _on_key(self, e: KeyEventArguments) -> None:
        if not self._active:
            return
        # 只在 keydown 處理；忽略 repeat
        if not e.action.keydown or e.action.repeat:
            return

        self._last_ts = time.time()
        name = (e.key.name or '').lower()

        # ---- 移動 / 姿態（離散步進）----
        if name in {'w','a','s','d','q','e','i','k'}:
            self._step(name); self._publish(); return

        # ---- 清零 ----
        if name == 'c':
            self._clear(); return

        # ---- UI 熱鍵 ----
        if e.key.space:
            self._toggle_pause(); return
        if name == 'n':
            try:
                self.state.execute_one_step = True
            except Exception:
                pass
            return
        if name == 'u':
            event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device='serial'); return
        if name == 'j':
            event_bus.publish(EVENT_DEVICE_CONNECT_REQUESTED, device='gamepad'); return
        if name == 'k':
            event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED); return

        # ---- 其他功能鍵 ----
        if name == 'r':
            event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type='hard'); return
        if e.key.enter:
            event_bus.publish(EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED); return
        if e.key.escape:
            event_bus.publish(EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED); return
        if name == 'f':
            # 正確帶參：is_floating
            try:
                new_val = not bool(getattr(self.state, 'manual_mode_is_floating', False))
            except Exception:
                new_val = True
            event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=new_val); return
        if self._is_help_hotkey(e):
            ui.notify('W/S 前後，A/D 左右平移，Q/E 旋轉，I/K 俯仰；C 清零；Space 播放/暫停，N 單步，U 串列、J 搖桿、K 硬體AI；R 重置；Enter/Esc FRW'); return