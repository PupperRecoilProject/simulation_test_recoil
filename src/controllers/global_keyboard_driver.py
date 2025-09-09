# -*- coding: utf-8 -*-
"""
GlobalKeyboardDriver — 離散步進 & 完整熱鍵
- 每次 keydown 就對指令向量加/減一步；keyup 不變
- 忽略輸入焦點
- UI 熱鍵：Space/N/U/J/V/O/X/R/~...
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
    EVENT_TERRAIN_CHANGE_REQUESTED,
    EVENT_POLICY_CHANGE_REQUESTED,   # 數字鍵選策略
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
        self.pitch_limit: float | None = None
        if hasattr(self.state.config, 'pitch_limit'):
            try:
                self.pitch_limit = float(self.state.config.pitch_limit)
            except Exception:
                self.pitch_limit = None

        # 允許從 config 覆寫預設逾時
        cfg_timeout = getattr(self.state.config, 'keyboard_driver_timeout_sec', None)
        if cfg_timeout is not None and timeout_sec == 30:
            try:
                timeout_sec = int(cfg_timeout)
            except Exception:
                pass

        self.timeout_sec = int(timeout_sec)
        self._last_ts: float | None = None
        self._active: bool = True
        self._auto_off: bool = False  # 逾時自動關閉後的鎖定旗標

        # 當前指令（持續累加）: [vy, vx, wz, pitch]
        self._vec = np.zeros(4, dtype=float)

        # 橫幅（可點擊重新啟用）
        self._banner = ui.label().classes(
            'fixed top-0 left-0 w-full bg-red-600 text-white text-center p-2 hidden z-50 cursor-pointer'
        )
        self._banner.on('click', self._on_banner_click)

        # 全域鍵盤（忽略輸入元件）
        self._kb = ui.keyboard(
            active=True,
            repeating=True,
            ignore=['input', 'textarea', 'select', '[contenteditable]'],
        )
        self._kb.on_key(self._on_key)

        # 操作手冊對話框
        with ui.dialog() as self._help_dlg, ui.card():
            ui.label('操作手冊 / 快捷鍵').classes('text-lg font-bold')
            self._help_md = ui.markdown(self._build_help_text()).classes('text-sm').style('max-height:60vh;overflow:auto')
            ui.button('關閉', on_click=self._help_dlg.close).classes('mt-2')

        ui.timer(5.0, self._check_timeout)
        self._try_subscribe_input_mode()
        self._show_banner()

    # ---------- 對外 ----------
    def enable(self) -> None:
        self._auto_off = False            # 解除逾時鎖定
        self._active = True
        self._kb.active = True
        self._show_banner()

    def disable(self) -> None:
        self._active = False
        self._kb.active = False
        self._show_banner(hide=True)

    def on_input_mode_change(self, mode: str) -> None:
        # 逾時自動關閉後，不因其他輸入模式事件而自動再啟用，避免誤觸
        if self._auto_off:
            return
        self.enable()

    def show_help(self) -> None:
        """提供給 UI header 按鈕呼叫。"""
        # 若目前在 auto-off，按「操作手冊(~)」也視為人工喚醒
        if self._auto_off or not self._active:
            self.enable()
            self.poke()
        try:
            self._help_md.set_content(self._build_help_text())
        except Exception:
            pass
        self._help_dlg.open()

    # 讓逾時可動態調整 / 關閉（0 表示關閉）
    def set_timeout_sec(self, seconds: int) -> None:
        self.timeout_sec = int(seconds)
        self.poke()
        ui.notify(
            f"鍵盤駕駛自動關閉已{'啟用' if self.timeout_sec > 0 else '關閉'}"
            + (f"（{self.timeout_sec}s）" if self.timeout_sec > 0 else "")
        )
        self._show_banner()

    def poke(self) -> None:
        """手動續命：重置逾時計時點（任何有效 keydown 都會呼叫）。"""
        self._last_ts = time.time()

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
            auto = f"Auto-off {self.timeout_sec}s" if self.timeout_sec > 0 else "Auto-off 關"
            self._banner.text = (
                '🚨 鍵盤駕駛：W/S 前後，A/D 左右平移，Q/E 旋轉，I/K 俯仰，C 清零；'
                'Space 播放/暫停，N 單步，U 串列、J 搖桿、V 硬體AI，O 切地形，'
                f'X 軟重置，R 硬重置，Enter/Esc FRW，~ 說明；數字鍵 0–9 選策略；{auto}'
            )
            self._banner.classes(remove='hidden')

    def _show_reactivate_banner(self) -> None:
        """逾時自動關閉後，顯示可點擊重新啟用的橫幅。"""
        self._banner.text = '⌛ 鍵盤駕駛已自動關閉（逾時）。點我重新啟用並顯示操作手冊 (~)'
        self._banner.classes(remove='hidden')

    def _on_banner_click(self, *_):
        # 使用者明確操作 → 解除 auto_off、啟用、續命並開手冊
        self.enable()
        self.poke()
        self.show_help()

    def _check_timeout(self) -> None:
        if not self._active or self._last_ts is None or self.timeout_sec <= 0:
            return
        if time.time() - self._last_ts > self.timeout_sec:
            # 進入 auto-off：關閉鍵盤並顯示可點擊的橫幅
            self._auto_off = True
            self._active = False
            self._kb.active = False
            self._show_reactivate_banner()
            ui.notify(f'鍵盤駕駛逾時（{self.timeout_sec}s）自動關閉')

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
            pass

    # —— 熱鍵判斷：~（Shift + `）——
    def _is_help_hotkey(self, e: KeyEventArguments) -> bool:
        """
        偵測 '~'：常見回報：
        - name='~' 或 'tilde'
        - 或 Shift + '`' / 'backquote' / 'grave' / 'grave_accent'
        """
        name = (e.key.name or '').lower()
        if name in {'~', 'tilde'}:
            return True
        shift = bool(getattr(e.key, 'shift', False))
        try:
            shift = shift or bool(getattr(e, 'modifiers', None) and getattr(e.modifiers, 'shift', False))
        except Exception:
            pass
        return shift and name in {'`', 'backquote', 'grave', 'grave_accent'}

    # —— 數字鍵 → 0-based 索引 ——
    def _digit_to_index(self, name: str) -> int | None:
        mapping = {
            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '0': 9,
            'digit1': 0, 'digit2': 1, 'digit3': 2, 'digit4': 3, 'digit5': 4,
            'digit6': 5, 'digit7': 6, 'digit8': 7, 'digit9': 8, 'digit0': 9,
            'numpad1': 0, 'numpad2': 1, 'numpad3': 2, 'numpad4': 3, 'numpad5': 4,
            'numpad6': 5, 'numpad7': 6, 'numpad8': 7, 'numpad9': 8, 'numpad0': 9,
        }
        return mapping.get(name)

    # —— 操作手冊內容（Markdown）——
    def _build_help_text(self) -> str:
        policies = list(getattr(self.state, 'available_policies', []))
        polylines = []
        for i, p in enumerate(policies[:10]):
            key = (i + 1) if i < 9 else 0
            polylines.append(f"- **{key}** → {p}")

        return (
            "### 鍵盤駕駛（NiceGUI 頁面）\n"
            "- **W/S**：前進/後退（vx ±）\n"
            "- **A/D**：左/右平移（vy ±）\n"
            "- **Q/E**：左/右旋轉（wz ±）\n"
            "- **I/K**：俯仰 ±（pitch）\n"
            "- **C**：清零（vy,vx,wz,pitch 全歸零）\n"
            "- **Space**：播放/暫停、**N**：單步\n"
            "- **U/J**：掃描序列埠/搖桿、**V**：切換硬體AI\n"
            "- **O**：切換地形模式、**X/R**：軟/硬重置\n"
            "- **Enter/Esc**：FRW 觸發/重置\n"
            "- **~**（Shift+`）：顯示本操作手冊\n"
            "\n### 模型/策略選擇（動態）\n" +
            ("\n".join(polylines) if polylines else "_（未提供清單）_") +
            "\n\n### MuJoCo 視窗（GLFW）\n"
            "- 同時保留 MuJoCo 內建快捷鍵；`G` 進/出 Joint Test；\n"
            "- ` 按鍵（GRAVE）在序列模式中用於返回前一模式；\n"
            "- 其他：R/X/Y/P/M/L/O… 如狀態列提示。"
        )

    # —— 主鍵盤處理 ——
    def _on_key(self, e: KeyEventArguments) -> None:
        if not self._active:
            return
        if not e.action.keydown or e.action.repeat:
            return

        # 任何有效 keydown → 續命
        self.poke()
        name = (e.key.name or '').lower()

        # ---- 數字鍵 0-9：動態選策略 ----
        idx = self._digit_to_index(name)
        if idx is not None:
            try:
                policies = list(getattr(self.state, 'available_policies', []))
                if 0 <= idx < len(policies):
                    event_bus.publish(EVENT_POLICY_CHANGE_REQUESTED, policy_name=policies[idx])
            except Exception:
                pass
            return

        # ---- 移動 / 姿態（離散步進）----
        if name in {'w', 'a', 's', 'd', 'q', 'e', 'i', 'k'}:
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
        if name == 'v':
            event_bus.publish(EVENT_HARDWARE_AI_TOGGLE_REQUESTED); return
        if name == 'o':
            event_bus.publish(EVENT_TERRAIN_CHANGE_REQUESTED, name='TOGGLE'); return

        # ---- 其他功能鍵 ----
        if name == 'x':
            event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type='soft'); return
        if name == 'r':
            event_bus.publish(EVENT_SIMULATION_RESET_REQUESTED, type='hard'); return
        if e.key.enter:
            event_bus.publish(EVENT_FIREARM_RECOIL_WARNING_TRIGGER_REQUESTED); return
        if e.key.escape:
            event_bus.publish(EVENT_FIREARM_RECOIL_WARNING_RESET_REQUESTED); return
        if name == 'f':
            try:
                new_val = not bool(getattr(self.state, 'manual_mode_is_floating', False))
            except Exception:
                new_val = True
            event_bus.publish(EVENT_MANUAL_FLOAT_TOGGLED, is_floating=new_val); return
        if self._is_help_hotkey(e):  # ~ 開啟操作手冊
            self.show_help(); return
