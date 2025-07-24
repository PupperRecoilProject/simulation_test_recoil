"""Simple joystick demo used for manual testing."""

import importlib.util
import time

import pytest

if importlib.util.find_spec("pygame") is None:
    pytest.skip("pygame not installed", allow_module_level=True)

import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("錯誤：未偵測到任何搖桿。")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"已連接到: {joystick.get_name()}")
print(f"搖桿有 {joystick.get_numaxes()} 個軸。")
print("\n請移動您的搖桿，觀察每個軸的編號和數值變化...")
print("按 Ctrl+C 結束測試。")

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                print(f"軸 (Axis) {event.axis}: {event.value:.3f}")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n測試結束。")

pygame.quit()