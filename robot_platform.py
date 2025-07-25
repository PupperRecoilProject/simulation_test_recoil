from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import time

from config import AppConfig
from state import TuningParams
from simulation import Simulation
from hardware_controller import HardwareController


class RobotPlatform(ABC):
    """Abstract interface for robot backends."""

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray, params: TuningParams) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def should_close(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def window(self):
        pass

    @property
    @abstractmethod
    def default_pose(self) -> np.ndarray:
        pass


class SimulationPlatform(RobotPlatform):
    """RobotPlatform implementation backed by MuJoCo simulation."""

    def __init__(self, config: AppConfig):
        self.sim = Simulation(config)
        self.config = config

    def setup(self) -> None:
        pass  # placeholder for future hooks

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        pos = self.sim.data.body('torso').xpos.copy()
        quat = self.sim.data.body('torso').xquat.copy()
        return {'pos': pos, 'quat': quat}

    def apply_action(self, action: np.ndarray, params: TuningParams) -> None:
        final_ctrl = self.sim.default_pose + action * params.action_scale
        self.sim.apply_position_control(final_ctrl, params)

    def step(self) -> None:
        self.sim.step()

    def should_close(self) -> bool:
        return self.sim.should_close()

    def reset(self) -> None:
        self.sim.reset()

    def close(self) -> None:
        self.sim.close()

    @property
    def window(self):
        return self.sim.window

    @property
    def default_pose(self) -> np.ndarray:
        return self.sim.default_pose


class HardwarePlatform(RobotPlatform):
    """Minimal hardware-backed platform using HardwareController."""

    def __init__(self, config: AppConfig, hw_controller: HardwareController):
        self.config = config
        self.hw_controller = hw_controller
        self._default_pose = np.zeros(config.num_motors)

    def setup(self) -> None:
        self.hw_controller.connect_and_start()

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        with self.hw_controller.lock:
            rpy = self.hw_controller.hw_state.rpy_rad.copy()
        return {"rpy_rad": rpy}

    def apply_action(self, action: np.ndarray, params: TuningParams) -> None:
        # HardwareController internally applies actions; store as command
        with self.hw_controller.lock:
            self.hw_controller.hw_state.command = action.copy()

    def step(self) -> None:
        time.sleep(self.config.control_dt)

    def should_close(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.hw_controller.stop()

    @property
    def default_pose(self) -> np.ndarray:
        return self._default_pose

    @property
    def window(self):
        return None

