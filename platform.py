from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from config import AppConfig
from state import SimulationState, TuningParams
from simulation import Simulation


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
    def step(self, state: SimulationState) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
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

    def step(self, state: SimulationState) -> None:
        self.sim.step(state)

    def reset(self) -> None:
        self.sim.reset()

    def close(self) -> None:
        self.sim.close()

    @property
    def default_pose(self) -> np.ndarray:
        return self.sim.default_pose
