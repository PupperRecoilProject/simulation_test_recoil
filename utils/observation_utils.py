import numpy as np


def calculate_relative_joint_positions(absolute_positions: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
    """計算相對關節角度。
    Compute joint angles relative to default pose.
    """
    return absolute_positions - default_pose
