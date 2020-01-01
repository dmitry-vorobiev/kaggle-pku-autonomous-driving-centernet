import math
import numpy as np


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    yaw, pitch, roll = -angle[0], -angle[1], -angle[2]
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(yaw), -math.sin(yaw)],
        [0, math.sin(yaw), math.cos(yaw)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Ry @ Rx @ Rz
    if is_dir:
        R = R[:, 2]
    return R