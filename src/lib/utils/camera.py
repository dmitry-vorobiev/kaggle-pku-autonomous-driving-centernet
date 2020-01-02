import math
import numpy as np


def create_camera_matrix():
    return np.array([
        [2304.5479, 0, 1686.2379, 0],
        [0, 2305.8757, 1354.9849, 0],
        [0, 0, 1., 0]
    ], dtype=np.float32)


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