import math
import numpy as np


def create_camera_matrix():
    return np.array([
        [2304.5479, 0, 1686.2379, 0],
        [0, 2305.8757, 1354.9849, 0],
        [0, 0, 1., 0]
    ], dtype=np.float32)

def project_point(p, calib):
    p = np.dot(calib[:,:3], p)
    p = p[:2] / p[2]
    return p

def euler_angles_to_rotation_matrix(angle):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
    Output:
        R: 3 x 3 matrix
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
    return R


def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions representation.
    Input:
        angle: n x 3 matrix, each row is [roll, pitch, yaw]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = -angle[:, 1], -angle[:, 0], -angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    return q


def quaternion_to_euler_angle(q):
    """Convert quaternion to euler angel.
    Input:
        q: 1 x 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [roll, pitch, yaw]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return np.array([-Y, -X, -Z], dtype=np.float32)