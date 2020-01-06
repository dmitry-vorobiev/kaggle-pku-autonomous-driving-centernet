import math

import cv2
import numpy as np


def create_camera_matrix():
    return np.array([
        [2304.5479, 0, 1686.2379, 0],
        [0, 2305.8757, 1354.9849, 0],
        [0, 0, 1., 0]
    ], dtype=np.float32)


def proj_point(p, calib):
    p = np.dot(calib[:,:3], p)
    p = p[:2] / p[2]
    return p


def proj_points(pts_3d, rot_euler, translation, calib):
    # project 3D points to 2d image plane
    rot_mat = euler_angles_to_rotation_matrix(rot_euler).T
    rvect, _ = cv2.Rodrigues(rot_mat)
    imgpts, jac = cv2.projectPoints(
        np.float32(pts_3d), rvect, translation, calib[:,:3],
        distCoeffs=None)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    return imgpts


def calc_bbox(pts_3d, rot_euler, translation, calib):
    pts_2d = proj_points(pts_3d, rot_euler, translation, calib)
    x1, y1, x2, y2 = (pts_2d[:, 0].min(), pts_2d[:, 1].min(),
                      pts_2d[:, 0].max(), pts_2d[:, 1].max())
    return [x1, y1, x2, y2]


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


def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 0 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                print(q)
                q[3] = 0
    return q
