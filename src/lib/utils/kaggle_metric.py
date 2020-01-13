import math
import numpy as np
from scipy.spatial.transform import Rotation as R

MAX_VAL = 10**10


def trans_dist(pred, gt, abs_dist=False):    
    diff = np.sqrt(np.square(pred - gt).sum())
    if not abs_dist:
        diff0 = np.sqrt(np.square(gt).sum())
        diff /= diff0
    return diff


def rot_dist(pred, gt):
    q1 = R.from_euler('xyz', gt)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    
    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (math.acos(W) * 360) / math.pi
    if W > 180:
        W = 360 - W
    return W
