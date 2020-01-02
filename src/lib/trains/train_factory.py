from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from.car_pose_6dof import CarPose6DoFTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer,
  'car_pose_6dof': CarPose6DoFTrainer,
}
