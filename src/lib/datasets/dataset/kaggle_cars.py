from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data


class KaggleCars(data.Dataset):
    num_classes = 1
    default_resolution = [384, 1280]
    # TODO: correct image stats
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(KaggleCars, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'pku-autonomous-driving')
        self.img_dir = os.path.join(self.data_dir, '%s_images' % split)
        self.max_objs = 50
        # self.class_name = ['__background__', 'Car']

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing pku-autonomous-driving %s data.' % split)
        self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.images = self.get_img_list(opt.trainval)
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def get_img_list(self, with_valid=False):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :return:
        """
        images = []
        if self.split == 'train':
            train_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.data_dir, 'split', self.split + '.txt'))]

            if with_valid:
                valid_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.data_dir, 'split', 'val.txt'))]
                val_list_delete = []
                train_list_all += [x for x in valid_list_all if x not in val_list_delete]   
                print("Val delete %d images." % len(val_list_delete))
                
            train_list_delete = []
            images = [x for x in train_list_all if x not in train_list_delete]
            print("Train delete %d images" % len(train_list_delete))
        elif self.split == 'test':
            images = [x[:-4] for x in os.listdir(self.img_dir)]
        return images