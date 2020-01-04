from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import OrderedDict

from utils import car_models
from utils.camera import create_camera_matrix

class KaggleCars(data.Dataset):
    num_classes = 1
    default_resolution = [384, 1280]
    # TODO: correct image stats
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(KaggleCars, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'pku-autonomous-driving')
        self.img_dir = os.path.join(self.data_dir, KaggleCars._img_dir_name(split))
        self.calib = create_camera_matrix()
        self.max_objs = 50
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
        self.car_models = self.load_car_models()
        self.images = self.get_img_list(opt.trainval)
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _img_dir_name(split):
        dir_name = 'test_images' if split == 'test' else 'train_images'
        return dir_name

    def _read_split_file(self, split):
        path = os.path.join(self.data_dir, 'split', split+'.txt')
        with open(path) as f:
            return [line.rstrip('\n')[:-4] for line in f]

    def get_img_list(self, with_valid=False):
        """
        Get the image list,
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :return:
        """
        images = []
        ignore = set()
        if self.split == 'test':
            images = [x[:-4] for x in os.listdir(self.img_dir)]
        else:
            images = self._read_split_file(self.split)
            if self.split == 'train' and with_valid:
                val = self._read_split_file('val')
                images += val
            ignore = set(self._read_split_file('ignore'))
            images = [x for x in images if x not in ignore]
        print("Loaded %d %s images, skipped: %d" % (len(images), self.split, len(ignore)))
        return images

    def load_car_models(self):
        """Load all the car models
        """
        car_models_all = OrderedDict([])
        print('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = os.path.join(self.data_dir, 'car_models_json', model.name+'.json')
            with open(car_model) as json_file:
                car = json.load(json_file)
            for key in ['vertices', 'faces']:
                car[key] = np.array(car[key])
            # fix the inconsistency between obj and pkl
            car['vertices'][:, [0, 1]] *= -1
            car_models_all[model.name] = car 
        return car_models_all

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, img_id+'.txt')
            with open(out_path, 'w') as f:
                for cls_ind in results[img_id]:
                    for j in range(len(results[img_id][cls_ind])):
                        class_name = car_models.car_id2name[cls_ind].name
                        f.write('%s |' % class_name)
                        for i in range(len(results[img_id][cls_ind][j])):
                            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
                        f.write('\n')

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)