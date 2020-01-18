from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.metrics import average_precision_score

from utils import car_models
from utils.geometry import calc_bbox, create_camera_matrix, proj_points
from utils.image import car_6dof_pixel_tfms
from utils.kaggle_cars_utils import load_car_models, parse_annot_str, parse_pred_str
from utils.kaggle_metric import rot_dist, trans_dist

class KaggleCars(data.Dataset):
    num_classes = 1
    default_resolution = [512, 1280]
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
        self.loc_norm = [519.834, 689.119, 3502.94] # max for x,y,z center
        self.tfms = car_6dof_pixel_tfms(opt)
        self.split = split
        self.opt = opt

        print('==> Initializing pku-autonomous-driving %s data.' % split)
        self.images = self.get_img_list(opt.trainval)
        self.models = self.load_car_models()
        self.anns = self.load_annotations(self.images, self.models)
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _img_dir_name(split):
        dir_name = 'test_images' if split == 'test' else 'train_images'
        return dir_name

    def _read_split_file(self, split):
        split_dir = self.opt.split_dir
        if not os.path.isdir(split_dir):
            split_dir = os.path.join(self.data_dir, 'split')
        path = os.path.join(split_dir, split+'.txt')
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
            df = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
            images = df['ImageId'].values.tolist()
        else:
            images = self._read_split_file(self.split)
            if self.split == 'train' and with_valid:
                val = self._read_split_file('val')
                images += val
            ignore = set(self._read_split_file('ignore'))
            images = [x for x in images if x not in ignore]
        print("===> Loaded %d %s images IDs, skipped: %d" % (len(images), self.split, len(ignore)))
        return images

    def load_car_models(self):
        model_dir = os.path.join(self.data_dir, 'car_models_json')
        models_3D = load_car_models(model_dir)
        print("====> Loaded %d 3D models" % len(models_3D))
        return models_3D

    def load_annotations(self, img_ids, models_3D):
        anns = []
        if self.split != 'test':
            df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            anns = [self.gen_img_annotation(img_id, df, models_3D) for img_id in img_ids]
            print('=====> Loaded %d %s annotations.' % (len(anns), self.split))
        return anns

    def gen_img_annotation(self, img_id, df, models_3D):
        cond = df['ImageId'] == img_id
        cars = df.loc[cond, 'PredictionString'].values[0]
        cars = parse_annot_str(str(cars))
        for car in cars:
            car_name = car_models.car_id2name[car['car_id']].name
            car_model = models_3D[car_name]
            pts_2d = proj_points(car_model['vertices'], car['rotation'],
                                 car['location'], self.calib)
            bbox = calc_bbox(pts_2d)
            car['bbox'] = np.array(bbox)
        return cars

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        shape = (len(results), 2)
        df = pd.DataFrame(np.empty(shape), columns=['ImageId', 'PredictionString'])
        for i, img_id in enumerate(results.keys()):
            df.loc[i, 'ImageId'] = img_id
            preds = []
            for cls_ind in results[img_id]:
                for j in range(len(results[img_id][cls_ind])):
                    car = results[img_id][cls_ind][j]
                    # yaw, pitch, roll, x, y, z
                    pose = [str(car[i]) for i in range(6)]
                    conf = str(car[-1])
                    pose.append(conf)
                    pose_str = ' '.join(pose)
                    preds.append(pose_str)
            df.loc[i, 'PredictionString'] = ' '.join(preds)
        df['PredictionString'].fillna('', inplace=True)
        out_path = os.path.join(results_dir, 'predictions.csv')
        df.to_csv(out_path, index=False)

    def calc_metric(self, save_dir):
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        pred_df = pd.read_csv(os.path.join(save_dir, 'results', 'predictions.csv'))

        max_tr_dist = 10**10
        thresh_trans = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        thresh_rot = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
        metrics = []

        for i_thresh in range(10):
            keep_gt = False
            thresh_trans_dist = thresh_trans[i_thresh]
            thresh_rot_dist = thresh_rot[i_thresh]

            results, scores = [], []
            for i_img in range(len(pred_df)):
                img_id = pred_df.loc[i_img, 'ImageId']
                gt = train_df.loc[train_df['ImageId'] == img_id, 'PredictionString'].values[0]
                gt = parse_annot_str(str(gt))
                pred = pred_df.loc[i_img, 'PredictionString']
                pred = parse_pred_str(pred)
                    
                for pred_car in sorted(pred, key=lambda x: x['score'], reverse=True):
                    # find nearest GT
                    min_tr_dist = max_tr_dist
                    min_idx = -1
                    gt_idxs = list(range(len(gt)))
                    for i_gt in gt_idxs:
                        gt_car = gt[i_gt]
                        tr_dist = trans_dist(pred_car['location'], gt_car['location'])
                        if tr_dist < min_tr_dist:
                            min_tr_dist = tr_dist
                            min_rt_dist = rot_dist(pred_car['rotation'], gt_car['rotation'])
                            min_idx = i_gt
                    # set the result
                    if min_tr_dist < thresh_trans_dist and min_rt_dist < thresh_rot_dist:
                        if not keep_gt and min_idx > -1:
                            gt.pop(min_idx)
                        results.append(1)
                    else:
                        results.append(0)
                    scores.append(pred_car['score'])

            if np.sum(results) > 0:
                n_gt = sum([len(cars) for cars in self.anns])
                n_tp = np.sum(results)
                recall = n_tp / n_gt
                ap = average_precision_score(results, scores) * recall
            else:
                ap = 0 # maybe mean value ?
            metrics.append(ap)
            print('thresh {}: rot: {}, trans: {}, AP: {}'.format(
                i_thresh, thresh_rot_dist, thresh_trans_dist, ap))

        mAP = np.mean(metrics)
        print('mean AP:', mAP)

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        if self.split == 'val':
            self.calc_metric(save_dir)
