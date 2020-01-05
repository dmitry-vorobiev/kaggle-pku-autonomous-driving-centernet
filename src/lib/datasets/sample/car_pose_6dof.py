from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch.utils.data as data

from utils.camera import create_camera_matrix, euler_angles_to_rotation_matrix, euler_angles_to_quaternions, proj_point, calc_bbox
from utils.car_models import car_id2name
from utils.image import get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.kaggle_cars_utils import pad_img_sides, parse_annot_str


class CarPose6DoFDataset(data.Dataset):
    def __getitem__(self, index):
        inp_w, inp_h = self.opt.input_w, self.opt.input_h
        out_w, out_h = self.opt.output_w, self.opt.output_h
        pad_w_pct = self.opt.pad_img_ratio
        num_classes = self.opt.num_classes
        calib = self.calib

        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        img = cv2.imread(img_path)
        if self.opt.img_bottom_half:
            h_mid = img.shape[0] // 2
            img = img[h_mid:]

        height, width = img.shape[0], img.shape[1]
        c = np.array([width / 2., height / 2.])
        if self.opt.keep_res:
            s = np.array([inp_w, inp_h], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        aug = False
        # if False and self.split == 'train' and np.random.random() < self.opt.aug_ddd:
        #     aug = True
        #     sf = self.opt.scale
        #     cf = self.opt.shift
        #     s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        #     c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        #     c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

        trans_input = get_affine_transform(c, s, 0, [inp_w, inp_h])
        inp = cv2.warpAffine(
            img, trans_input, (inp_w, inp_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        hm = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        rot = np.zeros((self.max_objs, 4), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        dep = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

        trans_output = get_affine_transform(c, s, 0, [out_w, out_h])
        if pad_w_pct > 0:
            inp = pad_img_sides(inp, pad_w_pct)
            hm = pad_img_sides(hm, pad_w_pct)

        anns = self.find_car_poses(img_id)
        num_objs = min(len(anns), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            cls_id = 0 # ann['car_id']
            ct = proj_point(ann['location'], calib)
            bbox = ann['bbox']
            if pad_w_pct > 0:
                offset = int(width * pad_w_pct / 2)
                ct[0] += offset
                bbox[[0, 2]] += offset
            if self.opt.img_bottom_half:
                ct[1] -= height
                bbox[[1, 3]] -= height
            ct = affine_transform(ct, trans_output)
            ct[0] = np.clip(ct[0], 0, hm.shape[2] - 1)
            ct[1] = np.clip(ct[1], 0, hm.shape[1] - 1)
            ct_int = ct.astype(np.int32)
            # if flipped:
            #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm.shape[2] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm.shape[1] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                draw_gaussian(hm[0], ct, radius)

                wh[k] = 1. * w, 1. * h
                rot[k] = euler_angles_to_quaternions(ann['rotation'])
                dep[k] = ann['location'][-1]
                ind[k] = ct_int[1] * hm.shape[2] + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1 if not aug else 0
                rot_mask[k] = 1
                # x, y, score, r1-r4, depth, wh?, cls
                gt_det.append([ct[0], ct[1], 1, *rot[k].tolist(), dep[k], cls_id])
                if self.opt.reg_bbox:
                    gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]

        ret = {'input': inp, 'hm': hm, 'dep': dep, 'rot': rot, 'ind': ind,
               'reg_mask': reg_mask, 'rot_mask': rot_mask}
        if self.opt.reg_bbox:
            ret.update({'wh': wh})
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not ('train' in self.split):
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 11), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
                    'image_path': img_path, 'img_id': img_id}
            ret['meta'] = meta
        return ret

    def find_car_poses(self, img_id):
        cond = self.df['ImageId'] == img_id
        cars = self.df.loc[cond, 'PredictionString'].values[0]
        cars = parse_annot_str(str(cars))
        for car in cars:
            car_name = car_id2name[car['car_id']].name
            car_model = self.car_models[car_name]
            bbox = calc_bbox(car_model['vertices'], car['rotation'], 
                             car['location'], self.calib)
            car['bbox'] = np.array(bbox)
        return cars
