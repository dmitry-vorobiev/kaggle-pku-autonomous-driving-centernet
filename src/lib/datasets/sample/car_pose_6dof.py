from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch.utils.data as data

from utils.camera import create_camera_matrix, euler_angles_to_rotation_matrix
from utils.car_models import car_id2name
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.kaggle_cars_utils import parse_annot_str


class CarPose6DoFDataset(data.Dataset):
    def __getitem__(self, index):
        calib = self.calib
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        img = cv2.imread(img_path)
        if self.opt.img_bottom_half:
            h_mid = img.shape[0] // 2
            img = img[h_mid:]

        input_shape = (self.opt.input_w, self.opt.input_h)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
        if self.opt.keep_res:
            s = np.array(input_shape, dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        aug = False
        if False and self.split == 'train' and np.random.random() < self.opt.aug_ddd:
            aug = True
            sf = self.opt.scale
            cf = self.opt.shift
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

        trans_input = get_affine_transform(c, s, 0, input_shape)
        inp = cv2.warpAffine(img, trans_input, input_shape, flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        out_w, out_h = self.opt.output_w, self.opt.output_h
        num_classes = self.opt.num_classes
        trans_output = get_affine_transform(c, s, 0, [out_w, out_h])

        hm = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        dep = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

        anns = self.find_car_poses(img_id)
        bboxes = self.calc_2D_boxes(anns, [width, height])
        num_objs = min(len(anns), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                        draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            cls_id = ann['car_id']
            bbox = np.array(bboxes[k])
            # if flipped:
            #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, out_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, out_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, 
                               (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[0], ct, radius)

                wh[k] = 1. * w, 1. * h
                gt_det.append([ct[0], ct[1], 1] + ann['pose'].tolist() + [cls_id])
                if self.opt.reg_bbox:
                    gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
                dep[k] = ann['pose'][-1]
                ind[k] = ct_int[1] * out_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1 if not aug else 0
                rot_mask[k] = 1

        ret = {'input': inp, 'hm': hm, 'dep': dep, 'ind': ind, 
               'reg_mask': reg_mask, 'rot_mask': rot_mask}
        if self.opt.reg_bbox:
            ret.update({'wh': wh})
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not ('train' in self.split):
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                    np.zeros((1, 18), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
                    'image_path': img_path, 'img_id': img_id}
            ret['meta'] = meta
        return ret

    def find_car_poses(self, img_id):
        cars = self.df.loc[self.df['ImageId'] == img_id, 'PredictionString'].values[0]
        return parse_annot_str(str(cars))

    def calc_2D_boxes(self, anns, image_shape):
        """Calc 2D bboxes using CAD models"""
        boxes = []
        for i, ann in enumerate(anns):
            car_name = car_id2name[ann['car_id']].name
            car = self.car_models[car_name]
            pose = np.array(ann['pose'])
            
            # project 3D points to 2d image plane
            rot_mat = euler_angles_to_rotation_matrix(pose[:3])
            rvect, _ = cv2.Rodrigues(rot_mat)
            imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], 
                                            self.calib[:,:3], distCoeffs=None)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            
            x1, y1, x2, y2 = (imgpts[:, 0].min(), imgpts[:, 1].min(), 
                              imgpts[:, 0].max(), imgpts[:, 1].max())
            if self.opt.img_bottom_half:
                halved_h = int(image_shape[1])
                y1 -= halved_h
                y2 -= halved_h
            boxes.append([x1, y1, x2, y2])
        return boxes