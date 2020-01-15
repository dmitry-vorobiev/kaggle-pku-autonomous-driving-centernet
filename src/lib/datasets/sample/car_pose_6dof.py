from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch.utils.data as data

from utils.geometry import create_camera_matrix, euler_angles_to_rotation_matrix, euler_angles_to_quaternions, proj_point, quaternion_upper_hemispher, rotate
from utils.image import get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, pad_img_sides


class CarPose6DoFDataset(data.Dataset):
    def __getitem__(self, index):
        inp_w, inp_h = self.opt.input_w, self.opt.input_h
        out_w, out_h = self.opt.output_w, self.opt.output_h
        pad_w_pct = self.opt.pad_img_ratio
        num_classes = self.opt.num_classes
        calib = self.calib

        # store all 2D point shifts before any resize
        xy_off = np.array([0, 0])
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        img = cv2.imread(img_path)
        if self.opt.img_bottom_half:
            half_h = img.shape[0] // 2
            img = img[half_h:]
            xy_off[1] -= half_h

        height, width = img.shape[0], img.shape[1]
        c = np.array([width / 2., height / 2.])
        if self.opt.keep_res:
            s = np.array([inp_w, inp_h], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        aug, flipped = False, False
        if self.split == 'train':
            if np.random.random() < self.opt.aug_scale:
                aug = True
                sf = self.opt.scale
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1)
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [inp_w, inp_h])
        inp = cv2.warpAffine(
            img, trans_input, (inp_w, inp_h), flags=cv2.INTER_LINEAR)
        inp = self.tfms(inp)
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
        subcls = np.zeros((self.max_objs), dtype=np.uint8)

        trans_output = get_affine_transform(c, s, 0, [out_w, out_h])
        if pad_w_pct > 0:
            inp = pad_img_sides(inp, pad_w_pct)
            hm = pad_img_sides(hm, pad_w_pct)
            xy_off[0] += int(width * pad_w_pct / 2)

        anns = self.anns[index]
        num_objs = min(len(anns), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            cls_id = 0  # ann['car_id']
            ct = proj_point(ann['location'], calib)
            rot_eul = np.copy(ann['rotation'])
            bbox = ann['bbox']
            if flipped:
                ct[0] = width-1 - ct[0]
                bbox[[0, 2]] = width-1 - bbox[[2, 0]]
                rot_eul[[1, 2]] = -rot_eul[[1, 2]]
            ct += xy_off
            bbox += np.hstack([xy_off, xy_off])

            ct = affine_transform(ct, trans_output)
            ct0 = np.copy(ct)
            ct[0] = np.clip(ct[0], 0, hm.shape[2] - 1)
            ct[1] = np.clip(ct[1], 0, hm.shape[1] - 1)
            ct_int = ct.astype(np.int32)
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm.shape[2] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm.shape[1] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            pad_px = (hm.shape[2] - out_w) // 2
            if h > 0 and w > 0 and bbox[2] > pad_px and bbox[0] < hm.shape[2] - pad_px:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                draw_gaussian(hm[0], ct, radius)

                wh[k] = 1. * w, 1. * h
                rot_eul[1] = rotate(rot_eul[1], np.pi/2)
                rot_eul[2] = rotate(rot_eul[2], np.pi)
                q = euler_angles_to_quaternions(rot_eul)[0]
                q = quaternion_upper_hemispher(q)
                rot[k] = q
                dep[k] = ann['location'][-1]
                ind[k] = ct_int[1] * hm.shape[2] + ct_int[0]
                reg[k] = ct0 - ct_int
                reg_mask[k] = 1 if not aug else 0
                rot_mask[k] = 1
                # x, y, score, r1-r4, depth, wh?, cls
                gt_det.append(
                    [ct[0], ct[1], 1, *rot[k].tolist(), dep[k], cls_id])
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
