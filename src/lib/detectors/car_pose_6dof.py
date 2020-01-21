from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch


from models.decode import car_pose_6dof_decode
from models.utils import flip_tensor
from utils import car_models
from utils.geometry import create_camera_matrix
from utils.image import get_affine_transform, pad_img_sides
from utils.post_process import car_6dof_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y, draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector

class CarPose6DoFDetector(BaseDetector):
    def __init__(self, opt):
        super(CarPose6DoFDetector, self).__init__(opt)
        self.calib = create_camera_matrix()

    def set_models(self, models_3D):
        self.models = models_3D

    def pre_process(self, image, scale, calib=None):
        if self.opt.img_bottom_half:
            x_offset = image.shape[0] // 2
            image = image[x_offset:]

        height, width = image.shape[0:2]
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([width / 2, height / 2], dtype=np.float32)
        if self.opt.keep_res:
            s = np.array([inp_width, inp_height], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(
            image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.mean) / self.std
        inp_image = inp_image.transpose(2, 0, 1)
        if self.opt.pad_img_ratio > 0:
            inp_image = pad_img_sides(inp_image, self.opt.pad_img_ratio)

        calib = np.array(calib, dtype=np.float32) if calib is not None \
                else self.calib
        images = torch.from_numpy(inp_image[np.newaxis, ...])
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.opt.down_ratio, 
                'out_width': inp_width // self.opt.down_ratio,
                'calib': calib}
        return images, meta
  
    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            wh = output['wh'] if self.opt.reg_bbox else None
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = car_pose_6dof_decode(
                output['hm'], output['rot'], output['dep'],
                wh=wh, reg=reg, K=self.opt.K)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = car_6dof_post_process(
            dets.copy(), [meta['c']], [meta['s']], [meta['calib']], 
            self.opt)
        self.this_calib = meta['calib']
        return detections[0]

    def merge_outputs(self, detections):
        results = detections[0]
        for j in range(1, self.num_classes + 1):
            if len(results[j]) > 0:
                keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = car_6dof_post_process(
            dets.copy(), [meta['c']], [meta['s']], [meta['calib']], 
            self.opt)
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.std + self.mean) * 255).astype(np.uint8)
        car_name = car_models.models[0].name
        car_model = self.models[car_name]
        if self.opt.debug_heatmap:
            pred = debugger.gen_colormap(
                output['hm'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.render_cars(
            img, dets[0], car_model, [meta['c']], [meta['s']], 
            [meta['calib']], self.opt, '3d_pred')
  
    def show_results(self, debugger, image, results):
        debugger.show_all_imgs(pause=self.pause)
