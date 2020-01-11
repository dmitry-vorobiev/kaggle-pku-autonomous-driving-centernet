from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import car_pose_6dof_decode
from models.utils import _sigmoid
from utils import car_models
from utils.debugger import Debugger
from utils.post_process import car_6dof_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class CarPose6DoFLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CarPose6DoFLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = L1Loss()
        # TODO: another rotation loss
        self.crit_rot = BinRotLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, dep_loss, rot_loss = 0, 0, 0
        wh_loss, off_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.

            if opt.eval_oracle_dep:
                output['dep'] = torch.from_numpy(gen_oracle_map(
                    batch['dep'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    opt.output_w, opt.output_h)).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.dep_weight > 0:
                dep_loss += self.crit_reg(output['dep'], batch['reg_mask'],
                                          batch['ind'], batch['dep']) / opt.num_stacks
            if opt.rot_weight > 0:
                rot_pred = F.normalize(output['rot'], p=2, dim=1)
                rot_loss += self.crit_reg(rot_pred, batch['rot_mask'],
                                          batch['ind'], batch['rot']) / opt.num_stacks
            if opt.reg_bbox and opt.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['rot_mask'],
                                         batch['ind'], batch['wh']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['rot_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
        loss = (opt.hm_weight * hm_loss + opt.dep_weight * dep_loss +
                opt.rot_weight * rot_loss + opt.wh_weight * wh_loss +
                opt.off_weight * off_loss)

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss,
                      'rot_loss': rot_loss, 'wh_loss': wh_loss,
                      'off_loss': off_loss}
        return loss, loss_stats


class CarPose6DoFTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CarPose6DoFTrainer, self).__init__(
            opt, model, optimizer=optimizer)

    def set_models(self, models_3D):
        self.models = models_3D

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dep_loss', 'rot_loss',
                       'wh_loss', 'off_loss']
        loss = CarPose6DoFLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = car_pose_6dof_decode(
            output['hm'], output['rot'], output['dep'],
            wh=wh, reg=reg, K=opt.K)
        # x, y, score, r1-r4, depth, wh?, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        c = batch['meta']['c'].detach().numpy()
        s = batch['meta']['s'].detach().numpy()
        calib = batch['meta']['calib'].detach().numpy()
        # yaw, pitch, roll, x, y, z, wh?, score
        dets_pred = car_6dof_post_process(
            dets.copy(), c, s, calib, opt)
        dets_gt = car_6dof_post_process(
            batch['meta']['gt_det'].detach().numpy().copy(),
            c, s, calib, opt)
        car_name = car_models.models[0].name
        car_model = self.models[car_name]
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
                                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
            if opt.debug_gen_hm:
                pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'hm_pred')
                debugger.add_blend_img(img, gt, 'hm_gt')
            debugger.add_car_masks(img, dets_pred[0], car_model, c, s, calib, opt, '3d_pred')
            debugger.add_car_masks(img, dets_gt[0], car_model, c, s, calib, opt, '3d_gt')
            if opt.debug == 4:
                prefix = '{}_{}_'.format(iter_id, batch['meta']['img_id'][0])
                debugger.save_all_imgs(opt.debug_dir, prefix=prefix)
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = car_pose_6dof_decode(
            output['hm'], output['rot'], output['dep'],
            wh=wh, reg=reg, K=opt.K)
        # x, y, score, r1-r4, depth, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        c = batch['meta']['c'].detach().numpy()
        s = batch['meta']['s'].detach().numpy()
        calib = batch['meta']['calib'].detach().numpy()
        # yaw, pitch, roll, x, y, z, wh?, score
        dets_pred = car_6dof_post_process(
            dets.copy(), c, s, calib, opt)
        img_id = batch['meta']['img_id'][0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
            results[img_id][j] = results[img_id][j][keep_inds]
