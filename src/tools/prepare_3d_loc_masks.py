import os
import sys
import cv2
import numpy as np
from multiprocessing import Pool

from opts import opts
from datasets.dataset_factory import get_dataset
from utils import car_models
from utils.geometry import affine_to_world_pts, proj_world_pts

def create_mask(path, img_anns, models_3D, calib, opt,
                shape=(2710, 3384), 
                max_objs=100):
    height, width = shape
    xy_off = np.array([0, 0])
    if opt.img_bottom_half:
        height = height // 2
        xy_off[1] -= height

    mask = np.zeros((3, height, width), dtype=np.float32)
    num_objs = min(len(img_anns), max_objs)
    norms = opt.norm_xyz
    img_anns.sort(key=lambda x: x['location'][2], reverse=True)

    for k in range(num_objs):
        ann = img_anns[k]
        car_name = car_models.car_id2name[ann['car_id']].name
        model_3D = models_3D[car_name]
        
        pts_world = affine_to_world_pts(
            model_3D['vertices'], ann['rotation'], ann['location'])
        pts_2d = proj_world_pts(pts_world, calib)
        pts_2d += xy_off
        pts_world= pts_world.T
                
        for face in model_3D['faces'] - 1:
            pts_face_2d = np.int32(pts_2d[face]).reshape((-1, 1, 2))
            pts_face_3d = np.int32(pts_world[face])
            for dim in range(3):
                v = pts_face_3d[:,dim].mean() / norms[dim]
                cv2.drawContours(mask[dim], [pts_face_2d], 0, v, -1)

    np.savez_compressed(path, [mask])


def proc(idx):
    img_id = dataset.images[idx]
    img_anns = dataset.anns[idx]
    path = os.path.join(render_dir, img_id)
    create_mask(
        path, img_anns, dataset.models, dataset.calib, opt, shape=shape)
    return img_id


if __name__ == '__main__':
    sys.argv.append('car_pose_6dof')
    opt = opts().parse()
    opt.trainval = True

    render_dir = opt.xyz_masks_dir
    if not os.path.isdir(render_dir):
        render_dir = os.path.join(
            opt.data_dir, 'pku-autonomous-driving', 'train_3d_masks')
    if not os.path.isdir(render_dir):
        os.mkdir(render_dir)

    Dataset = get_dataset('kaggle_cars', 'car_pose_6dof')
    dataset = Dataset(opt, 'train')
    shape = (2710, 3384)
    
    n = len(dataset)
    pool = Pool(processes=opt.num_workers)
    for i, img_id in enumerate(pool.imap(proc, range(n))):
        print('[%4d / %d] %s' % (i, n, img_id))
