# Peking University/Baidu - Autonomous Driving

This is a fork of [CenterNet repo](https://github.com/xingyizhou/CenterNet) with an additional pipeline and some minor functionality added.
To learn more about the competition please visit [kaggle.com](https://www.kaggle.com/c/pku-autonomous-driving).

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

(OPTIONAL) You may want to install NVIDIA Apex to be able to use mixed precision training. Please, visit [this repo](https://github.com/NVIDIA/apex) for more info.

## Setup data

By default for *--dataset kaggle_cars* the script will search data in *data/pku-autonomous-driving*. You can create a symlink like this one:

~~~
$ ln -s /home/user/projects/kaggle_cars/input/pku-autonomous-driving /home/user/projects/kaggle_cars/centernet/data
~~~

To split images for train and validation create folder *split* with *train.txt, val.txt, ignore.txt*, containing names of the files with extensions. Something like this:

~~~
$ mkdir split
$ ls train_images | head -n 3800 > split/train.txt
$ ls train_images | tail -n 462 > split/val.txt
~~~

If you intend to use 3D location masks, you have to generate it first using this script. By default it creates *train_3d_masks* folder in the dataset directory. You may overwrite this behaviour with *--xyz_masks_dir*.

~~~
$ python src/tools/prepare_3d_loc_masks.py --num_workers 4 --norm_xyz 519.834,689.119,3502.94
~~~

## Run train

~~~
$ python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --xyz_mask \
 --batch_size 10 --master_batch_size 5 --num_grad_accum 2 \
 --num_epochs 20 --lr 1e-4 --lr_step 15,25 --weight_decay 0.01 \
 --use_swa --swa_start 12_000 --swa_freq 20 --swa_manual \
 --aug_blur 0.25 --blur_limit 3,9 \
 --aug_noise 0.2 --noise_scale 0.03,0.09 \
 --aug_hue 0.2 --hue_shift_limit 30 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.1 --contrast_limit 0.1 \
 --center_thresh 0.3 \
 --gpus 0,1
~~~

## Run evaluation

Just add *--test --resume* flags to the command line above. If you want to load specific weights, pass it to *--load_model* .

## Save averaged model weights

Run train script with *--test --save_avg_weights*. The averaged weights file name ends with *_avg* suffix.

## Run predictions on test set

~~~
$ python test.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --xyz_mask \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_20_avg.pth --resume \
 --peak_thresh 0.5 --K 50 \
 --gpus 1 --trainval --not_prefetch_test
~~~

## Debug / visualisations

Use *--render_cars* with *--debug 4* to see rendered car models. Just add it these parameters to train/test command lines. Adding *--debug_heatmap* will also visualise heatmaps.

## Citation

~~~
@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
~~~