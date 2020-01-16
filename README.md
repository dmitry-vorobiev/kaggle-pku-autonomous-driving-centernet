# Peking University/Baidu - Autonomous Driving

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

(OPTIONAL) You may want to install NVIDIA Apex to be able to use mixed precision training. Please, visit [this repo](https://github.com/NVIDIA/apex) for more info.

## Run train

~~~
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --arch hourglass \
 --mixed_precision --opt_level O1 --max_loss_scale 8192 \
 --batch_size 7 --master_batch_size 3 \
 --num_epochs 25 --lr 7e-5 --lr_step 15,20 \
 --use_swa --swa_start 10_000 --swa_freq 50 \
 --flip 0.5 --aug_scale 0.2 --scale 0.15 \
 --aug_blur 0.15 --aug_gamma 0.2 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.08 --contrast_limit 0.08 \
 --center_thresh 0.3 \
 --gpus 0,1
~~~

## Run evaluation on validation set

Just add **--test --resume** flags to train command line. If you want to load specific weights, pass it to **--load_model** flag.

## Run predictions on test set

~~~
python test.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_15.pth --resume \
 --peak_thresh 0.3 --K 50 \
 --gpus 1 --trainval --not_prefetch_test
~~~