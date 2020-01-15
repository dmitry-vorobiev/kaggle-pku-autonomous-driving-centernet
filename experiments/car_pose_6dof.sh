cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --batch_size 12 --master_batch_size 6 \
 --num_epochs 20 --lr 3e-5 --lr_step 10,15 \
 --use_swa --swa_start 3600 --swa_freq 50 \
 --flip 0.5 --aug_shift -1 --shift 0.04 --aug_scale 0.2 --scale 0.15 \
 --aug_blur 0.15 --aug_gamma 0.2 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.08 --contrast_limit 0.08 \
 --center_thresh 0.3 --peak_thresh 0.3 --vis_thresh 0.3 \
 --gpus 0,1 --debug 4
cd ..
