cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_14_avg.pth --resume \
 --xyz_mask \
 --dlm_weight 5 --rot_weight 10 \
 --batch_size 5 --master_batch_size 5 --num_grad_accum 2 \
 --num_epochs 20 --lr 7e-5 --lr_step 10,15 \
 --swa_start 6000 --swa_freq 50 --swa_manual \
 --flip -1 \
 --aug_blur 0.25 --blur_limit 3,9 \
 --aug_noise 0.2 --noise_scale 0.03,0.09 \
 --aug_hue 0.2 --hue_shift_limit 30 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.1 --contrast_limit 0.1 \
 --center_thresh 0.3 --peak_thresh 0.3 --vis_thresh 0.3 \
 --gpus 1 --debug 0 --render_cars
cd ..
