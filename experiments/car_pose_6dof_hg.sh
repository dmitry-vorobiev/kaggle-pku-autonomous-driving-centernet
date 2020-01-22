cd src
# train
python main.py car_pose_6dof --exp_id hg --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/hg/model_2.pth --resume \
 --arch hourglass --xyz_mask \
 --xyz_weight 5 --rot_weight 10 \
 --batch_size 7 --master_batch_size 3 --num_grad_accum 4 \
 --num_epochs 10 --lr 5e-5 --lr_step 12,15 --weight_decay 0.01 \
 --use_swa --swa_start 6000 --swa_freq 50 --swa_manual \
 --mixed_precision --opt_level O1 --max_loss_scale 8192 \
 --flip -1 \
 --aug_blur 0.25 --blur_limit 3,9 \
 --aug_noise 0.2 --noise_scale 0.03,0.09 \
 --aug_hue 0.2 --hue_shift_limit 30 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.1 --contrast_limit 0.1 \
 --center_thresh 0.3 --peak_thresh 0.3 --vis_thresh 0.3 \
 --val_intervals 1 --save_all \
 --gpus 1,0 --debug 0
cd ..
