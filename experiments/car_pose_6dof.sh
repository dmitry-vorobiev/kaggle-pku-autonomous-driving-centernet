cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_14.pth --resume \
 --xyz_mask --dlm_weight 5 --rot_weight 10 \
 --batch_size 6 --master_batch_size 6 --num_grad_accum 2 \
 --num_epochs 20 --lr 7e-5 --lr_step 10,15 \
 --use_swa --swa_start 6000 --swa_freq 50 --swa_manual \
 --flip -1 --aug_scale 0.2 --scale 0.15 \
 --aug_blur 0.15 --aug_gamma 0.2 \
 --aug_brightness_contrast 0.3 --brightness_limit 0.08 --contrast_limit 0.08 \
 --center_thresh 0.3 --peak_thresh 0.3 --vis_thresh 0.3 \
 --gpus 1 --debug 0 --no_color_aug --render_cars --test --save_avg_weights
cd ..
