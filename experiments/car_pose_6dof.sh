cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --batch_size 12 --master_batch_size 6 \
 --num_epochs 20 --lr 7e-5 --lr_step 10,15 \
 --use_swa --swa_start 10 --swa_freq 5 \
 --mixed_precision --opt_level O1 --max_loss_scale 8192 \
 --flip -1 --aug_shift -1 --shift 0.04 \
 --center_thresh 0.25 --peak_thresh 0.25 --vis_thresh 0.3 \
 --gpus 0,1 --debug 4
cd ..
