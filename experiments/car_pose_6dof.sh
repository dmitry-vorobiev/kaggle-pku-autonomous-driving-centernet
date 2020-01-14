cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_10.pth --resume \
 --batch_size 12 --master_batch_size 6 \
 --num_epochs 20 --lr 3e-5 --lr_step 10,15 \
 --use_swa --swa_start 3600 --swa_freq 50 \
 --flip -1 --aug_shift -1 --shift 0.04 \
 --center_thresh 0.25 --peak_thresh 0.25 --vis_thresh 0.3 \
 --gpus 0,1 --debug 4
cd ..
