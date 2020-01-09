cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --batch_size 12 --master_batch_size 6 --num_epochs 70 --lr_step 45,60 \
 --flip -1 --aug_shift -1 --shift 0.04 \
 --center_thresh 0.25 \
 --gpus 0,1 --debug 0 --resume --test
cd ..
