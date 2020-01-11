cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_14.pth --resume \
 --batch_size 12 --master_batch_size 6 \
 --num_epochs 20 --lr 3e-5 --lr_step 10,15 \
 --flip -1 --aug_shift -1 --shift 0.04 \
 --center_thresh 0.25 --peak_thresh 0.25 \
 --gpus 0,1 --debug 4 --test
cd ..
