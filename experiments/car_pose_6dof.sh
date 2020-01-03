cd src
# train
python main.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars --batch_size 6 --num_epochs 70 --lr_step 45,60 --gpus 1 --test
cd ..
