cd src
# test
python test.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_14.pth --resume \
 --center_thresh 0.25 --peak_thresh 0.25 --vis_thresh 0.25 --K 50 \
 --gpus 1 --trainval --not_prefetch_test --debug 0
cd ..