cd src
# test
python test.py car_pose_6dof --exp_id car_pose_default --dataset kaggle_cars \
 --xyz_mask \
 --load_model ../exp/car_pose_6dof/car_pose_default/model_14_avg.pth --resume \
 --center_thresh 0.3 --peak_thresh 0.45 --vis_thresh 0.45 --K 50 \
 --gpus 1 --trainval --not_prefetch_test --debug 0 --render_cars
cd ..