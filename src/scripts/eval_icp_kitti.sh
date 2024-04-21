#! /bin/bash

source env.sh

cd ..

python3 eval.py --model_name icp --dataset_type Kitti --noise_type downsample --train_data_dir "$KITTI_DATASET_PATH" --test_data_dir "$KITTI_TEST_DATASET_PATH" --log_dir $LOGDIR --gpu 1 --num_points 10000 --val_batch_size 1
