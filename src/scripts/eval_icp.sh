#! /bin/bash

source env.sh

cd ..

#python3 eval.py --model_name icp --dataset_type Sparse3DMatch --noise_type rigid3d --train_data_dir $DATASET_PATH --test_data_dir $TEST_DATASET_PATH --log_dir $LOGDIR --gpu 1

python3 eval.py --model_name icp --dataset_type 3DMatch --noise_type downsample --train_data_dir $DATASET_PATH --test_data_dir $TEST_DATASET_PATH --log_dir $LOGDIR --gpu 1 --num_points 10000 --val_batch_size 1
