#! /bin/bash

source env.sh

cd ..

python3 eval.py --model_name identity --noise_type downsample --train_data_dir $KITTI_DATASET_PATH --test_data_dir $TEST_DATASET_PATH --log_dir $LOGDIR
