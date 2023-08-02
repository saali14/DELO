#! /bin/bash

source env.sh

cd ..

python3 eval.py --model_name identity --noise_type downsample --train_data_dir '/home/daniel/Desktop/shared/MasterThesis/datasets/3DMatch/3DMatch_5cm/test' --test_data_dir $TEST_DATASET_PATH --log_dir $LOGDIR