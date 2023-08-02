#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 7
#SBATCH -G 1 
#SBATCH --qos normal
#SBATCH --time=10:00:00 
#SBATCH --job-name=KITTI.DGCNN.POT2K.BS8
#SBATCH --error=%x-%j.error
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=Your_Email@EMAIL.XX 

path_to_conda="/home/users/Your_User_Name/miniconda3 (OR the CONDA You have)"
module purge
source ${path_to_conda}/bin/activate 
conda activate LiDAR-RGBD-SLAM

echo "== JOB STARTED AT $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

KITTI_TRAIN_DATASET_PATH="/path/to/KITTI/dataset/For_Training"
KITTI_TEST_DATASET_PATH="/path/to/KITTI/dataset/For_Testing"
LOGDIR="./logs_KITTI"

cd ..

python3 train.py --acceleratorType cuda --model_name dcp --dataset_type Kitti --noise_type downsample --train_data_dir "$KITTI_TRAIN_DATASET_PATH" --test_data_dir "$KITTI_TEST_DATASET_PATH" --log_dir $LOGDIR --periodic_save_dir $LOGDIR --max_epochs 200 --num_points 1024 --train_batch_size 16 --val_batch_size 16 --head partial
