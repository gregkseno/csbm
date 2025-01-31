#!/bin/bash
#SBATCH --job-name=images_disc_sbm
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

# export LC_ALL="en_US.UTF-8"
# export LD_LIBRARY_PATH="/usr/lib64-nvidia"
# export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
# ldconfig /usr/lib64-nvidia
source ~/anaconda3/bin/activate disc_sbm

accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32883 \
    scripts/train.py \
        --config './configs/images.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'