#!/bin/bash
#SBATCH --job-name=train-cmnist
#SBATCH --partition=gpu_a100
#SBATCH --error=runs/train-cmnist-%j.err
#SBATCH --output=runs/train-cmnist-%j.log
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

source activate csbm
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32883 \
    scripts/train.py \
        --config './configs/cmnist.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
