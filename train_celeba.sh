#!/bin/bash
#SBATCH --job-name=train-celeba
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/train-celeba-%j.err
#SBATCH --output=runs/train-celeba-%j.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate csbm
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32884 \
    scripts/train.py \
        --config './configs/celeba.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
