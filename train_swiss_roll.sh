#!/bin/bash
#SBATCH --job-name=train-swiss-roll
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/train-swiss-roll-%j.err
#SBATCH --output=runs/train-swiss-roll-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate csbm
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=29004 \
    scripts/train.py \
        --config './configs/swiss_roll.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
