#!/bin/bash
#SBATCH --job-name=train-amazon
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507-4
#SBATCH --error=runs/train-amazon-%j.err
#SBATCH --output=runs/train-amazon-%j.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
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
        --config './configs/amazon.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
