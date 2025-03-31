#!/bin/bash
#SBATCH --job-name=train-yelp
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507
#SBATCH --error=runs/train-yelp-%j.err
#SBATCH --output=runs/train-yelp-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate csbm
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32884 \
    scripts/train.py \
        --config './configs/yelp.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
