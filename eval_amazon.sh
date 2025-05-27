#!/bin/bash
#SBATCH --job-name=eval-amazon
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507-3
#SBATCH --error=runs/eval-amazon-%j.err
#SBATCH --output=runs/eval-amazon-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

# ./experiments/texts/amazon/uniform/dim_100_aplha_0.005_16.05.25_23:12:19
# ./experiments/texts/amazon/uniform/dim_100_aplha_0.01_20.05.25_17:47:55

source activate csbm
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32384 \
    scripts/eval.py \
        --exp_path './experiments/texts/amazon/uniform/dim_100_aplha_0.01_20.05.25_17:47:55' \
        --iteration 5 \
        --data_dir './data'
