#!/bin/bash
#SBATCH --job-name=eval-cmnist
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/eval-cmnist-%j.err
#SBATCH --output=runs/eval-cmnist-%j.log
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

# 2 steps: ./experiments/images/gaussian/dim_32_aplha_0.01_09.01.25_10:11:45
# 4 steps: ./experiments/images/gaussian/dim_32_aplha_0.01_29.01.25_10:02:47
# 10 steps: ./experiments/images/gaussian/dim_32_aplha_0.01_28.01.25_15:30:39
# 25 steps: ./experiments/images/gaussian/dim_32_aplha_0.01_29.01.25_20:37:01
# 100 steps: ./experiments/images/cmnist/gaussian/dim_32_aplha_0.01_21.05.25_05:58:37
# uniform 25 steps: ./experiments/images/cmnist/uniform/dim_32_aplha_0.01_26.03.25_18:13:18
# uniform 25 steps: ./experiments/images/cmnist/uniform/dim_32_aplha_0.05_26.03.25_18:09:45

source activate csbm
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32183 \
    scripts/eval.py \
        --exp_path './experiments' \
        --iteration 3 \
        --data_dir './data'
