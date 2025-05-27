#!/bin/bash
#SBATCH --job-name=eval-celeba
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507-3
#SBATCH --error=runs/eval-celeba-%j.err
#SBATCH --output=runs/eval-celeba-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.01_13.03.25_19:29:49 
# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.005_14.03.25_00:25:49

source activate csbm
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32484 \
    scripts/eval.py \
        --exp_path './experiments/quantized_images/celeba/uniform/dim_128_aplha_0.005_14.03.25_00:25:49' \
        --iteration 4 \
        --data_dir './data'
