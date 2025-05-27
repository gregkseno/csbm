#!/bin/bash
#SBATCH --job-name=eval-celeba
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507-3
#SBATCH --error=runs/eval-celeba-%j.err
#SBATCH --output=runs/eval-celeba-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

# New
# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.005_14.03.25_00:25:49
# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.01_13.03.25_19:29:49 

# Old
# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.005_27.01.25_21:56:36
# ./experiments/quantized_images/celeba/uniform/dim_128_aplha_0.01_14.01.25_21:22:30

source activate csbm
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32481 \
    scripts/eval.py \
        --exp_path './experiments/quantized_images/celeba/uniform/dim_128_aplha_0.005_27.01.25_21:56:36' \
        --iteration 4 \
        --data_dir './data'
