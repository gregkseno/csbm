#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --partition=gpu
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

module load compilers/gcc-12.2.0
module load gpu/cuda-12.3
source activate disc_sbm

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32489 \
    generate.py \
        --exp_path './experiments/quantized_images/uniform/dim_128_aplha_0.005_27.01.25_21:56:36' \
        --iterations 4 3 2
