#!/bin/bash
#SBATCH --job-name=quantized_images_disc_sbm
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

module load compilers/gcc-12.2.0
module load gpu/cuda-12.3
source activate disc_sbm

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32884 \
    scripts/train.py \
        --config './configs/quantized_images.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'