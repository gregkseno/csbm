#!/bin/bash
#SBATCH --job-name=toy_disc_sbm
#SBATCH --partition=gpu
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
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
    --main_process_port=29004 \
    scripts/train.py \
        --config './configs/toy.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
# --multi_gpu \     --cpu \