#!/bin/bash
#SBATCH --job-name=train-tokenizer
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507
#SBATCH --error=runs/train-tokenizer-%j.err
#SBATCH --output=runs/train-tokenizer-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

source activate csbm
python scripts/train_tokenizer.py \
    --config './configs/yelp.yaml' \
    --data_dir './data'
