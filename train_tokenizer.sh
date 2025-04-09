#!/bin/bash
#SBATCH --job-name=train-tokenizer
#SBATCH --partition=ais-cpu
#SBATCH --error=runs/train-tokenizer-%j.err
#SBATCH --output=runs/train-tokenizer-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

source activate csbm
python scripts/train_tokenizer.py \
    --config './configs/yelp.yaml' \
    --data_dir './data'
