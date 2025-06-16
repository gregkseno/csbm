#!/bin/bash
#SBATCH --job-name=quantize-celeba
#SBATCH --partition=ais-gpu
#SBATCH --error=runs/quantize-celeba-%j.err
#SBATCH --output=runs/quantize-celeba-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

source activate csbm
python scripts/quatize.py --config './configs/celeba.yaml' --data_dir './data'
