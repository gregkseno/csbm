#!/bin/bash
#SBATCH --job-name=quantize-afhq
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2507
#SBATCH --error=runs/quantize-afhq-%j.err
#SBATCH --output=runs/quantize-afhq-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

python scripts/quatize.py --config './configs/afhq.yaml' --data_dir './data'
