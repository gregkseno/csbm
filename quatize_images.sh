#!/bin/bash
#SBATCH --job-name=quantized_images_disc_sbm
#SBATCH --partition=ais-gpu
#SBATCH --reservation=icml2025
#SBATCH --nodelist=gn34
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00

module load compilers/gcc-12.2.0
module load gpu/cuda-12.3
source activate disc_sbm

python quatize.py --config './configs/quantized_images.yaml' --data_dir './data'