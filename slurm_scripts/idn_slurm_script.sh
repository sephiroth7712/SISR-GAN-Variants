#!/bin/bash
#SBATCH --job-name idn_gan
#SBATCH --output idn_gan_output.log
#SBATCH --error idn_gan_error.log
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 8G

# Adjust the partition, module and conda environment names for your cluster.
module load anaconda3

conda activate espgan_pytorch

cd "$SLURM_SUBMIT_DIR"

python3 train.py --model_name idn --crop_size 128 --upscale_factor 4 --num_epochs 2000 --warmup_epochs 400
