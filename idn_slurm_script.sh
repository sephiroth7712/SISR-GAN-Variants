#!/bin/bash
#SBATCH --job-name idn_gan
#SBATCH --output idn_gan_output.log
#SBATCH --error idn_gan_error.log
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 8G

module load anaconda3

conda activate espgan_pytorch

cd /lustre/home/kurisummootr/CMSC636/Project/ESPGAN-PyTorch

python3 train.py --model_name idn --crop_size 128 --upscale_factor 4 --num_epochs 2000 --warmup_batches 400