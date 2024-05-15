
# SISR-GAN-Variants

This repository contains the code and resources for the project "Evaluation of Alternative Generator Models in GAN-based Single Image Super-Resolution." The aim of this project is to investigate the performance of alternative lightweight generator models (ESPCN, FSRCNN, and IDN) within the SRGAN framework for single image super-resolution (SISR) tasks.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction
Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution image from a single low-resolution input. Generative Adversarial Networks (GANs) have shown promising results in SISR tasks. This project explores the integration of lightweight generator models into the SRGAN framework to improve efficiency and performance.

## Models
The following generator models are investigated in this study:
- ESPCN: Efficient Sub-Pixel Convolutional Neural Network
- FSRCNN: Fast Super-Resolution Convolutional Neural Network
- IDN: Information Distillation Network
- SRResNet: Super-Resolution Residual Network (baseline)

## Dataset
The models are trained and evaluated on the DIV2K dataset, a widely used benchmark for SISR tasks. The dataset consists of high-quality images with diverse contents and realistic degradations.

## Experimental Setup
The experiments are conducted using the following setup:
- Pre-training of generator models for 400 epochs
- GAN training for 2,000 epochs
- Image cropping to 128x128 patches for augmentation
- Evaluation metrics: PSNR and SSIM
- Testing datasets: Set5 and Set14

## Usage
To train and evaluate the models, follow these steps:
1. Clone the repository: `git clone https://github.com/sephiroth7712/SISR-GAN-Variants.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare the dataset: Download the DIV2K dataset and place it in the `data/` directory
4. Train the models: 
	```bash
	python3 train.py --model_name espcn --crop_size 128 --upscale_factor 4 --num_epochs 8000 --warmup_batches 5000
	```
5. Evaluate the models:
	```bash
	python3 test.py --model_name espcn --upscale_factor 4 --ckpt ./path/to/model/checkpoint.pth --image_name ./path/to/image.png
	```

## Acknowledgements
We would like to acknowledge the following repositories for their valuable code contributions:
- [SRGAN](https://github.com/leftthomas/SRGAN) by leftthomas
- [ESPCN-PyTorch](https://github.com/Lornatang/ESPCN-PyTorch) by Lornatang
- [IDN-Pytorch](https://github.com/yjn870/IDN-pytorch) by yjn870
- [FSRCNN-Pytorch](https://github.com/Lornatang/FSRCNN-PyTorch) by Lornatang
