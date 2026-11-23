# SISR-GAN-Variants

[![lint](https://github.com/sephiroth7712/SISR-GAN-Variants/actions/workflows/lint.yml/badge.svg)](https://github.com/sephiroth7712/SISR-GAN-Variants/actions/workflows/lint.yml)

This repository contains the code and resources for the project "Evaluation of Alternative Generator Models in GAN-based Single Image Super-Resolution." The aim of this project is to investigate the performance of alternative lightweight generator models (ESPCN, FSRCNN, and IDN) within the SRGAN framework for single image super-resolution (SISR) tasks.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Usage](#usage)
- [Development](#development)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [License](#license)

## Introduction
Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution image from a single low-resolution input. Generative Adversarial Networks (GANs) have shown promising results in SISR tasks. This project explores the integration of lightweight generator models into the SRGAN framework to improve efficiency and performance.

## Models
The following generator models are investigated in this study. All of them are trained against the same SRGAN discriminator (5,215,425 parameters).

| Model | `--model_name` | Generator parameters (×4) | Paper |
|---|---|---|---|
| ESPCN: Efficient Sub-Pixel Convolutional Neural Network | `espcn` | 37,200 | Shi et al., CVPR 2016 |
| FSRCNN: Fast Super-Resolution Convolutional Neural Network | `fsrcnn` | 24,683 | Dong et al., ECCV 2016 |
| IDN: Information Distillation Network | `idn` | 590,915 | Hui et al., CVPR 2018 |
| SRResNet: Super-Resolution Residual Network (baseline) | `srgan` | 734,219 | Ledig et al., CVPR 2017 |

The `srgan` generator follows [leftthomas/SRGAN](https://github.com/leftthomas/SRGAN) and has 5 residual blocks; the SRGAN paper uses 16. The code also includes an EDSR generator (`--model_name edsr`, 43,090,467 parameters at ×4; Lim et al., CVPR Workshops 2017).

## Dataset
The models are trained and evaluated on the DIV2K dataset, a widely used benchmark for SISR tasks. The dataset consists of high-quality images with diverse contents and realistic degradations. Download the high-resolution training and validation images from the [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/). The Set5 and Set14 benchmark images are used for testing; they are available from several sources, for example the `data/` folder of the [SelfExSR repository](https://github.com/jbhuang0604/SelfExSR) (use the `*_HR.png` files).

The scripts expect this layout, relative to the repository root:

```
data/
├── DIV2K_train_HR/      # 800 training images (0001.png ... 0800.png)
├── DIV2K_valid_HR/      # 100 validation images (0801.png ... 0900.png)
└── test/
    └── SRF_4/           # one folder per upscale factor: SRF_2, SRF_4, SRF_8
        ├── data/        # one file per test image, e.g. Set5_baby.png (only the names are used)
        └── target/      # the high-resolution test images, with the same file names
```

Prefix every benchmark image with its dataset name (`Set5_`, `Set14_`): `test_benchmark.py` groups its results by the part of the file name before the first underscore. For each file name in `data/` it loads the image with that name from `target/` and creates the low-resolution input from it by bicubic downsampling, so `data/` can simply hold copies of the `target/` images.

## Experimental Setup
The experiments are conducted using the following setup:
- Pre-training of generator models for 400 epochs
- GAN training for 2,000 epochs
- Image cropping to 128x128 patches for augmentation
- Evaluation metrics: PSNR and SSIM
- Testing datasets: Set5 and Set14

## Results
A qualitative comparison of the results can be found [here](https://imgversus.com/a/w3S22qqV).

### Evaluation protocol
`test_benchmark.py` runs the generators in inference mode (`model.eval()`) and writes, for each dataset, the mean of every per-image score (the median for `runtime`) to `statistics/<model_name>_srf_<scale>_test_results.csv`:

| Columns | Meaning |
|---|---|
| `pre_psnr`, `pre_ssim` | Generator after pre-training: PSNR and SSIM on RGB values in [0, 1], whole image. |
| `psnr`, `ssim` | Generator after GAN training, same protocol. |
| `pre_psnr_y`, `pre_ssim_y`, `psnr_y`, `ssim_y` | The same outputs scored the way most super-resolution papers report results: rounded to 8 bits, Y channel (ITU-R BT.601), `<scale>` pixels removed from each border. |
| `bicubic_psnr`, `bicubic_ssim`, `bicubic_psnr_y`, `bicubic_ssim_y` | Bicubic upscaling of the same low-resolution input, as a baseline. |
| `runtime` | Median seconds per forward pass of the pre-trained generator (both checkpoints share its architecture), after 3 warm-up passes. |

## Usage
Run all commands from the repository root.

1. Clone the repository: `git clone https://github.com/sephiroth7712/SISR-GAN-Variants.git`
2. Create an environment with Python 3.9–3.12 and install the dependencies:
	```bash
	conda create -n espgan_pytorch python=3.10
	conda activate espgan_pytorch
	pip install -r requirements.txt
	```
	The SLURM scripts activate an environment with this name. On Windows, PyPI only provides a CPU build of PyTorch; use the selector on [pytorch.org](https://pytorch.org/get-started/locally/) to install a CUDA build.
3. Prepare the datasets as described in [Dataset](#dataset).
4. Train a model:
	```bash
	python3 train.py --model_name espcn --crop_size 128 --upscale_factor 4 --warmup_epochs 400 --num_epochs 2000
	```
	`--warmup_epochs` is the number of epochs the generator is pre-trained with a pixel (MSE) loss before GAN training starts; `--warmup_batches` is accepted as an older name for it. Checkpoints are written to `epochs/<model_name>/`: `pre_netG_epoch_<scale>_<epoch>.pth` every 20 pre-training epochs, and `netG_epoch_<scale>_<epoch>.pth` and `netD_epoch_<scale>_<epoch>.pth` every 200 GAN epochs. The last epoch of each phase is always saved. Sample validation images go to `pre_training_results/` and `training_results/`.

	On a SLURM cluster, submit the matching script instead. `slurm_scripts/` holds the command used for each model:
	```bash
	sbatch slurm_scripts/espcn_slurm_script.sh
	```
	The GAN phase uses a VGG16 perceptual loss whose ImageNet weights (about 530 MB) are downloaded when training starts. If your compute nodes have no internet access, download them once on a login node:
	```bash
	python3 -c "from loss import ensure_vgg_weights; ensure_vgg_weights()"
	```
5. Upscale a single image:
	```bash
	python3 test.py --model_name espcn --upscale_factor 4 --ckpt netG_epoch_4_2000.pth --image_name path/to/image.png
	```
	`--ckpt` is a file name in `epochs/<model_name>/` or a path to a `.pth` file. The result is saved as `out_srf_<scale>_<image name>` in the current directory. The script uses the GPU when one is available; add `--test_mode CPU` to run on the CPU.
6. Evaluate on Set5 and Set14:
	```bash
	python3 test_benchmark.py --model_name espcn --upscale_factor 4 --pre_ckpt pre_netG_epoch_4_400.pth --gan_ckpt netG_epoch_4_2000.pth
	```
	This scores the generator after pre-training and after GAN training. Side-by-side images go to `benchmark_results/<model_name>/SRF_<scale>/` and the average scores per dataset to `statistics/<model_name>_srf_<scale>_test_results.csv`.
7. Upscale a video:
	```bash
	python3 test_video.py --model_name espcn --upscale_factor 4 --ckpt netG_epoch_4_2000.pth --video_name path/to/video.mp4
	```
	This writes the upscaled video to `out_srf_<scale>_<video file name without extension>.avi` in the current directory, and a side-by-side comparison with bicubic upscaling to `compare_srf_<scale>_<video file name without extension>.avi`.
8. Upscale an image with plain bicubic interpolation, for a visual baseline:
	```bash
	python3 bicubic.py --image path/to/image.png --scale 4 --output upscaled_image.png
	```

## Development
The code is checked with [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`). GitHub Actions runs `ruff check .` and byte-compiles every file on each push. To run the check locally:
```bash
pip install ruff==0.17.6
ruff check .
```

## Acknowledgements
We would like to acknowledge the following repositories for their valuable code contributions:
- [SRGAN](https://github.com/leftthomas/SRGAN) by leftthomas
- [ESPCN-PyTorch](https://github.com/Lornatang/ESPCN-PyTorch) by Lornatang
- [IDN-Pytorch](https://github.com/yjn870/IDN-pytorch) by yjn870
- [FSRCNN-Pytorch](https://github.com/Lornatang/FSRCNN-PyTorch) by Lornatang
- [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) by sanghyun-son

Their licenses and copyright notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## References
- W. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," CVPR 2016. [arXiv:1609.05158](https://arxiv.org/abs/1609.05158)
- C. Dong, C. C. Loy, X. Tang, "Accelerating the Super-Resolution Convolutional Neural Network," ECCV 2016. [arXiv:1608.00367](https://arxiv.org/abs/1608.00367)
- Z. Hui, X. Wang, X. Gao, "Fast and Accurate Single Image Super-Resolution via Information Distillation Network," CVPR 2018. [arXiv:1803.09454](https://arxiv.org/abs/1803.09454)
- C. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," CVPR 2017. [arXiv:1609.04802](https://arxiv.org/abs/1609.04802)
- B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," CVPR Workshops 2017. [arXiv:1707.02921](https://arxiv.org/abs/1707.02921)
- E. Agustsson, R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," CVPR Workshops 2017 (DIV2K).
- M. Bevilacqua, A. Roumy, C. Guillemot, M.-L. Alberi-Morel, "Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding," BMVC 2012 (Set5).
- R. Zeyde, M. Elad, M. Protter, "On Single Image Scale-Up Using Sparse-Representations," Curves and Surfaces 2010 (Set14).

## License
This project is released under the [Apache License 2.0](LICENSE). It includes code adapted from the projects listed in [Acknowledgements](#acknowledgements); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their licenses.
