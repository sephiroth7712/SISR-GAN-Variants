import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TestDatasetFromFolder, display_transform
from models import GENERATORS
from models.common import psnr, psnr_y, ssim, ssim_y
from utils import get_device, load_generator

# Rows of the results CSV: these first, then any other dataset prefixes found.
DATASET_ORDER = ["Set5", "Set14"]
# Columns of the results CSV. pre_* = generator after pre-training, no prefix = after
# GAN training, bicubic_* = bicubic upscaling. *_y = Y channel of 8-bit images with a
# border of `upscale_factor` pixels removed (the protocol of most SR papers).
METRICS = [
    "pre_psnr", "pre_ssim", "psnr", "ssim",
    "pre_psnr_y", "pre_ssim_y", "psnr_y", "ssim_y",
    "bicubic_psnr", "bicubic_ssim", "bicubic_psnr_y", "bicubic_ssim_y",
    "runtime",
]


def synchronize(device):
    """Wait for queued CUDA work, so wall-clock timings cover the whole forward pass."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="Test Benchmark Datasets")
    parser.add_argument(
        "--upscale_factor",
        default=4,
        type=int,
        choices=[2, 4, 8],
        help="super resolution upscale factor",
    )
    parser.add_argument(
        "--model_name",
        default="espcn",
        type=str,
        choices=sorted(GENERATORS),
        help="generator model name",
    )
    parser.add_argument(
        "--pre_ckpt",
        default="pre_netG_epoch_4_400.pth",
        type=str,
        help="generator checkpoint after pre-training "
        "(a file name in epochs/<model_name>/ or a path)",
    )
    parser.add_argument(
        "--gan_ckpt",
        default="netG_epoch_4_2000.pth",
        type=str,
        help="generator checkpoint after GAN training "
        "(a file name in epochs/<model_name>/ or a path)",
    )
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name
    PRE_CKPT = opt.pre_ckpt
    GAN_CKPT = opt.gan_ckpt

    device = get_device()
    model_pre = load_generator(MODEL_NAME, UPSCALE_FACTOR, PRE_CKPT, device)
    model_gan = load_generator(MODEL_NAME, UPSCALE_FACTOR, GAN_CKPT, device)

    test_set = TestDatasetFromFolder("data/test", upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(
        dataset=test_set, num_workers=4, batch_size=1, shuffle=False
    )
    test_bar = tqdm(test_loader, desc="[testing benchmark datasets]")

    out_path = "benchmark_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/"
    os.makedirs(out_path, exist_ok=True)

    # Warm up, so one-off CUDA initialisation and kernel selection are not timed.
    if len(test_set) > 0:
        with torch.no_grad():
            warmup_input = test_set[0][1].unsqueeze(0).to(device)
            for _ in range(3):
                model_pre(warmup_input)

    # results[<dataset prefix>][<metric>] -> list of per-image values
    results = {}
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        with torch.no_grad():
            image_name = image_name[0]
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            hr_restore_img = hr_restore_img.to(device)

            # Both checkpoints share one architecture, so timing one of them is enough.
            synchronize(device)
            start_time = time.perf_counter()
            pre_sr_image = model_pre(lr_image)
            synchronize(device)
            runtime = time.perf_counter() - start_time
            sr_image = model_gan(lr_image)

            pre_psnr_score = psnr(pre_sr_image, hr_image).cpu()
            pre_ssim_score = ssim(pre_sr_image, hr_image).cpu()
            psnr_score = psnr(sr_image, hr_image).cpu()
            ssim_score = ssim(sr_image, hr_image).cpu()
            extra_scores = {
                "pre_psnr_y": psnr_y(pre_sr_image, hr_image, UPSCALE_FACTOR),
                "pre_ssim_y": ssim_y(pre_sr_image, hr_image, UPSCALE_FACTOR),
                "psnr_y": psnr_y(sr_image, hr_image, UPSCALE_FACTOR),
                "ssim_y": ssim_y(sr_image, hr_image, UPSCALE_FACTOR),
                "bicubic_psnr": psnr(hr_restore_img, hr_image),
                "bicubic_ssim": ssim(hr_restore_img, hr_image),
                "bicubic_psnr_y": psnr_y(hr_restore_img, hr_image, UPSCALE_FACTOR),
                "bicubic_ssim_y": ssim_y(hr_restore_img, hr_image, UPSCALE_FACTOR),
            }

            test_images = torch.stack(
                [
                    display_transform()(lr_image.squeeze(0)),
                    display_transform()(hr_image.data.cpu().squeeze(0)),
                    display_transform()(hr_restore_img.cpu().squeeze(0)),
                    display_transform()(pre_sr_image.data.cpu().squeeze(0)),
                    display_transform()(sr_image.data.cpu().squeeze(0)),
                ]
            )
            image = vutils.make_grid(test_images, nrow=5, padding=10)
            vutils.save_image(
                image,
                out_path
                + image_name.split(".")[0]
                + "_psnr_%.4f_ssim_%.4f." % (psnr_score, ssim_score)
                + image_name.split(".")[-1],
                padding=5,
            )

            # save psnr\ssim, grouped by the dataset prefix of the file name
            scores = results.setdefault(
                image_name.split("_")[0], {metric: [] for metric in METRICS}
            )
            scores["pre_psnr"].append(pre_psnr_score)
            scores["pre_ssim"].append(pre_ssim_score)
            scores["psnr"].append(psnr_score)
            scores["ssim"].append(ssim_score)
            for metric, value in extra_scores.items():
                scores[metric].append(value.cpu())
            scores["runtime"].append(runtime)

    datasets = [name for name in DATASET_ORDER if name in results]
    datasets += sorted(name for name in results if name not in DATASET_ORDER)
    saved_results = {metric: [] for metric in METRICS}
    for name in datasets:
        for metric in METRICS:
            values = np.array(results[name][metric])
            saved_results[metric].append(
                np.median(values) if metric == "runtime" else values.mean()
            )

    out_path = "statistics/"
    os.makedirs(out_path, exist_ok=True)
    data_frame = pd.DataFrame(saved_results, index=datasets)
    data_frame.to_csv(
        out_path + MODEL_NAME + "_srf_" + str(UPSCALE_FACTOR) + "_test_results.csv",
        index_label="DataSet",
    )


if __name__ == "__main__":
    main()
