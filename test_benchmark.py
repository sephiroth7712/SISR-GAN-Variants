import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import TestDatasetFromFolder, display_transform
from models import get_generator
from models.common import psnr, ssim
import time

parser = argparse.ArgumentParser(description="Test Benchmark Datasets")
parser.add_argument(
    "--upscale_factor", default=4, type=int, help="super resolution upscale factor"
)
parser.add_argument(
    "--model_name",
    default="espcn",
    type=str,
    help="generator model name",
)
parser.add_argument(
    "--pre_ckpt",
    default="pre_netG_epoch_4_400.pth",
    type=str,
    help="generator model checkpoint file after pre-training",
)
parser.add_argument(
    "--gan_ckpt",
    default="netG_epoch_4_2000.pth",
    type=str,
    help="generator model checkpoint file after gan-training",
)
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
PRE_CKPT = opt.pre_ckpt
GAN_CKPT = opt.gan_ckpt

results = {
    "Set5": {"pre_psnr": [], "pre_ssim": [], "psnr": [], "ssim": [], "runtime": []},
    "Set14": {"pre_psnr": [], "pre_ssim": [], "psnr": [], "ssim": [], "runtime": []},
}

model_pre = get_generator(MODEL_NAME)(1, UPSCALE_FACTOR)
model_gan = get_generator(MODEL_NAME)(1, UPSCALE_FACTOR)
if torch.cuda.is_available():
    model_pre = model_pre.cuda()
    model_gan = model_gan.cuda()

model_pre.load_state_dict(torch.load("epochs/" + MODEL_NAME + "/" + PRE_CKPT))
model_gan.load_state_dict(torch.load("epochs/" + MODEL_NAME + "/" + GAN_CKPT))

test_set = TestDatasetFromFolder("data/test", upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc="[testing benchmark datasets]")

out_path = "benchmark_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    with torch.no_grad():
        image_name = image_name[0]
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        start_time = time.time()
        pre_sr_image = model_pre(lr_image)
        runtime = time.time() - start_time
        sr_image = model_gan(lr_image)

        pre_psnr_score = psnr(pre_sr_image, hr_image).cpu()
        pre_ssim_score = ssim(pre_sr_image, hr_image).cpu()
        psnr_score = psnr(sr_image, hr_image).cpu()
        ssim_score = ssim(sr_image, hr_image).cpu()

        test_images = torch.stack(
            [
                display_transform()(lr_image.squeeze(0)),
                display_transform()(hr_image.data.cpu().squeeze(0)),
                display_transform()(hr_restore_img.cpu().squeeze(0)),
                display_transform()(pre_sr_image.data.cpu().squeeze(0)),
                display_transform()(sr_image.data.cpu().squeeze(0)),
            ]
        )
        image = utils.make_grid(test_images, nrow=5, padding=10)
        utils.save_image(
            image,
            out_path
            + image_name.split(".")[0]
            + "_psnr_%.4f_ssim_%.4f." % (psnr_score, ssim_score)
            + image_name.split(".")[-1],
            padding=5,
        )

        # save psnr\ssim
        results[image_name.split("_")[0]]["pre_psnr"].append(pre_psnr_score)
        results[image_name.split("_")[0]]["pre_ssim"].append(pre_ssim_score)
        results[image_name.split("_")[0]]["psnr"].append(psnr_score)
        results[image_name.split("_")[0]]["ssim"].append(ssim_score)
        results[image_name.split("_")[0]]["runtime"].append(runtime)

out_path = "statistics/"
saved_results = {"pre_psnr": [], "pre_ssim": [], "psnr": [], "ssim": [], "runtime": []}
for item in results.values():
    pre_psnr_scores = np.array(item["pre_psnr"])
    pre_ssim_scores = np.array(item["pre_ssim"])
    psnr_scores = np.array(item["psnr"])
    ssim_scores = np.array(item["ssim"])
    runtimes = np.array(item["runtime"])
    if (
        len(pre_psnr_scores) == 0
        or len(pre_ssim_scores) == 0
        or len(psnr_scores) == 0
        or len(ssim_scores) == 0
        or len(runtimes) == 0
    ):
        pre_psnr_scores = "No data"
        pre_ssim_scores = "No data"
        psnr_scores = "No data"
        ssim_scores = "No data"
        runtimes = "No data"
    else:
        pre_psnr_mean = pre_psnr_scores.mean()
        pre_ssim_mean = pre_ssim_scores.mean()
        psnr_mean = psnr_scores.mean()
        ssim_mean = ssim_scores.mean()
        runtime_median = np.median(runtimes)
    saved_results["pre_psnr"].append(pre_psnr_mean)
    saved_results["pre_ssim"].append(pre_ssim_mean)
    saved_results["psnr"].append(psnr_mean)
    saved_results["ssim"].append(ssim_mean)
    saved_results["runtime"].append(runtime_median)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(
    out_path + MODEL_NAME + "_srf_" + str(UPSCALE_FACTOR) + "_test_results.csv",
    index_label="DataSet",
)
