import numpy as np
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

# from utils import resolve_and_plot

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, lr.unsqueeze(0)[0])


def resolve(model, lr_batch):
    lr_batch = lr_batch.type(torch.FloatTensor)
    sr_batch = model(lr_batch)
    sr_batch = torch.clamp(sr_batch, 0, 255)
    sr_batch = torch.round(sr_batch)
    sr_batch = sr_batch.type(torch.FloatTensor)


def evaluate(model, dataset):
    psnr_values = []
    ssim_values = []
    i = 0
    for lr, hr in dataset:
        sr = resolve(model, lr)

        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        ssim_value = ssim(hr, sr)[0]
        ssim_values.append(ssim_value)

    return torch.mean(psnr_values), torch.mean(ssim_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return peak_signal_noise_ratio(x1, x2)


def ssim(x1, x2):
    return structural_similarity_index_measure(x1, x2)


def pixel_shuffle(scale):
    return lambda x: torch.nn.functional.pixel_shuffle(x, scale)
