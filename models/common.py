import numpy as np
import torch
from torch.nn import functional, Module, Conv2d, PReLU, PixelShuffle, BatchNorm2d
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


# ---------------------------------------
#  Utility
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: functional.pixel_shuffle(x, scale)


class ResidualBlock(Module):
    def __init__(self, channels, kernel_size=3, padding=1, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        if batch_norm:
            self.bn1 = BatchNorm2d(channels)
        self.prelu = PReLU()
        self.conv2 = Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        if batch_norm:
            self.bn2 = BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.batch_norm:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.batch_norm:
            residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = Conv2d(
            in_channels, in_channels * up_scale**2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffle(up_scale)
        self.prelu = PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
