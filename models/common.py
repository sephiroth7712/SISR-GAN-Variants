# ResidualBlock and UpsampleBLock are based on leftthomas/SRGAN
# (https://github.com/leftthomas/SRGAN), MIT License. Modified for this project;
# see THIRD_PARTY_NOTICES.md.
import torch
from torch.nn import Module, Conv2d, PReLU, PixelShuffle, BatchNorm2d
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    """PSNR of RGB images with values in [0, 1]."""
    return peak_signal_noise_ratio(x1, x2, data_range=1.0)


def ssim(x1, x2):
    """SSIM of RGB images with values in [0, 1]."""
    return structural_similarity_index_measure(x1, x2, data_range=1.0)


def quantize(img):
    """Round images with values in [0, 1] to 8-bit levels, as when saving them as PNG."""
    return torch.round(img.clamp(0.0, 1.0) * 255.0) / 255.0


def rgb_to_y(img):
    """Y (luma) channel of RGB images in [0, 1], ITU-R BT.601 as in MATLAB's rgb2ycbcr.

    Returns values in [16/255, 235/255], shape (N, 1, H, W).
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def shave(img, border):
    """Remove ``border`` pixels from every side of the image."""
    h, w = img.shape[-2:]
    return img[..., border:h - border, border:w - border]


def psnr_y(sr, hr, scale):
    """PSNR as usually reported in SR papers: 8-bit images, Y channel, ``scale``-pixel border removed."""
    return psnr(shave(rgb_to_y(quantize(sr)), scale), shave(rgb_to_y(quantize(hr)), scale))


def ssim_y(sr, hr, scale):
    """SSIM as usually reported in SR papers: 8-bit images, Y channel, ``scale``-pixel border removed."""
    return ssim(shave(rgb_to_y(quantize(sr)), scale), shave(rgb_to_y(quantize(hr)), scale))


# ---------------------------------------
#  Building blocks
# ---------------------------------------


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
