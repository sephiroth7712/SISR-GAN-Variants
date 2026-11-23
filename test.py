import argparse
import datetime
import os

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from models import GENERATORS
from utils import get_device, load_generator


def main():
    parser = argparse.ArgumentParser(description="Test Single Image")
    parser.add_argument(
        "--upscale_factor",
        default=4,
        type=int,
        choices=[2, 4, 8],
        help="super resolution upscale factor",
    )
    parser.add_argument(
        "--test_mode",
        default=None,
        type=str,
        choices=["GPU", "CPU"],
        help="run on the GPU or the CPU (default: the GPU when one is available)",
    )
    parser.add_argument(
        "--image_name", type=str, required=True, help="low resolution image to upscale"
    )
    parser.add_argument(
        "--model_name",
        default="espcn",
        type=str,
        choices=sorted(GENERATORS),
        help="generator model name",
    )
    parser.add_argument(
        "--ckpt",
        default="pre_netG_epoch_4_400.pth",
        type=str,
        help="generator checkpoint: a file name in epochs/<model_name>/ or a path to a .pth file",
    )
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    IMAGE_NAME = opt.image_name
    MODEL_NAME = opt.model_name
    CKPT = opt.ckpt

    if opt.test_mode is None:
        device = get_device()
    else:
        device = torch.device("cuda" if opt.test_mode == "GPU" else "cpu")

    model = load_generator(MODEL_NAME, UPSCALE_FACTOR, CKPT, device)

    image = Image.open(IMAGE_NAME).convert("RGB")
    with torch.no_grad():
        image = ToTensor()(image).unsqueeze(0).to(device)

        start = datetime.datetime.now()
        out = model(image)
        elapsed = datetime.datetime.now() - start
        print("Time: " + str(elapsed) + "s")
        # round to the nearest 8-bit level (ToPILImage on a float tensor truncates)
        out_img = ToPILImage()(torch.round(out[0].clamp(0, 1) * 255).byte().cpu())
        out_img.save(
            "out_srf_" + str(UPSCALE_FACTOR) + "_" + os.path.basename(IMAGE_NAME)
        )


if __name__ == "__main__":
    main()
