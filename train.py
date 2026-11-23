import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss, PixelLoss, ensure_vgg_weights
from models import GENERATORS, get_generator
from models.common import psnr, ssim
from models.discriminator import Discriminator
from utils import get_device

parser = argparse.ArgumentParser(description="Train Super Resolution Models")
parser.add_argument(
    "--model_name",
    default="espcn",
    type=str,
    choices=sorted(GENERATORS),
    help="name of the super resolution model",
)
parser.add_argument(
    "--crop_size", default=88, type=int, help="training images crop size"
)
parser.add_argument(
    "--upscale_factor",
    default=4,
    type=int,
    choices=[2, 4, 8],
    help="super resolution upscale factor",
)
parser.add_argument("--num_epochs", default=100, type=int, help="train epoch number")
parser.add_argument(
    "--warmup_epochs",
    "--warmup_batches",
    default=500,
    type=int,
    help="number of generator pre-training (warm-up) epochs before GAN training "
    "(--warmup_batches is the old name of this option)",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed for a reproducible run (default: not seeded)",
)
# Accepted so that old commands keep working; training always uses a single device.
parser.add_argument("--num_gpu", default=1, type=int, help=argparse.SUPPRESS)


def validate(netG, val_loader, device, out_path, epoch, save_desc):
    """Score netG on the validation set and save grids of (bicubic, HR, SR) images."""
    netG.eval()
    os.makedirs(out_path, exist_ok=True)
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {"ssims": 0, "psnrs": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results["batch_sizes"] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            valing_results["ssims"] += ssim(sr, hr) * batch_size
            valing_results["psnrs"] += psnr(sr, hr) * batch_size
            valing_results["psnr"] = valing_results["psnrs"] / valing_results["batch_sizes"]
            valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]
            val_bar.set_description(
                desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                % (valing_results["psnr"], valing_results["ssim"])
            )

            val_images.extend(
                [
                    display_transform()(val_hr_restore.squeeze(0)),
                    display_transform()(hr.cpu().squeeze(0)),
                    display_transform()(sr.cpu().squeeze(0)),
                ]
            )
        val_images = torch.split(torch.stack(val_images), 15)
        for index, image in enumerate(tqdm(val_images, desc=save_desc), start=1):
            image = vutils.make_grid(image, nrow=3, padding=5)
            vutils.save_image(
                image, out_path + "epoch_%d_index_%d.png" % (epoch, index), padding=5
            )
    return valing_results


if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)  # seeds the CPU and all CUDA devices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    MODEL_NAME = opt.model_name
    WARMUP_EPOCHS = opt.warmup_epochs
    device = get_device()
    ckpt_dir = os.path.join("epochs", MODEL_NAME)

    os.makedirs(ckpt_dir, exist_ok=True)
    # Fail now, not after the warm-up phase, if the VGG16 weights can't be downloaded.
    ensure_vgg_weights()

    train_set = TrainDatasetFromFolder(
        "data/DIV2K_train_HR", crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )
    val_set = ValDatasetFromFolder("data/DIV2K_valid_HR", upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=64, shuffle=True
    )
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = get_generator(MODEL_NAME)(1, UPSCALE_FACTOR)
    optimizerG = optim.Adam(netG.parameters())
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    pixel_criterion = PixelLoss()
    netG.to(device)
    pixel_criterion.to(device)

    print(f"# pretraining the generator for {WARMUP_EPOCHS} epochs")
    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}
        netG.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            real_img = target.to(device)
            z = data.to(device)
            fake_img = netG(z)
            netG.zero_grad()

            pre_g_loss = pixel_criterion(fake_img, real_img)
            pre_g_loss.backward()

            optimizerG.step()

            running_results["loss"] += pre_g_loss.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss: %.4f"
                % (
                    epoch,
                    WARMUP_EPOCHS,
                    running_results["loss"] / running_results["batch_sizes"],
                )
            )

        if epoch % 20 == 0 or epoch == WARMUP_EPOCHS:
            validate(
                netG,
                val_loader,
                device,
                "pre_training_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/",
                epoch,
                "[saving pre training results]",
            )
            torch.save(
                netG.state_dict(),
                os.path.join(ckpt_dir, "pre_netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch)),
            )

    # Rebuild the generator (with a fresh optimizer) and load the last warm-up weights.
    # The rebuild also keeps the random number sequence of the original training runs.
    netG = get_generator(MODEL_NAME)(1, UPSCALE_FACTOR)
    optimizerG = optim.Adam(netG.parameters())
    netG.to(device)

    if WARMUP_EPOCHS > 0:
        netG.load_state_dict(
            torch.load(
                os.path.join(
                    ckpt_dir, "pre_netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, WARMUP_EPOCHS)
                ),
                map_location=device,
                weights_only=True,
            )
        )

    netD = Discriminator(1)
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss()
    discriminator_criterion = torch.nn.BCELoss()

    netD.to(device)
    generator_criterion.to(device)
    discriminator_criterion.to(device)

    optimizerD = optim.Adam(netD.parameters())

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target.to(device)
            z = data.to(device)
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img)
            fake_out = netD(fake_img)

            d_loss_real = discriminator_criterion(real_out, torch.ones_like(real_out))
            d_loss_fake = discriminator_criterion(fake_out, torch.zeros_like(fake_out))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.mean().item() * batch_size
            running_results["g_score"] += fake_out.mean().item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        if epoch % 200 == 0 or epoch == NUM_EPOCHS:
            validate(
                netG,
                val_loader,
                device,
                "training_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/",
                epoch,
                "[saving training results]",
            )
            # save model parameters
            torch.save(
                netG.state_dict(),
                os.path.join(ckpt_dir, "netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch)),
            )
            torch.save(
                netD.state_dict(),
                os.path.join(ckpt_dir, "netD_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch)),
            )
