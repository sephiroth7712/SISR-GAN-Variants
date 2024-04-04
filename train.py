import argparse
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from data import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss, PixelLoss
from models.common import ssim, psnr
from models.discriminator import Discriminator
from models.ESPCN import ESPCN

parser = argparse.ArgumentParser(description="Train Super Resolution Models")
parser.add_argument(
    "--model_name", default="espcn", type=str, help="name of the super resolution model"
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
parser.add_argument("--num_gpu", default=1, type=int, help="Number of GPUs to use")
parser.add_argument(
    "--warmup_batches",
    default=500,
    type=int,
    help="Number of warmup batches for the generator",
)

if __name__ == "__main__":
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    MODEL_NAME = opt.model_name
    N_GPU = opt.num_gpu
    WARMUP_BATCHES = opt.warmup_batches

    train_set = TrainDatasetFromFolder(
        "data/DIV2K_train_HR", crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )
    val_set = ValDatasetFromFolder("data/DIV2K_valid_HR", upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=64, shuffle=True
    )
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = ESPCN(N_GPU, UPSCALE_FACTOR)
    optimizerG = optim.Adam(netG.parameters())
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    pixel_criterion = PixelLoss()

    if torch.cuda.is_available():
        netG.cuda()
        pixel_criterion.cuda()

    print(f"# pretraining the generator for {WARMUP_BATCHES} steps")
    for epoch in range(1, WARMUP_BATCHES + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}
        netG.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
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
                    WARMUP_BATCHES,
                    running_results["loss"] / running_results["batch_sizes"],
                )
            )

        netG.eval()
        out_path = (
            "pre_training_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/"
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if epoch % 500 == 0:
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {
                    "mse": 0,
                    "ssims": 0,
                    "psnr": 0,
                    "ssim": 0,
                    "batch_sizes": 0,
                }
                val_images = []
                for val_lr, val_hr_restore, val_hr in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results["batch_sizes"] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results["mse"] += batch_mse * batch_size
                    batch_ssim = ssim(sr, hr)
                    valing_results["ssims"] += batch_ssim * batch_size
                    valing_results["psnr"] = psnr(sr, hr)
                    valing_results["ssim"] = (
                        valing_results["ssims"] / valing_results["batch_sizes"]
                    )
                    val_bar.set_description(
                        desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                        % (valing_results["psnr"], valing_results["ssim"])
                    )

                    val_images.extend(
                        [
                            display_transform()(val_hr_restore.squeeze(0)),
                            display_transform()(hr.data.cpu().squeeze(0)),
                            display_transform()(sr.data.cpu().squeeze(0)),
                        ]
                    )
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc="[saving pre training results]")
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(
                        image,
                        out_path + "epoch_%d_index_%d.png" % (epoch, index),
                        padding=5,
                    )
                    index += 1

            torch.save(
                netG.state_dict(),
                "epochs/"
                + MODEL_NAME
                + "/pre_netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch),
            )

    # Reloading the Generator with latest weights after warmup
    netG = ESPCN(N_GPU, UPSCALE_FACTOR)
    optimizerG = optim.Adam(netG.parameters())
    if torch.cuda.is_available():
        netG.cuda()

    netG.load_state_dict(
        torch.load(
            "epochs/"
            + MODEL_NAME
            + "/pre_netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, WARMUP_BATCHES)
        )
    )

    # netG.load_state_dict(
    #     torch.load(
    #         "epochs/" + MODEL_NAME + "/pre_netG_epoch_%d_0.pth" % (UPSCALE_FACTOR)
    #     )
    # )

    netD = Discriminator(N_GPU)
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss()
    discriminator_criterion = torch.nn.BCELoss()

    if torch.cuda.is_available():
        netD.cuda()
        generator_criterion.cuda()
        discriminator_criterion.cuda()

    optimizerD = optim.Adam(netD.parameters())

    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

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
            g_update_first = True
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img)
            fake_out = netD(fake_img)

            d_loss_real = discriminator_criterion(real_out, torch.ones_like(real_out))
            d_loss_fake = discriminator_criterion(fake_out, torch.ones_like(fake_out))
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

        if epoch % 200 == 0:
            netG.eval()
            out_path = (
                "training_results/" + MODEL_NAME + "/SRF_" + str(UPSCALE_FACTOR) + "/"
            )
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {
                    "mse": 0,
                    "ssims": 0,
                    "psnr": 0,
                    "ssim": 0,
                    "batch_sizes": 0,
                }
                val_images = []
                for val_lr, val_hr_restore, val_hr in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results["batch_sizes"] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results["mse"] += batch_mse * batch_size
                    batch_ssim = ssim(hr, sr)
                    valing_results["ssims"] += batch_ssim * batch_size
                    valing_results["psnr"] = psnr(sr, hr)
                    valing_results["ssim"] = (
                        valing_results["ssims"] / valing_results["batch_sizes"]
                    )
                    val_bar.set_description(
                        desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                        % (valing_results["psnr"], valing_results["ssim"])
                    )

                    val_images.extend(
                        [
                            display_transform()(val_hr_restore.squeeze(0)),
                            display_transform()(hr.data.cpu().squeeze(0)),
                            display_transform()(sr.data.cpu().squeeze(0)),
                        ]
                    )
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc="[saving training results]")
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(
                        image,
                        out_path + "epoch_%d_index_%d.png" % (epoch, index),
                        padding=5,
                    )
                    index += 1

            # save model parameters
            torch.save(
                netG.state_dict(),
                "epochs/"
                + MODEL_NAME
                + "/netG_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch),
            )
            torch.save(
                netD.state_dict(),
                "epochs/"
                + MODEL_NAME
                + "/netD_epoch_%d_%d.pth" % (UPSCALE_FACTOR, epoch),
            )
            # save loss\scores\psnr\ssim
            # results["d_loss"].append(
            #     running_results["d_loss"] / running_results["batch_sizes"]
            # )
            # results["g_loss"].append(
            #     running_results["g_loss"] / running_results["batch_sizes"]
            # )
            # results["d_score"].append(
            #     running_results["d_score"] / running_results["batch_sizes"]
            # )
            # results["g_score"].append(
            #     running_results["g_score"] / running_results["batch_sizes"]
            # )
            # results["psnr"].append(valing_results["psnr"])
            # results["ssim"].append(valing_results["ssim"])
            # out_path = "statistics/"
            # data_frame = pd.DataFrame(
            #     data={
            #         "Loss_D": results["d_loss"].cpu(),
            #         "Loss_G": results["g_loss"].cpu(),
            #         "Score_D": results["d_score"].cpu(),
            #         "Score_G": results["g_score"].cpu(),
            #         "PSNR": results["psnr"].cpu(),
            #         "SSIM": results["ssim"],
            #     },
            #     index=range(1, epoch + 1),
            # )
            # data_frame.to_csv(
            #     out_path + "srf_" + str(UPSCALE_FACTOR) + "_train_results.csv",
            #     index_label="Epoch",
            # )
