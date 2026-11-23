import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from models import GENERATORS
from utils import get_device, load_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Single Video")
    parser.add_argument(
        "--upscale_factor",
        default=4,
        type=int,
        choices=[2, 4, 8],
        help="super resolution upscale factor",
    )
    parser.add_argument(
        "--video_name", type=str, required=True, help="low resolution video to upscale"
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
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name
    CKPT = opt.ckpt

    device = get_device()
    model = load_generator(MODEL_NAME, UPSCALE_FACTOR, CKPT, device)

    videoCapture = cv2.VideoCapture(VIDEO_NAME)
    if not videoCapture.isOpened():
        raise IOError("Could not open the video '%s'" % VIDEO_NAME)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    sr_video_size = (
        int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR,
    )
    compared_video_size = (
        int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR
        + 10
        + int(
            int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10)
            / int(
                10
                * int(
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR)
                    // 5
                    + 1
                )
            )
            * int(
                int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR) // 5
                - 9
            )
        ),
    )
    video_stem = os.path.splitext(os.path.basename(VIDEO_NAME))[0]
    output_sr_name = "out_srf_" + str(UPSCALE_FACTOR) + "_" + video_stem + ".avi"
    output_compared_name = (
        "compare_srf_" + str(UPSCALE_FACTOR) + "_" + video_stem + ".avi"
    )
    sr_video_writer = cv2.VideoWriter(
        output_sr_name, cv2.VideoWriter_fourcc("M", "P", "E", "G"), fps, sr_video_size
    )
    compared_video_writer = cv2.VideoWriter(
        output_compared_name,
        cv2.VideoWriter_fourcc("M", "P", "E", "G"),
        fps,
        compared_video_size,
    )
    if not (sr_video_writer.isOpened() and compared_video_writer.isOpened()):
        raise IOError(
            "Could not open the output videos '%s' and '%s'"
            % (output_sr_name, output_compared_name)
        )
    # read frame
    success, frame = videoCapture.read()
    test_bar = tqdm(
        range(int(frame_numbers)), desc="[processing video and saving result videos]"
    )
    for index in test_bar:
        if success:
            with torch.no_grad():
                # OpenCV decodes frames as BGR; the models were trained on RGB images.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = ToTensor()(frame).unsqueeze(0).to(device)

                out = model(image)
                out = out.cpu()
                out_img = out.data[0].numpy()
                out_img = np.clip(np.round(out_img * 255.0), 0, 255).astype(np.uint8)
                out_img = np.ascontiguousarray(out_img.transpose((1, 2, 0)))
                # save sr video (OpenCV writes BGR)
                sr_video_writer.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

                # make compared video and crop shot of left top\right top\center\left bottom\right bottom
                out_img = ToPILImage()(out_img)
                crop_out_imgs = transforms.FiveCrop(size=out_img.width // 5 - 9)(
                    out_img
                )
                crop_out_imgs = [
                    np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img))
                    for img in crop_out_imgs
                ]
                out_img = transforms.Pad(padding=(5, 0, 0, 5))(out_img)
                compared_img = transforms.Resize(
                    size=(sr_video_size[1], sr_video_size[0]),
                    interpolation=Image.BICUBIC,
                )(ToPILImage()(frame))
                crop_compared_imgs = transforms.FiveCrop(
                    size=compared_img.width // 5 - 9
                )(compared_img)
                crop_compared_imgs = [
                    np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img))
                    for img in crop_compared_imgs
                ]
                compared_img = transforms.Pad(padding=(0, 0, 5, 5))(compared_img)
                # concatenate all the pictures to one single picture
                top_image = np.concatenate(
                    (np.asarray(compared_img), np.asarray(out_img)), axis=1
                )
                bottom_image = np.concatenate(
                    crop_compared_imgs + crop_out_imgs, axis=1
                )
                bottom_image = np.asarray(
                    transforms.Resize(
                        size=(
                            int(
                                top_image.shape[1]
                                / bottom_image.shape[1]
                                * bottom_image.shape[0]
                            ),
                            top_image.shape[1],
                        )
                    )(ToPILImage()(bottom_image))
                )
                final_image = np.concatenate((top_image, bottom_image))
                # save compared video (OpenCV writes BGR)
                compared_video_writer.write(cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                # next frame
                success, frame = videoCapture.read()
