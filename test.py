import argparse
import datetime
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from models import get_generator

parser = argparse.ArgumentParser(description="Test Single Image")
parser.add_argument(
    "--upscale_factor", default=4, type=int, help="super resolution upscale factor"
)
parser.add_argument(
    "--test_mode",
    default="GPU",
    type=str,
    choices=["GPU", "CPU"],
    help="using GPU or CPU",
)
parser.add_argument("--image_name", type=str, help="test low resolution image name")
parser.add_argument(
    "--model_name",
    default="pre_netG_epoch_4_5000.pth",
    type=str,
    help="generator model epoch name",
)
parser.add_argument(
    "--ckpt",
    default="pre_netG_epoch_4_400.pth",
    type=str,
    help="generator model checkpoint file after pre-training",
)
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == "GPU" else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
CKPT = opt.ckpt

model = get_generator(MODEL_NAME)(1, UPSCALE_FACTOR)

if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load("epochs/" + MODEL_NAME + "/" + CKPT))
else:
    model.load_state_dict(
        torch.load("epochs/" + MODEL_NAME, map_location=lambda storage, loc: storage)
    )

image = Image.open(IMAGE_NAME)
with torch.no_grad():
    image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = datetime.datetime.now()
    out = model(image)
    elapsed = datetime.datetime.now() - start
    print("Time: " + str(elapsed) + "s")
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save("out_srf_" + str(UPSCALE_FACTOR) + "_" + os.path.basename(IMAGE_NAME))
