"""Upscale an image with bicubic interpolation, as a visual baseline."""
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description="Upscale an image with bicubic interpolation")
    parser.add_argument(
        "--image",
        default="./data/test/SRF_4/data/Set5_baby.png",
        help="image to upscale",
    )
    parser.add_argument("--scale", default=4, type=int, help="upscale factor")
    parser.add_argument(
        "--output", default="upscaled_image.png", help="where to save the upscaled image"
    )
    opt = parser.parse_args()

    # Read the input image
    image = cv2.imread(opt.image)
    if image is None:
        raise FileNotFoundError("Could not read the image '%s'" % opt.image)

    # Upscale the image using bicubic interpolation
    upscaled_image = cv2.resize(
        image, None, fx=opt.scale, fy=opt.scale, interpolation=cv2.INTER_CUBIC
    )

    # Save the upscaled image
    cv2.imwrite(opt.output, upscaled_image)


if __name__ == "__main__":
    main()
