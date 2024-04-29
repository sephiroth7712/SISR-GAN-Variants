import cv2

# Read the input image
image = cv2.imread("./data/test/SRF_4/data/Set5_baby.png")

# Define the upscaling factor
scale_factor = 4

# Upscale the image using bicubic interpolation
upscaled_image = cv2.resize(
    image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
)

# Save the upscaled image
cv2.imwrite("upscaled_image.png", upscaled_image)
