import cv2
import numpy as np


IMAGE_PATH = "outputs/frames/frame_0000.jpg"


# Read the image
image = cv2.imread(IMAGE_PATH)


# Check whether image was loaded
if image is None:
    print("Error: Could not load image.")
    exit()


print("Image loaded successfully!")
print()

# Image dimensions
print("Image shape:", image.shape)

# Image data type
print("Image data type:", image.dtype)

# Minimum pixel value
print("Minimum pixel value:", image.min())

# Maximum pixel value
print("Maximum pixel value:", image.max())

# Total number of values
print("Total values:", image.size)

# First pixel
print("First pixel:", image[0, 0])