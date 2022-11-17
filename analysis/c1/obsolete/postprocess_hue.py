"""
Testing script for identifying the CO2 phase (water and pure) as
binary field. Based on hue (first component of HSV).
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from fluidflower import BenchmarkRig
from scipy import ndimage as ndi

# Choose image

## Early Image
# img = np.load("signal_hue_early.npy")
# baseline_images = '/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG'
# ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
# ff.load_and_process_image('/home/jakub/images/ift/benchmark/c1/211124_time104502_DSC00515.JPG')
# original = ff.img.img

## Mid Image
# img = np.load("signal_hue_mid.npy")
# baseline_images = '/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG'
# ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
# ff.load_and_process_image('/home/jakub/images/ift/benchmark/c1/211124_time131522_DSC00966.JPG')
# original = ff.img.img

# Late Image
img = np.load("signal_hue_late.npy")
baseline_images = "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
ff.load_and_process_image(
    "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
)
original = ff.img.img

show_all = False

if show_all:
    plt.imshow(img)
    plt.show()

print("start")

factor = 8
img = cv2.resize(img, (factor * 280, factor * 150), interpolation=cv2.INTER_AREA)

if show_all:
    plt.imshow(img)
    plt.show()

img = skimage.restoration.denoise_tv_chambolle(img, 0.0001, eps=1e-5, max_num_iter=200)

if show_all:
    plt.imshow(img)
    plt.show()

hist = np.histogram(np.ravel(img), bins=100)[0]
thresh = skimage.filters.threshold_otsu(img)
print(thresh, thresh / np.max(img) * 100)
if show_all:
    plt.plot(hist)
    plt.show()

thresh = 0.6 * thresh  # implemented currently
thresh = 0.01  # mimick calibration

mask = img > thresh
if show_all:
    plt.imshow(mask)
    plt.show()

# Remove small objects
mask = ndi.morphology.binary_opening(mask, structure=np.ones((2, 2)))
if show_all:
    plt.imshow(mask)
    plt.show()


plt.imshow(original)
shape = original.shape[:2]
mask_fine = skimage.img_as_ubyte(
    cv2.resize(mask.astype(float), tuple(reversed(shape))).astype(bool)
)

print(shape)
np.save("mask.npy", mask)
np.save("mask_fine.npy", mask_fine)

# Remove small objects - CO2 comes in larger packs
mask_fine = skimage.morphology.remove_small_holes(mask_fine, area_threshold=10**2)
# mask_fine = ndi.morphology.binary_opening(mask_fine, structure=np.ones((10,10)))
# if show_all:
#    plt.imshow(mask)
#    plt.show()
#
# plt.imshow(mask_fine, alpha = 0.05)
# plt.show()

mask_fine = skimage.util.img_as_ubyte(mask_fine)

contours, hierarchy = cv2.findContours(
    mask_fine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
original_copy = np.copy(original)
cv2.drawContours(original_copy, contours, -1, (0, 255, 0), 1)
plt.imshow(original_copy)
plt.show()
