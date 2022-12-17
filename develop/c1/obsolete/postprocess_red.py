from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from fluidflower import BenchmarkRig
from scipy import ndimage as ndi

img = np.load("signal_red.npy")

plt.imshow(img)
plt.show()

img = cv2.resize(img, (280, 150), interpolation=cv2.INTER_AREA)

plt.imshow(img)
plt.show()

img = skimage.restoration.denoise_tv_chambolle(img, 0.001)

plt.imshow(img)
plt.show()

hist = np.histogram(np.ravel(img), bins=100)[0]
thresh = skimage.filters.threshold_otsu(img)
print(thresh, thresh / np.max(img) * 100)
plt.plot(hist)
plt.show()

mask = img > 0.9 * thresh
plt.imshow(mask)
plt.show()

baseline_images = "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
ff.load_and_process_image(
    "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
)
original = ff.img.img

plt.imshow(original)
shape = original.shape[:2]
print(shape)
mask_fine = skimage.img_as_ubyte(
    cv2.resize(mask.astype(float), tuple(reversed(shape))).astype(bool)
)
plt.imshow(mask_fine, alpha=0.05)
plt.show()

contours, hierarchy = cv2.findContours(
    mask_fine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
original_copy = np.copy(original)
cv2.drawContours(original_copy, contours, -1, (0, 255, 0), 1)
plt.imshow(original_copy)
plt.show()
