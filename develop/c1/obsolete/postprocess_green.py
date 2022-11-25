"""
Testing script for identifying the CO2 phase (water and pure) as
binary field. Based on green (first component of HSV).
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from fluidflower import BenchmarkRig
from scipy import ndimage as ndi

# Choose image

for i in range(0, 3):

    # Early Image
    if i == 0:
        img = np.load("signal_green_early.npy")
        original = np.load("original_early.npy")

    # Mid Image
    if i == 1:
        img = np.load("signal_green_mid.npy")
        original = np.load("original_mid.npy")

    # Late Image
    if i == 2:
        img = np.load("signal_green_late.npy")
        original = np.load("original_late.npy")

    show_all = True

    # if show_all:
    #    plt.imshow(img)
    #    plt.show()

    factor = 2
    img = cv2.resize(img, (factor * 280, factor * 150), interpolation=cv2.INTER_AREA)

    # if show_all:
    #    plt.imshow(img)
    #    plt.show()

    img = skimage.restoration.denoise_tv_chambolle(
        img, 0.0001, eps=1e-5, max_num_iter=1000
    )

    if show_all:
        plt.imshow(img)
        plt.show()

    hist = np.histogram(np.ravel(img[img > 0.01 * np.max(img)]), bins=100)[0]
    thresh = skimage.filters.threshold_otsu(img)
    print(thresh, thresh / np.max(img) * 100)
    if show_all:
        plt.plot(hist)
        plt.show()

    thresh = 27

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
    plt.imshow(mask_fine, alpha=0.05)
    plt.show()

    contours, hierarchy = cv2.findContours(
        mask_fine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    original_copy = np.copy(original)
    cv2.drawContours(original_copy, contours, -1, (0, 255, 0), 1)
    plt.imshow(original_copy)
    plt.show()
