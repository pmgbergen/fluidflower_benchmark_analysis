"""
1. Step of compaction analysis for the FluidFlower.
"""

from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest.
path_dst = Path("original/DSC06341.JPG")
# path_src = Path("original/20210916-124123.JPG")

# Base analysis object to organize the reading of images
analysis = darsia.AnalysisBase(path_dst, "./config_preparation_dst.json")

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_dst = analysis.base

# Store the corrected images (as numpy arrays).
Path("./corrected").mkdir(exist_ok=True)
np.save("corrected/dst.npy", img_dst.img)

# Store corrected images as (JPG)
cv2.imwrite(
    "corrected/dst.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(img_dst.img), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

# Same for alternative setup
path_zero = Path("original/DSC08428.JPG")

# Base analysis object to organize the reading of images
analysis = darsia.AnalysisBase(path_zero, "./config_preparation_zero.json")

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_zero = analysis.base

# Store the corrected images (as numpy arrays).
Path("./corrected").mkdir(exist_ok=True)
np.save("corrected/zero.npy", img_zero.img)

# Store corrected images as (JPG)
cv2.imwrite(
    "corrected/zero.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(img_zero.img), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

if True:
    plt.figure("dst")
    plt.imshow(img_dst.img)
    plt.figure("zero")
    plt.imshow(img_zero.img)
    plt.show()
