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
path_src = Path("original/20211001-145251.JPG")
path_dst = Path("original/211124_time082740_DSC00067.JPG")

# Base analysis object to organize the reading of images
analysis = darsia.AnalysisBase(path_dst, "./config.json") # NOTE: Config.json is tailored to dst.
analysis.load_and_process_image(path_src)

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_src = analysis.img
img_dst = analysis.base

# Store the corrected images (as numpy arrays).
Path("./corrected").mkdir(exist_ok=True)
np.save("corrected/src.npy", img_src.img)
np.save("corrected/dst.npy", img_dst.img)

# Store corrected images as (JPG)
cv2.imwrite(
    "corrected/src.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(img_src.img), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)
cv2.imwrite(
    "corrected/dst.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(img_dst.img), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)


if True:
    plt.figure("src")
    plt.imshow(img_src.img)
    plt.figure("dst")
    plt.imshow(img_dst.img)
    plt.show()
