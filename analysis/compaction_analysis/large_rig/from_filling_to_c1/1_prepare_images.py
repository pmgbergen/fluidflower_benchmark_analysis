"""
1. Step of compaction analysis for the FluidFlower.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
from pathlib import Path
import cv2

from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
import darsia

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest.
path_dst = Path("original/DSC06341.JPG")
path_src = Path("original/20210916-124123.JPG")

# Base analysis object to organize the reading of images
analysis = darsia.AnalysisBase(path_src, "./config_preparation.json")

# Read image
analysis.load_and_process_image(path_dst)

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_src = analysis.base
img_dst = analysis.img

# Store the corrected images (as numpy arrays).
Path("./corrected").mkdir(exist_ok=True)
np.save("corrected/src.npy", img_src.img)
np.save("corrected/dst.npy", img_dst.img)

# Store corrected images as (JPG)
cv2.imwrite("corrected/src.jpg", cv2.cvtColor(skimage.img_as_ubyte(img_src.img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imwrite("corrected/dst.jpg", cv2.cvtColor(skimage.img_as_ubyte(img_dst.img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if True:
    plt.figure()
    plt.imshow(img_src.img)
    plt.figure()
    plt.imshow(img_dst.img)
    plt.show()



# Same for alternative setup
path_zero = Path("original/DSC08428.JPG")

# Base analysis object to organize the reading of images
analysis = darsia.AnalysisBase(path_zero, "./config_zero.json")

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_zero = analysis.base

# Store the corrected images (as numpy arrays).
Path("./corrected").mkdir(exist_ok=True)
np.save("corrected/zero.npy", img_zero.img)

# Store corrected images as (JPG)
cv2.imwrite("corrected/zero.jpg", cv2.cvtColor(skimage.img_as_ubyte(img_zero.img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
