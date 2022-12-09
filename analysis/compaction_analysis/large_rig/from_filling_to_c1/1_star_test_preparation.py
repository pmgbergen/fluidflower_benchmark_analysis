"""
Determine compaction of FluidFlower by comparing two different images.

Reference image is taken just after filling the rig with sand.
Test image is just before C1.

"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
from pathlib import Path

from benchmark.rigs.largefluidflower import LargeFluidFlower
import darsia
import cv2
import copy

# ! ----- Preliminaries - prepare two images for compaction analysis

# Image after sand filling
path_zero = Path("corrected/zero.jpg")
fluidflower_zero = LargeFluidFlower(path_zero, "./config_compaction_zero.json", False)
labels_zero = fluidflower_zero.labels.copy()
img_zero = fluidflower_zero.base

# Some later image
path_src = Path("corrected/src.jpg")
fluidflower_src = LargeFluidFlower(path_src, "./config_compaction_src.json", False)
labels_src = fluidflower_src.labels.copy()
img_src = fluidflower_src.base

# Image before running C1
path_dst = Path("corrected/dst.jpg")
fluidflower_dst = LargeFluidFlower(path_dst, "./config_compaction_dst.json", False)
labels_dst = fluidflower_dst.labels.copy()
img_dst = fluidflower_dst.base

# Fix a reference image
img_ref = img_zero.copy()


# ! ---- 0. Iteration
print("0. iteration")
plt.figure("Initial comparison")
plt.imshow(skimage.util.compare_images(img_dst.img, img_ref.img, method="blend"))
plt.show()
