"""
Determine compaction of FluidFlower by comparing two different images.

The images correspond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, compaction/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
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

print("WARNING: It assumed that the corrected images are blackened by the user.")
print("For the analysis in the paper, inkscape and a manual selection has been used.")

# Read and detect the black regions.
dst = cv2.imread("blackened/dst.png")
src = cv2.imread("blackened/src.png")

dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

dst_mask = dst_gray == 0
src_mask = src_gray == 0

# Store to file
np.save("blackened/dst_mask.npy", dst_mask)
np.save("blackened/src_mask.npy", src_mask)
