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


print("WARNING: It assumed that the corrected images are labeled by the user.")
print(
    "For the analysis in the paper, inkscape and a manual selection has been used."
)

# Read and detect the black regions.
c1 = cv2.imread("labels/c1_labels.png")
#c3 = cv2.imread("labels/c3.png")
#c1_with_fault = cv2.imread("labels/c1_with_fault.png")
#c3_with_fault = cv2.imread("labels/c3_with_fault.png")

gray_c1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
#gray_c3 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
#gray_c1_with_fault = cv2.cvtColor(c1_with_fault, cv2.COLOR_BGR2GRAY)
#gray_c3_with_fault = cv2.cvtColor(c3_with_fault, cv2.COLOR_BGR2GRAY)

c1_water = gray_c1 == 0
#c3_water = gray_c3 == 0
#c1_water_and_fault = gray_c1_with_fault == 0
#c3_water_and_fault = gray_c3_with_fault == 0

# Store to file
np.save("labels/c1_water.npy", c1_water)
#np.save("labels/c3_water.npy", c3_water)
#np.save("labels/c1_water_and_fault.npy", c1_water_and_fault)
#np.save("labels/c3_water_and_fault.npy", c3_water_and_fault)

# Control
plt.figure("c1 water")
plt.imshow(c1_water)
plt.show()
