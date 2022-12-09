"""
Determine compaction of FluidFlower by comparing two different images.

The images correpsond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, compaction/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
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

if False:

    # Paths to two images of interest.
    path_src = Path("images/src_corrected.npy")
    path_dst = Path("images/dst_corrected.npy")
    
    img_src = np.load(path_src)
    img_dst = np.load(path_dst)
    
    labels = darsia.segment(img_src, "supervised", "scharr", monochromatic_color = "red", marker_points = [[1030, 3720]])
    
    plt.figure()
    plt.imshow(img_src[:,:,0])
    plt.figure()
    plt.imshow(img_src[:,:,1])
    plt.figure()
    plt.imshow(img_src[:,:,2])
    plt.figure()
    plt.imshow(cv2.cvtColor(img_src, cv2.COLOR_RGB2HSV)[:,:,2])
    plt.show()

else:

    print("WARNING: It assumed that the corrected images are blackened by the user.")
    print("For the analysis in the paper, inkscape and a manual selection has been used.")

    assert Path("blackened/src.png").exists()
    assert Path("blackened/dst.png").exists()
    assert Path("blackened/zero.png").exists()
