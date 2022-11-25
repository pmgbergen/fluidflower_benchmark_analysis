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

# Early Image
img = np.load("signal_green_early.npy")
baseline_images = "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
ff.load_and_process_image(
    "/home/jakub/images/ift/benchmark/c1/211124_time104502_DSC00515.JPG"
)
original = ff.img.img
np.save("original_early.npy", original)

# Mid Image
img = np.load("signal_green_mid.npy")
baseline_images = "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
ff.load_and_process_image(
    "/home/jakub/images/ift/benchmark/c1/211124_time131522_DSC00966.JPG"
)
original = ff.img.img
np.save("original_mid.npy", original)

# Late Image
img = np.load("signal_green_late.npy")
baseline_images = "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)
ff.load_and_process_image(
    "/home/jakub/images/ift/benchmark/c1/211128_time015922_DSC03851.JPG"
)
original = ff.img.img
np.save("original_late.npy", original)
