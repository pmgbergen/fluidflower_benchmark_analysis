"""
Analysis of FluidFlower Benchmark Run C1.
"""
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

from fluidflower import BenchmarkRig

folder = Path("/home/jakub/images/ift/benchmark/well_test")
pt1 = Path("211002 well tests")
pt2 = Path("211002 well tests_overnight")
pt3 = Path("211003 well tests continued_1")
baseline = Path("baseline")
processed = Path("processed")


# Define FluidFlower with first 10 baseline images
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))[:20]
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=True)

## Completely flushed rig for calibration
# img_flushed = folder / Path("220202_well_test") / Path("DSC04224.JPG")
# ff.load_and_process_image(img_flushed)
# plt.figure()
# plt.imshow(ff.img.img)
# plt.show()
# full_tracer = ff.determine_concentration()
# plt.figure()
# plt.imshow(full_tracer.img)
# plt.show()


# Extract concentration.
images = [
    #    list(sorted((folder / pt3).glob("*.JPG")))[10],
    #    list(sorted((folder / pt3).glob("*.JPG")))[20],
    #    list(sorted((folder / pt3).glob("*.JPG")))[30],
    list(sorted((folder / pt3).glob("*.JPG")))[0],
]

print(images)
for img in images:

    tic = time.time()

    ff.load_and_process_image(img)

    concentration = ff.determine_concentration()

    print(f"Elapsed time: {time.time() - tic}.")

    plt.figure()
    plt.imshow(concentration.img)
    plt.show()
