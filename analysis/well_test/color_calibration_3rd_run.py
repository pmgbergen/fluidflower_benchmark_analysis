"""
Simple calibration based on a few images taken from the first
optimal control run of the PoroTwin1 project.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import time

from fluidflower import BenchmarkRig

folder = Path("/home/jakub/images/ift/benchmark/well_test")
pt1 = Path("211002 well tests")
pt2 = Path("211002 well tests_overnight")
pt3 = Path("211003 well tests continued_1")
baseline = Path("baseline")
processed = Path("processed")


# Define FluidFlower with first 10 baseline images
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))[:20]
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)

# Define test images, used for the calibration
injection_images = list(sorted((folder / pt1).glob("*.JPG")))[5:20]

# Calibrate concentration analysis based on injection rate
injection_rate = 2250  # ml/hr
ff.calibrate_concentration_analysis(
    paths=paths, injection_rate=injection_rate, initial_guess=[1, 20], tol=1e-1
)
# TODO update this info!
# The calibration is equivalent with:
# ff.concentration_analysis.update(scaling = ???)

# Store calibration to file
ff.concentration_analysis.write_calibration_to_file(
    "concentration_calibration.json", "cleaning_filter_from_calibration.npy"
)

# Test run
for path in injection_images:
    ff.load_and_process_image(path)
    concentration = ff.determine_concentration()
    concentration.plt_show()
