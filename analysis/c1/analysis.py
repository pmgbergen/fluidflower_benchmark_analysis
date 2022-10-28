"""
Analysis of FluidFlower Benchmark Run C1.
"""
import json
from pathlib import Path

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis

# Read user-defined paths to images, number of baseline images, and config file
with open(Path("user_data.json"), "r") as openfile:
    user_data = json.load(openfile)

# Short cuts.
# Path to images
images_folder = Path(user_data["images folder"])
# Type of images ("*.JPG", or "*.TIF")
file_ending = user_data["file ending"]
# Number of images characterized as baseline images
num_baseline_images = user_data["number baseline images"]
# Path to analysis specific config file
config = Path(user_data["config"])

# Define the location for images of C1 (all images in the folder except the baseline images)
images = list(sorted(images_folder.glob(file_ending)))[num_baseline_images:]

# Define the location of all baseline images
baseline = list(sorted(images_folder.glob(file_ending)))[:num_baseline_images]

# Define FluidFlower based on a full set of basline images
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=True,  # print intermediate results to screen
)

# Perform standardized CO2 batch analysis on all images from C1.
co2_analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=True,  # print contour lines for CO2 onto image
    fingering_analysis_box_C=False,  # determine and print the length of the fingers in box C
    write_contours_to_file=True,
    # ...for more options, check the keyword arguments of BenchmarkCO2Analysis.batch_analysis.
)
