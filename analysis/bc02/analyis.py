"""
Analysis of FluidFlower Benchmark Run BC02.
"""
from pathlib import Path

from benchmark.standardsetups.mediumco2analysis import MediumCO2Analysis

# Define the location for images of BC02 (all images in the folder)
images_folder = Path("/home/jakub/images/ift/medium/BC02")
images = list(sorted(images_folder.glob("*.JPG")))

# Define the location of all baseline images
baseline_folder = images_folder / Path("baseline")
baseline = list(sorted(baseline_folder.glob("*.JPG")))

# Define the location of the config file for BC02
config = Path("./config.json")

# Define FluidFlower based on a full set of basline images
analysis = MediumCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
)

# Perform standardized CO2 batch analysis on all images from BC02.
analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=True,  # print contour lines for CO2 onto image
)
