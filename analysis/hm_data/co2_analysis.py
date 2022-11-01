"""
Analysis of FluidFlower Benchmark Run C1 tailored to the needs of the history matching study.
"""
from pathlib import Path

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis

# Define location of images and config file
images_folder ="/home/jakub/images/ift/benchmark/c1"
file_ending = "*.JPG"
config = "./config.json"

images = list(sorted(Path(images_folder).glob(file_ending)))
baseline = list(sorted(Path(Path(images_folder) / Path("baseline")).glob(file_ending)))[:10]

# Define FluidFlower based on a full set of basline images
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=False,  # print intermediate results to screen
)

# Perform standardized CO2 batch analysis on all images from C1.
co2_analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=True,  # print contour lines for CO2 onto image
    fingering_analysis_box_C=False,  # determine and print the length of the fingers in box C
    write_contours_to_file=False,
    # ...for more options, check the keyword arguments of BenchmarkCO2Analysis.batch_analysis.
)
