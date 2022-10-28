"""
Analysis of FluidFlower Benchmark Run C2.
"""
from pathlib import Path
from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis

# Define the location for images of C2 (all images in the folder)
images_folder = Path("/home/jakub/images/ift/benchmark/c2")
images = list(sorted(images_folder.glob("*.JPG")))

# Define the location of all baseline images
baseline_folder = images_folder / Path("baseline")
baseline = list(sorted(baseline_folder.glob("*.JPG")))

# Define the location of the config file for C2
config = Path("./config.json")

# Define FluidFlower based on a full set of basline images
analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity = False,   # print intermediate results to screen
)

# Perform standardized CO2 batch analysis on all images from C2.
analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=False,  # print contour lines for CO2 onto image
    write_contours_to_file = True,
    fingering_analysis_box_C = True,
    # ...for more options, check the keyword arguments of BenchmarkCO2Analysis.batch_analysis.
)
