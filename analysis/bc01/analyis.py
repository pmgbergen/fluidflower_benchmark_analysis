"""
Analysis of FluidFlower Benchmark Run BC01.
"""
from benchmark.standardsetups.mediumco2analysis import MediumCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
analysis = MediumCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=True,  # print intermediate results to screen
)

# Consider only every second image
images = images[::2]

# Perform standardized CO2 batch analysis on all images from BC02.
analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=False,  # print contour lines for CO2 onto image
    write_contours_to_file = True, # print to file the same plot as prompted for plot_contours
    write_segmentation_to_file = True,
)
