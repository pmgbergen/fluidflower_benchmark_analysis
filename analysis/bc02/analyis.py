"""
Analysis of FluidFlower Benchmark Run BC02.
"""
from benchmark.standardsetups.mediumco2analysis import MediumCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

images = [
    images[10],
    images[13],
    images[16],
    images[19],
    images[21],
    images[24],
    images[27],
    images[30],
    images[33],
    images[36],
    images[50],
    images[60],
    images[70],
    images[80],
    images[90],
    images[100],
    images[125],
    images[150],
    images[175],
    images[200],
    images[225],
    images[250],
    images[280],
]

# Define FluidFlower based on a full set of basline images
analysis = MediumCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=True,  # print intermediate results to screen
)

# Perform standardized CO2 batch analysis on all images from BC02.
analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=False,  # print contour lines for CO2 onto image
    write_contours_to_file = True, # print to file the same plot as prompted for plot_contours,
)
