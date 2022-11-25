"""
Analysis of FluidFlower Benchmark Run C3 - use the corrected images.
"""
from pathlib import Path
import cv2
import skimage

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=True,  # print intermediate results to screen
)

images = [
    images[20],
    images[40],
    images[60],
    images[80],
    Path("/media/jakub/Elements/Jakub/benchmark/data/c3/211214_time132000_DSC01907.TIF"), # CO2(g) too large
    Path("/media/jakub/Elements/Jakub/benchmark/data/c3/211215_time082000_DSC04514.TIF"), # upper plume too large?
    Path("/media/jakub/Elements/Jakub/benchmark/data/c3/211215_time162000_DSC04610.TIF"), # yellow blob?
    Path("/media/jakub/Elements/Jakub/benchmark/data/c3/211215_time212000_DSC04670.TIF"), # green holes?
    Path("/media/jakub/Elements/Jakub/benchmark/data/c3/211219_time112000_DSC05702.TIF"), # yellow blob?
]

# Perform standardized CO2 batch analysis on all images.
co2_analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=False,  # print contour lines for CO2 onto image
    write_contours_to_file=True,
#    write_segmentation_to_file = True,
#    write_coarse_segmentation_to_file = True,
    # ...for more options, check the keyword arguments of BenchmarkCO2Analysis.batch_analysis.
)

# Write final results to file
co2_analysis.write_results_to_file()
