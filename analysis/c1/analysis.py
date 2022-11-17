"""
Analysis of FluidFlower Benchmark Run C1.
"""
from pathlib import Path
import cv2
import skimage

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

baseline_path = Path("/home/jakub/images/ift/benchmark/c1/baseline")
baseline = list(sorted(baseline_path.glob("*JPG")))[:20]

# Define FluidFlower based on a full set of basline images
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=False,  # print intermediate results to screen
)

da_img = co2_analysis.base
new_img = da_img.add_grid(dx = 0.1, dy= 0.1)
import matplotlib.pyplot as plt
plt.imshow(new_img.img)
plt.show()

# Perform standardized CO2 batch analysis on all images from C1.
co2_analysis.batch_analysis(
    images=images,  # paths to images to be considered
    plot_contours=False,  # print contour lines for CO2 onto image
    fingering_analysis_box_C=False,  # determine and print the length of the fingers in box C
    write_contours_to_file=True,
    # ...for more options, check the keyword arguments of BenchmarkCO2Analysis.batch_analysis.
)

# Write final results to file
co2_analysis.write_results_to_file()
