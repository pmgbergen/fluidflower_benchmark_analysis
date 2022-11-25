"""
Preparations for the analysis of FluidFlower benchmark.
Provide segmentation of the geometry.
"""
from pathlib import Path
import cv2
import skimage

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data_fine.json")

# Define FluidFlower analysis
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=True,  # flag controlling whether aux. data needs update
)

# Extract segentation of geometry
fine_geometry_segmentation = co2_analysis.labels
