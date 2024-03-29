"""
Analysis of FluidFlower Benchmark Run C*.
"""
from pathlib import Path
import cv2
import skimage
import json

from benchmark.standardsetups.largerigco2analysis import LargeRigCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
co2_analysis = LargeRigCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
)

#images = images[::5] + images[-6:]

# Perform standardized CO2 segmentation on all images
co2_analysis.batch_segmentation(images)
