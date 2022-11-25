"""
Analysis of FluidFlower Benchmark Run C3.
"""
from pathlib import Path
import cv2
import skimage
import json

from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
co2_analysis = BenchmarkCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
)

# Perform standardized CO2 segmentation on all images
co2_analysis.batch_segmentation(images)

## Print threshold cache
out_file = open("co2_threshold.json", "w")
json.dump(co2_analysis.co2_analysis.threshold_cache, out_file, indent=4)
out_file.close()
gas_out_file = open("co2_gas_threshold.json", "w")
json.dump(co2_analysis.co2_gas_analysis.threshold_cache, gas_out_file, indent=4)
gas_out_file.close()
