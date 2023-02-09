"""
Standard segmentation script, for TableTopFluidFlower rig.
"""
from benchmark.standardsetups.tabletopco2analysis import TableTopFluidFlowerCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data

# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
analysis = TableTopFluidFlowerCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
)

# Perform standardized CO2 batch analysis.
analysis.batch_segmentation(images)
