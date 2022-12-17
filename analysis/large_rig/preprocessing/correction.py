# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:52:46 2022

@author: bbe020
"""

import os

import darsia as da
from pathlib import Path
import json

# Define paths to data
images = Path("/media/jakub/Elements/Jakub/benchmark/data/large_rig/tmp")
baseline = Path("/media/jakub/Elements/Jakub/benchmark/data/large_rig/tmp/211124_time082740_DSC00067.TIF")

# Define paths for storing the corrected data
output = Path("/media/jakub/Elements/Jakub/benchmark/data/large_rig/out")

# Define path to config file for correction
config_path = Path("../config_correction.json")
f = open(config_path, "r")
config = json.load(f)
f.close()

# Define correction objects
curvature_correction = da.CurvatureCorrection(config=config["curvature"])
drift_correction = da.DriftCorrection(base=baseline, config=config["drift"])
color_correction = da.ColorCorrection(config=config["color"])

# Correct each original image and store the corrected image to file
for img in images.glob("*"):

    # Correct
    corrected_image = da.Image(
        img,
        color_correction=color_correction,
        drift_correction=drift_correction,
        curvature_correction=curvature_correction,
    )

    # Write to file
    corrected_image.write(output / img.name)
