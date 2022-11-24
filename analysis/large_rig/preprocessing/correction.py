# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:52:46 2022

@author: bbe020
"""

import os

import darsia as da
from pathlib import Path

# Define path to config file for correction
config_path = Path("../config_correction.json")

# Define paths to data
folders_path = Path("H:/sets/raw_tiff2/")
folders = os.listdir(folders_path)

base_img_path = Path("H:/sets/raw_tiff2/c1/211124_time082740_DSC00067.TIF")

# Define paths for storing the corrected data
save_path = Path("H:/sets/tiff_corrected/")

# Define correction objects
curv_correction = da.CurvatureCorrection(config=config_path)
drift_correction = da.DriftCorrection(base=base_img_path, config=config_path)
color_correction = da.ColorCorrection(config=config_path)

# Perform correction for all images
for folder in folders:

    # Add folder structure if required
    images_path = folders_path + folder + "/"
    images = os.listdir(images_path)
    try:
        os.mkdir(save_path + folder)
    except:
        None

    # Correct each original image and store the corrected image to file
    for img in images:
        new_corrected_image = da.Image(
            images_path + img,
            color_correction=color_correction,
            drift_correction=drift_correction,
            curvature_correction=curv_correction,
            width=2.8,
            height=1.5,
        )
        new_corrected_image.write(save_path + "/" + folder + "/" + img)
