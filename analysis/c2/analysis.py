"""
Analysis of FluidFlower Benchmark Run C2.
"""
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

from benchmark.utils.fluidflower import BenchmarkCO2Analysis

# NOTE: Path needs update for each user - general process planned
folder = Path("/home/jakub/images/ift/benchmark/c2")
baseline = Path("baseline")

# Define FluidFlower based on a full set of basline images
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))
ff = BenchmarkCO2Analysis(baseline_images, config="./config.json", update_setup=False)

# Extract concentration.
images = list(sorted(folder.glob("*.JPG")))
for num, img in enumerate(images):

    # Information to the user
    print(f"working on {num}: {img.name}")

    tic = time.time()

    # Load the current image
    ff.load_and_process_image(img)

    # Determine binary mask detecting any(!) CO2
    co2 = ff.determine_co2_mask()

    # Determine binary mask detecting mobile CO2.
    co2_gas = ff.determine_co2_gas_mask(co2)

    # Create image with contours on top
    contours_co2, _ = cv2.findContours(
        skimage.img_as_ubyte(co2.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_co2_gas, _ = cv2.findContours(
        skimage.img_as_ubyte(co2_gas.img),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    original_img = np.copy(ff.img.img)
    original_img = skimage.img_as_ubyte(original_img)
    cv2.drawContours(original_img, contours_co2, -1, (0, 255, 0), 3)
    cv2.drawContours(original_img, contours_co2_gas, -1, (255, 255, 0), 3)

    # Plot corrected image with contours
    if True:
        plt.figure()
        plt.imshow(original_img)
        plt.show()

    # Write corrected image with contours to file
    img_id = Path(img.name).with_suffix("")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"segmentation/{img_id}_with_contours.jpg", original_img)

    # Store fine scale segmentation
    segmentation = np.zeros(ff.img.img.shape[:2], dtype=int)
    segmentation[co2.img] += 1
    segmentation[co2_gas.img] += 1
    np.save(f"segmentation/{img_id}_segmentation.npy", segmentation)

    # Store coarse scale segmentation
    coarse_shape = (150, 280)
    coarse_segmentation = np.zeros(coarse_shape, dtype=int)
    co2_coarse = skimage.img_as_bool(skimage.transform.resize(co2.img, coarse_shape))
    co2_gas_coarse = skimage.img_as_bool(
        skimage.transform.resize(co2_gas.img, coarse_shape)
    )
    coarse_segmentation[co2_coarse] += 1
    coarse_segmentation[co2_gas_coarse] += 1
    np.save(f"segmentation/{img_id}_coarse_segmentation.npy", coarse_segmentation)

    print(f"Elapsed time for {img.name}: {time.time()- tic}.")
