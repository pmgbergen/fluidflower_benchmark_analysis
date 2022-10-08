"""
Analysis of FluidFlower Benchmark Run C1.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

from fluidflower import BenchmarkRig

folder = Path("/home/jakub/images/ift/benchmark/c1")
baseline = Path("baseline")
processed = Path("processed")

# Define FluidFlower with first 10 baseline images
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))[:20]
ff = BenchmarkRig(baseline_images, config_source="./config.json", update_setup=False)

# Extract concentration.
images = [
    list(sorted(folder.glob("*.JPG")))[10],
    list(sorted(folder.glob("*.JPG")))[30],
    list(sorted(folder.glob("*.JPG")))[50],
    list(sorted(folder.glob("*.JPG")))[70],
    list(sorted(folder.glob("*.JPG")))[90],
    list(sorted(folder.glob("*.JPG")))[110],
    list(sorted(folder.glob("*.JPG")))[125],
]
print(images)
for img in images:
    ff.load_and_process_image(img)

    # Determine binary mask detecting any(!) CO2
    co2 = ff.determine_co2_mask()

    # Determine binary mask detecting mobile CO2.
    mobile_co2 = ff.determine_mobile_co2_mask(co2)

    # Create image with contours on top
    contours_co2, _ = cv2.findContours(
        skimage.util.img_as_ubyte(co2.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_mobile_co2, _ = cv2.findContours(
        skimage.util.img_as_ubyte(mobile_co2.img),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    original_img = np.copy(ff.img.img)
    cv2.drawContours(original_img, contours_co2, -1, (0, 255, 0), 3)
    cv2.drawContours(original_img, contours_mobile_co2, -1, (0, 0, 255), 3)

    # Test plot
    plt.figure()
    plt.imshow(original_img)
    plt.show()
