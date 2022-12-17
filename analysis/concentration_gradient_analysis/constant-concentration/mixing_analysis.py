"""
Script for the analysis of quantity 5 in the benchmark:

    Integral over box C of the L1 norm of the gradient
    of the water concentration scaled by the inverse of
    the maximum concentration value.

The approach used here assumes constant water concentration
in the CO2 water zone. Using the segmentation of the domain
into water, CO2(w) and CO2(g), identifies
"""

from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.utils.misc import read_time_from_path

for run in ["c1", "c2", "c3", "c4", "c5"]:

    # Fetch images and segmentations.
    images = list(
        sorted(
            Path(f"/media/jakub/Elements/Jakub/benchmark/data/large_rig/{run}").glob(
                "*"
            )
        )
    )[
        10:
    ]  # exclude baseline
    segmentations = list(
        sorted(
            Path(
                f"/media/jakub/Elements/Jakub/benchmark/results/large_rig/fixed-thresholds/{run}/npy_segmentation"
            ).glob("*.npy")
        )
    )
    concentrations = list(
        sorted(
            Path(
                f"/media/jakub/Elements/Jakub/benchmark/results/large_rig/fixed-thresholds/{run}/concentration_npy"
            ).glob("*.npy")
        )
    )
    num_concentrations = len(concentrations)

    # Setup up fluidflower
    base = images[0]
    config = Path("./config.json")
    fluidflower = LargeFluidFlower(base, config)

    # Fetch reference time
    ref_time = read_time_from_path(images[0])

    # Contour analysis.
    contour_analysis = darsia.ContourAnalysis(verbosity=False)
    co2_g_analysis = darsia.ContourAnalysis(verbosity=False)

    # Keep track of the quantity of interest for each segmentation, and the respective time
    concentration_gradient_integral = []
    rel_time = []

    # Analyze each concentration/time step separately.
    for i in range(num_concentrations):

        # Fix the photograph and concentration for each time step
        concentration = darsia.Image(np.load(concentrations[i]), width=2.8, height=1.5)
        image = darsia.Image(
            cv2.cvtColor(cv2.imread(str(images[i])), cv2.COLOR_BGR2RGB),
            width=2.8,
            height=1.5,
        )

        # Make relative concentration
        dissolution_limit = 1.8  # kg / m**3
        concentration.img /= dissolution_limit

        # Add timestamp from title
        time = read_time_from_path(images[i])
        # concentration.timestamp = time
        # image.timestamp = time
        relative_time_minutes = (time - ref_time).total_seconds() / 60
        rel_time.append(relative_time_minutes)

        # Plot the concentration and box C
        if False:
            concentration_img = np.zeros((*concentration.img.shape[:2], 3), dtype=float)
            for i in range(3):
                concentration_img[:, :, i] = concentration.img
            concentration_img = skimage.img_as_ubyte(concentration_img)

            # Add box C
            contours_box_C, _ = cv2.findContours(
                skimage.img_as_ubyte(fluidflower.mask_box_C),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(concentration_img, contours_box_C, -1, (180, 180, 180), 3)

            plt.imshow(concentration_img)
            plt.show()

        # Build a gradient - the first components is the negative gradient in y-direction, the second one it the gradient in x-direction.
        concentration_gradient = np.gradient(concentration.img)

        # Need to scale with physical dimensions
        concentration_gradient[0] /= -concentration.dy
        concentration_gradient[1] /= concentration.dx

        # Build the L1 norm of the gradient.
        l1_norm_concentration_gradient = np.sqrt(
            np.power(concentration_gradient[0], 2)
            + np.power(concentration_gradient[1], 2)
        )

        # Restrict to box C
        l1_integral_box_C = l1_norm_concentration_gradient[fluidflower.mask_box_C]

        # Compute the integral
        qty_5 = concentration.dx * concentration.dy * np.sum(l1_integral_box_C)

        #    # Apply contour analysis for box C - only consider only CO2(w) - not CO2(g)
        #    contour_analysis.load_labels(
        #        img = segmentation,
        #        roi = fluidflower.box_C,
        #        values_of_interest = [1],
        #    )
        #
        #    # Length of interface between CO2 and water, without the box.
        #    qty_5 = contour_analysis.length()
        concentration_gradient_integral.append(qty_5)
        print(f"{run}, timestep {i}, Qty 5: {qty_5}")

    # Put concentration_gradient_integral in perspective to box C.
    len_box_C = 1.5
    rel_concentration_gradient_integral = [
        qty / 1.5 for qty in concentration_gradient_integral
    ]

    Path("results").mkdir(parents=True, exist_ok=True)
    np.save(f"results/{run}_qty_5.npy", rel_concentration_gradient_integral)
    np.save(f"results/{run}_time.npy", rel_time)

    plt.figure("Qty 5")
    plt.plot(rel_time, rel_concentration_gradient_integral, label=f"{run}")
plt.legend()
plt.show()
