"""
Post analysis script analyzing the evolution of fingers in C1.
"""

import darsia
import cv2
import numpy as np
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.utils.misc import read_time_from_path

# Setup up flouidflower
base = Path("/home/jakub/images/ift/benchmark/baseline/corrected/211124_time083100_DSC00077.TIF")
config = Path("../config.json")
fluidflower = LargeFluidFlower(base, config)

baseline_offset = 10
start_index = 0
end_index = 127
originals = Path("/media/jakub/Elements/Jakub/benchmark/data/large_rig/c1").glob("*")
originals = list(sorted(originals))[baseline_offset:]

segmentations = Path("/media/jakub/Elements/Jakub/benchmark/develop/fingers/npy_segmentation_c1").glob("*.npy")
segmentations = list(sorted(segmentations))[start_index:end_index]

num_segmentations = len(segmentations)

# Contour analysis.
contour_analysis = darsia.ContourAnalysis(verbosity=False)
co2_g_analysis = darsia.ContourAnalysis(verbosity=False)
contour_evolution_analysis = darsia.ContourEvolutionAnalysis()

# Keep track of number, length of fingers
total_num_fingers = []
length_fingers = []
height_co2g = []

# Fetch reference time
ref_time = read_time_from_path(originals[0])
rel_time = []

# Start with a fixed one - as Darsia Image
for i in range(num_segmentations):

    # Convert segmentation to Image
    segmentation = darsia.Image(np.load(segmentations[i]), width=2.8, height=1.5)
    original = darsia.Image(cv2.cvtColor(cv2.imread(str(originals[i])), cv2.COLOR_BGR2RGB), width=2.8, height=1.5)

    # Add timestamp from title
    time = read_time_from_path(originals[i])
    segmentation.timestamp = time
    original.timestamp = time
    relative_time_hours = (time - ref_time).total_seconds() / 3600
    rel_time.append(relative_time_hours)
    
    # Plot the segmentation and box C
    if False:
        segmentation_img = np.zeros((*segmentation.img.shape[:2],3), dtype=float)
        for i in range(3):
            segmentation_img[:,:,i] = segmentation.img / np.max(segmentation.img)
        segmentation_img = skimage.img_as_ubyte(segmentation_img)
        
        # Add box A
        contours_box_A, _ = cv2.findContours(
            skimage.img_as_ubyte(fluidflower.mask_box_A),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(segmentation_img, contours_box_A, -1, (180, 180, 180), 3)
    
        # Add box C
        contours_box_C, _ = cv2.findContours(
            skimage.img_as_ubyte(fluidflower.mask_box_C),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(segmentation_img, contours_box_C, -1, (180, 180, 180), 3)
    
        plt.imshow(segmentation_img)
        plt.show()
    
    # Apply contour analysis for box C - only (all) CO2.
    contour_analysis.load_labels(
        img = segmentation,
        roi = fluidflower.box_C,
        values_of_interest = [1,2],
    )

    # Number of fingers of total CO2
    num_fingers = contour_analysis.number_peaks()
    total_num_fingers.append(num_fingers)
    print(f"Number of fingers: {num_fingers}")

    # Length of interface between CO2 and water, without the box.
    single_length_fingers = contour_analysis.length()
    length_fingers.append(single_length_fingers)
    #print(f"Contour length: {length_finger}")
    
    # Contour tip analysis.
    tips, valleys = contour_analysis.fingers()

    # Build up evolution of contours
    contour_evolution_analysis.add(tips, valleys)

    # Plot finger tips onto image
    if False:
        plt.figure("Original image with finger tips")
        plt.imshow(original.img)
        plt.scatter(tips[:,0,0] + fluidflower.box_C_roi[0,0], tips[:,0,1] + fluidflower.box_C_roi[0,0], c="r", s=20)
        plt.show()

    # Investigate CO2(g)
    co2_g_analysis.load_labels(
        img = segmentation,
        roi = fluidflower.box_A,
        values_of_interest = [2],
    )

    # Determine the mean height of the plume
    contour_co2g, _ = cv2.findContours(
        skimage.img_as_ubyte(co2_g_analysis.mask),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contour_co2g) == 0:
        height_co2g.append(0)
    elif len(contour_co2g) == 1:

        # Restrict the contour line to the one with lower y cooridinate (larger row component)
        contour_co2g = np.squeeze(contour_co2g[0])
        max0 = int(np.max(contour_co2g[:,0]))
        min0 = int(np.min(contour_co2g[:,0]))
        lower_contour = -np.ones(max0 - min0 + 1, dtype=int)
        for c in contour_co2g:
            lower_contour[int(c[0] - min0)] = max(lower_contour[int(c[0] - min0)], c[1])

        # Restrict to the 20-80% of the contour
        # and extract the mean of the contour line.
        mean_height_co2g = np.mean(lower_contour[int(0.2 * len(lower_contour)) : int(0.8 * len(lower_contour))])

        # Translate to global coordinate
        mean_height_co2g += fluidflower.box_A_roi[0,0]

        # Translate to local coordinates of box C
        mean_height_co2g -= fluidflower.box_C_roi[0,0]

        # Cache
        height_co2g.append(mean_height_co2g)

    else:
        height_co2g.append(0)
        print("Several contours... hopefully no fingers yet")
  
# Store number data to file
arr_num_fingers = np.transpose(
    np.vstack((np.array(rel_time), np.array(total_num_fingers)))
)
finger_header = f"Time in hours, number of finger tips in box C"
fmt = "%f", "%d"
np.savetxt("results/number_fingers.csv", arr_num_fingers, fmt=fmt, delimiter=",", header=finger_header)

arr_finger_length = np.transpose(
    np.vstack((np.array(rel_time), np.array(length_fingers)))
)
finger_header = f"Time in hours, finger length in box C"
fmt = "%f", "%f"
np.savetxt("results/length_fingers.csv", arr_finger_length, fmt=fmt, delimiter=",", header=finger_header)

assert False


# Find paths
contour_evolution_analysis.find_paths()

# Use auxiliary image just for extracting the dimensions
aux_img = darsia.Image(cv2.cvtColor(cv2.imread(str(originals[30])), cv2.COLOR_BGR2RGB), width=2.8, height=1.5)
aux_img_roi = darsia.extractROI(original, fluidflower.box_C)
#contour_evolution_analysis.plot(aux_img_roi)
contour_evolution_analysis.plot_paths(aux_img_roi)

# Determine total projected distance from CO2(g) to finger tips
plt.figure("Distance tip to CO2(g)")
for path in contour_evolution_analysis.paths:

    path_time = []
    path_distance = []

    for unit in path:
        path_time.append(rel_time[unit.time])
        pixel_distance = abs(unit.position[1] - height_co2g[unit.time])
        metric_distance = aux_img.coordinatesystem.pixelsToLength(pixel_distance, axis="y")
        path_distance.append(metric_distance)

    plt.plot(path_time, path_distance)

# Plot number of fingers
plt.figure("Number of fingers")
plt.plot(rel_time, total_num_fingers)
plt.figure("Total finger length, box C")
plt.plot(rel_time, length_fingers)
plt.figure("Mean height CO2g")
plt.plot(rel_time, height_co2g)
plt.show()
