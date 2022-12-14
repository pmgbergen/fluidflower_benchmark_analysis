"""
Determine compaction of FluidFlower by comparing two different images.

Reference image is taken just after filling the rig with sand.
Test image is just before C1.

"""

import copy
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower

# ! ----- Preliminaries - prepare two images for compaction analysis

# Image of benchmark description
img_src_arr = np.load("corrected/src.npy")
mask_src_arr = np.load("blackened/src_mask.npy")

# Image before running C1
img_dst_arr = np.load("corrected/dst.npy")
mask_dst_arr = np.load("blackened/dst_mask.npy")

# Darsia Image variants
img_src = darsia.Image(img_src_arr, width=2.8, height=1.5)
img_dst = darsia.Image(img_dst_arr, width=2.8, height=1.5)
mask_dst = darsia.Image(np.logical_not(mask_dst_arr), width=2.8, height=1.5)
mask_src = darsia.Image(np.logical_not(mask_src_arr), width=2.8, height=1.5)

# Verbosity
if False:
    plt.figure("src")
    plt.imshow(img_src.img)
    plt.figure("dst")
    plt.imshow(img_dst.img)
    plt.show()

# Preliminaries: Define base config for compaction analysis
base_config_compaction = {
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

# ! ---- Auxiliary methods for defining the multiscale approach

# Define one iteration of multiscale compaction analysis
def iteration(
    img_dst, img_src, mask_dst, mask_src, patches, plot=False
) -> tuple[darsia.Image, darsia.ReversedCompactionAnalysis]:
    # Define compaction analysis tool
    config_compaction = copy.deepcopy(base_config_compaction)
    config_compaction["N_patches"] = patches
    compaction_analysis = darsia.ReversedCompactionAnalysis(
        img_dst, mask=mask_dst, **config_compaction
    )

    transformed_img, patch_translation = compaction_analysis(
        img_src,
        plot_patch_translation=plot,
        return_patch_translation=True,
        mask=mask_src,
    )

    if plot:
        plt.figure("comparison")
        plt.imshow(
            skimage.util.compare_images(
                img_dst.img, transformed_img.img, method="blend"
            )
        )
        plt.show()

    return transformed_img, compaction_analysis


def refine_patches(p: list, levels=1) -> list:
    for level in range(levels):
        p = [p[i] * 2 for i in range(len(p))]
    return p


# ! ---- 0. Iteration
print("0. iteration")

# Make initial comparison
if False:
    plt.figure("Initial comparison")
    plt.imshow(skimage.util.compare_images(img_dst.img, img_src.img, method="blend"))
    plt.show()

# Store the initial solution
Path("compaction_corrected").mkdir(exist_ok=True)
cv2.imwrite(
    "compaction_corrected/src_0.jpg",
    img_src.img,
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

# ! ---- Multiscale analysis

# Specification of ms approach
patches = [4, 2]
num_ms_iterations = 5

# Total compaction
config_compaction = copy.deepcopy(base_config_compaction)
config_compaction["N_patches"] = refine_patches(patches, num_ms_iterations - 1)
compaction_analysis = darsia.ReversedCompactionAnalysis(img_src, **config_compaction)

# Initialization
img_src_deformed = img_src.copy()
mask_src_deformed = mask_src.copy()

# Iteration
for i in range(1, num_ms_iterations + 1):

    # Apply one level of multiscale compaction
    print(f"{i}. iteration")
    _, compaction_next = iteration(
        img_dst, img_src_deformed, mask_dst, mask_src_deformed, patches
    )

    # Update the total compaction analysis
    compaction_analysis.add(compaction_next)

    # Update for next iteration
    img_src_deformed = compaction_analysis.apply(img_src)
    mask_src_deformed = compaction_analysis.apply(mask_src)
    patches = refine_patches(patches)

    # Store deformed image
    cv2.imwrite(
        f"compaction_corrected/src_{i}.jpg",
        cv2.cvtColor(
            skimage.img_as_ubyte(np.clip(img_src_deformed.img, 0, 1)), cv2.COLOR_RGB2BGR
        ),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )

# Plot the result
if True:
    compaction_analysis.plot()

# ! ---- 3. Post analysis

# Plot the differences between the two original images and after the transformation.
if True:
    plt.figure("Comparison of deformed src and dst")
    plt.imshow(
        skimage.util.compare_images(img_dst.img, img_src_deformed.img, method="blend")
    )
    plt.show()

# Define boxes A, B, C as defined in the benchmark description.
# Box A, B, C in metric coorddinates (left top, and right lower point).
box_A = np.array([[1.1, 0.6], [2.8, 0.0]])
box_B = np.array([[0.0, 1.2], [1.1, 0.6]])
box_C = np.array([[1.1, 0.4], [2.6, 0.1]])
extended_box_C = np.array([[0.0, 0.0], [2.8, 0.4]])

# Box A, B, C in terms of pixels, adapted to the size of the base image
box_A_roi = img_src.coordinatesystem.coordinateToPixel(box_A)
box_B_roi = img_src.coordinatesystem.coordinateToPixel(box_B)
box_C_roi = img_src.coordinatesystem.coordinateToPixel(box_C)
extended_box_C_roi = img_src.coordinatesystem.coordinateToPixel(extended_box_C)

# Boolean masks for boxes A, B, C, adapted to the size of the base image
mask_box_A_arr = np.zeros(img_src.img.shape[:2], dtype=bool)
mask_box_B_arr = np.zeros(img_src.img.shape[:2], dtype=bool)
mask_box_C_arr = np.zeros(img_src.img.shape[:2], dtype=bool)
mask_extended_box_C_arr = np.zeros(img_src.img.shape[:2], dtype=bool)

mask_box_A_arr[darsia.bounding_box(box_A_roi)] = True
mask_box_B_arr[darsia.bounding_box(box_B_roi)] = True
mask_box_C_arr[darsia.bounding_box(box_C_roi)] = True
mask_extended_box_C_arr[darsia.bounding_box(extended_box_C_roi)] = True

mask_box_A = darsia.Image(mask_box_A_arr, width=2.8, height=1.5)
mask_box_B = darsia.Image(mask_box_B_arr, width=2.8, height=1.5)
mask_box_C = darsia.Image(mask_box_C_arr, width=2.8, height=1.5)
mask_extended_box_C = darsia.Image(mask_extended_box_C_arr, width=2.8, height=1.5)

# Deform boxes
mask_deformed_box_A = compaction_analysis.apply(mask_box_A)
mask_deformed_box_B = compaction_analysis.apply(mask_box_B)
mask_deformed_box_C = compaction_analysis.apply(mask_box_C)
mask_deformed_extended_box_C = compaction_analysis.apply(mask_extended_box_C)

# Overlay the original image with contours of Box A
contours_box_A, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_box_A.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box B
contours_box_B, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_box_B.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box C
contours_box_C, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_box_C.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box C
contours_extended_box_C, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_extended_box_C.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box A
contours_deformed_box_A, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_deformed_box_A.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box B
contours_deformed_box_B, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_deformed_box_B.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box C
contours_deformed_box_C, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_deformed_box_C.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Overlay the original image with contours of Box C
contours_deformed_extended_box_C, _ = cv2.findContours(
    skimage.img_as_ubyte(mask_deformed_extended_box_C.img),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Benchmark setup with original boxes
bm_img_with_original_boxes = np.copy(img_src.img)
cv2.drawContours(bm_img_with_original_boxes, contours_box_A, -1, (180, 180, 180), 3)
cv2.drawContours(bm_img_with_original_boxes, contours_box_B, -1, (180, 180, 180), 3)
cv2.drawContours(bm_img_with_original_boxes, contours_box_C, -1, (180, 180, 180), 3)
cv2.drawContours(
    bm_img_with_original_boxes, contours_extended_box_C, -1, (180, 180, 180), 3
)

# C1 run with original boxes
c1_img_with_original_boxes = np.copy(img_dst.img)
cv2.drawContours(c1_img_with_original_boxes, contours_box_A, -1, (180, 180, 180), 3)
cv2.drawContours(c1_img_with_original_boxes, contours_box_B, -1, (180, 180, 180), 3)
cv2.drawContours(c1_img_with_original_boxes, contours_box_C, -1, (180, 180, 180), 3)
cv2.drawContours(
    c1_img_with_original_boxes, contours_extended_box_C, -1, (180, 180, 180), 3
)

# Plot contours of deformed boxes on c1 images
c1_img_with_deformed_boxes = np.copy(img_dst.img)
cv2.drawContours(
    c1_img_with_deformed_boxes, contours_deformed_box_A, -1, (180, 180, 180), 3
)
cv2.drawContours(
    c1_img_with_deformed_boxes, contours_deformed_box_B, -1, (180, 180, 180), 3
)
cv2.drawContours(
    c1_img_with_deformed_boxes, contours_deformed_box_C, -1, (180, 180, 180), 3
)
cv2.drawContours(
    c1_img_with_deformed_boxes, contours_deformed_extended_box_C, -1, (180, 180, 180), 3
)

# Plot contours of both boxes on top of C1
c1_img_with_both_boxes = np.copy(img_dst.img)
cv2.drawContours(c1_img_with_both_boxes, contours_box_A, -1, (180, 180, 180), 3)
cv2.drawContours(c1_img_with_both_boxes, contours_box_B, -1, (180, 180, 180), 3)
cv2.drawContours(c1_img_with_both_boxes, contours_box_C, -1, (180, 180, 180), 3)
cv2.drawContours(
    c1_img_with_both_boxes, contours_extended_box_C, -1, (180, 180, 180), 3
)
cv2.drawContours(c1_img_with_both_boxes, contours_deformed_box_A, -1, (180, 0, 180), 3)
cv2.drawContours(c1_img_with_both_boxes, contours_deformed_box_B, -1, (180, 0, 180), 3)
cv2.drawContours(c1_img_with_both_boxes, contours_deformed_box_C, -1, (180, 0, 180), 3)
cv2.drawContours(
    c1_img_with_both_boxes, contours_deformed_extended_box_C, -1, (180, 0, 180), 3
)

# Plot
if True:
    plt.figure("Benchmark setup with original boxes.")
    plt.imshow(bm_img_with_original_boxes)
    plt.figure("C1 with original boxes.")
    plt.imshow(c1_img_with_original_boxes)
    plt.figure("C1 with deformed boxes.")
    plt.imshow(c1_img_with_deformed_boxes)
    plt.figure("C1 with both boxes.")
    plt.imshow(c1_img_with_both_boxes)
    plt.show()

assert False

if False:
    # Determine the displacement in metric units on pixel level.
    displacement = compaction_analysis.displacement()

    # Separate into the single components
    displacement_x_vector = displacement[:, 0]
    displacement_y_vector = displacement[:, 1]

    # Convert into mesh format
    Ny, Nx = labels_src.img.shape[:2]
    displacement_x = displacement_x_vector.reshape(Ny, Nx)
    displacement_y = displacement_y_vector.reshape(Ny, Nx)
    np.save("results/displacement_x.npy", displacement_x)
    np.save("results/displacement_y.npy", displacement_y)

    plt.figure("displacement x")
    plt.imshow(displacement_x)
    plt.figure("displacement y")
    plt.imshow(displacement_y)
    plt.show()
else:
    displacement_x = np.load("results/displacement_x.npy")
    displacement_y = np.load("results/displacement_y.npy")
