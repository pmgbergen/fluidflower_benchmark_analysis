"""
Determine compaction of FluidFlower by comparing two different images.

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

# C1
c1_npy = np.load("images/c1.npy")
c1_img = darsia.Image(c1_npy, width=2.8, height=1.5)
c1_water = np.load("labels/c1_water.npy")
c1_mask_img = darsia.Image(np.logical_not(c1_water), width=2.8, height=1.5)

# C2 
c2_npy = np.load("images/c2.npy")
c2_img = darsia.Image(c2_npy, width=2.8, height=1.5)
c2_water = c1_water.copy()
c2_mask_img = darsia.Image(np.logical_not(c2_water), width=2.8, height=1.5)

## C3
#c3_npy = np.load("images/c3.npy")
#c3_img = darsia.Image(c3_npy, width=2.8, height=1.5)
#c3_water = np.load("labels/c3_water.npy")
#
## C4
#c4_npy = np.load("images/c4.npy")
#c4_img = darsia.Image(c4_npy, width=2.8, height=1.5)
#c4_water = c4_water.copy()
#
## C5
#c5_npy = np.load("images/c5.npy")
#c5_img = darsia.Image(c5_npy, width=2.8, height=1.5)
#c5_water = c3_water.copy()

# Fix a reference image
img_ref = c1_img.copy()
img_dst = c2_img.copy()

# Masks
mask_ref = c1_mask_img.copy()
mask_dst = c2_mask_img.copy()
mask_ref_0 = mask_ref.copy()

# Preliminaries: Define base config for compaction analysis
base_config_compaction = {
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

# TODO
print("gh")

# Total compaction - initialize
config_compaction = copy.deepcopy(base_config_compaction)
config_compaction["N_patches"] = [64, 32]
#config_compaction["N_patches"] = [128, 64]
compaction_analysis = darsia.ReversedCompactionAnalysis(img_ref, **config_compaction)

# One iteration of multiscale compaction analysis
def iteration(
    img_dst, img, mask_dst, mask, patches, plot=False
) -> tuple[darsia.Image, darsia.ReversedCompactionAnalysis]:
    # Define compaction analysis tool
    config_compaction = copy.deepcopy(base_config_compaction)
    config_compaction["N_patches"] = patches
    compaction_analysis = darsia.ReversedCompactionAnalysis(
        img_dst, mask=mask_dst, **config_compaction
    )

    transformed_img, patch_translation = compaction_analysis(
        img, plot_patch_translation=plot, return_patch_translation=True, mask=mask
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


# ! ---- 0. Iteration
print("0. iteration")
if False:
    plt.figure("Initial comparison")
    plt.imshow(skimage.util.compare_images(img_dst.img, img_ref.img, method="blend"))
    plt.show()
Path("compaction_corrected").mkdir(exist_ok=True)
cv2.imwrite(
    "compaction_corrected/ref_0.jpg",
    img_ref.img,
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

# ! ---- Multiscale analysis
levels = [
    #[4, 2],
    #[8, 4],
    #[16, 8],
    #[32, 16],
    [64, 32],
    #[128, 64],
#    [256, 128],
]

print("1. iteration")
_, compaction_1 = iteration(img_dst, img_ref, mask_dst, mask_ref_0, levels[0])
mask_ref_1 = compaction_1.apply(mask_ref_0)
compaction_analysis.deduct(compaction_1)
img_ref_1 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_1.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_1.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

#print("2. iteration")
#_, compaction_2 = iteration(img_dst, img_ref_1, mask_dst, mask_ref_1, levels[1])
#mask_ref_2 = compaction_2.apply(mask_ref_1)
#compaction_analysis.add(compaction_2)
#img_ref_2 = compaction_analysis.apply(img_ref)
#
#cv2.imwrite(
#    "compaction_corrected/ref_2.jpg",
#    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_2.img, 0, 1)), cv2.COLOR_RGB2BGR),
#    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
#)
#
#print("3. iteration")
#_, compaction_3 = iteration(img_dst, img_ref_2, mask_dst, mask_ref_2, levels[2])
#mask_ref_3 = compaction_3.apply(mask_ref_2)
#compaction_analysis.add(compaction_3)
#img_ref_3 = compaction_analysis.apply(img_ref)
#
#cv2.imwrite(
#    "compaction_corrected/ref_3.jpg",
#    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_3.img, 0, 1)), cv2.COLOR_RGB2BGR),
#    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
#)
#
#print("4. iteration")
#_, compaction_4 = iteration(img_dst, img_ref_3, mask_dst, mask_ref_3, levels[3])
#mask_ref_4 = compaction_3.apply(mask_ref_3)
#compaction_analysis.add(compaction_4)
#img_ref_4 = compaction_analysis.apply(img_ref)
#
#cv2.imwrite(
#    "compaction_corrected/ref_4.jpg",
#    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_4.img, 0, 1)), cv2.COLOR_RGB2BGR),
#    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
#)
#
#print("5. iteration")
#_, compaction_5 = iteration(img_dst, img_ref_4, mask_dst, mask_ref_4, levels[4])
#mask_ref_5 = compaction_5.apply(mask_ref_4)
#compaction_analysis.add(compaction_5)
#img_ref_5 = compaction_analysis.apply(img_ref)
#
#cv2.imwrite(
#    "compaction_corrected/ref_5.jpg",
#    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_5.img, 0, 1)), cv2.COLOR_RGB2BGR),
#    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
#)

if True:
    mask_plot = mask_ref.copy()
    #mask_plot.img = skimage.img_as_bool(cv2.resize(skimage.img_as_ubyte(mask_plot.img), (64, 32), interpolation = cv2.INTER_NEAREST))
    compaction_analysis.plot(scaling = 10., mask = mask_plot)

if False:
    plt.figure("0")
    plt.imshow(mask_ref_0.img)
    plt.figure("1")
    plt.imshow(mask_ref_1.img)
    plt.figure("2")
    plt.imshow(mask_ref_2.img)
    plt.figure("3")
    plt.imshow(mask_ref_3.img)
    plt.figure("4")
    plt.imshow(mask_ref_4.img)
    plt.figure("5")
    plt.imshow(mask_ref_5.img)
    plt.show()

# ! ---- 3. Post analysis

# Apply the total deformation
img_ref_deformed = compaction_analysis.apply(img_ref)
labels_deformed = compaction_analysis.apply(labels_ref)

np.save("results/labels_ref.npy", labels_ref.img)
np.save("results/labels_deformed.npy", labels_deformed.img)

# For the entire rig
reservoir_deformed = compaction_analysis.apply(reservoir_ref)
np.save("results/reservoir_ref.npy", reservoir_ref.img)
np.save("results/reservoir_deformed.npy", reservoir_deformed.img)

# For the top boundary of the reservoir
reservoir_top_deformed = compaction_analysis.apply(reservoir_top_ref)
np.save("results/reservoir_top_ref.npy", reservoir_top_ref.img)
np.save("results/reservoir_top_deformed.npy", reservoir_top_deformed.img)

# Plot the differences between the two original images and after the transformation.
if True:
    plt.figure("Comparison of deformed ref and dst")
    plt.imshow(
        skimage.util.compare_images(img_dst.img, img_ref_deformed.img, method="blend")
    )

    print(img_ref_5.img.shape, img_ref_deformed.img.shape)

    plt.figure("Comparison of deformed ref x 2")
    plt.imshow(
        skimage.util.compare_images(img_ref_5.img, img_ref_deformed.img, method="blend")
    )

    plt.figure("original labels")
    plt.imshow(labels_ref.img)

    plt.figure("deformed labels")
    plt.imshow(labels_deformed.img)

    plt.figure("original reservoir")
    plt.imshow(reservoir_ref.img)

    plt.figure("deformed reservoir")
    plt.imshow(reservoir_deformed.img)

    plt.show()

    # Store compaction corrected image
    cv2.imwrite(
        "compaction_corrected/ref_reverse.jpg",
        cv2.cvtColor(
            skimage.img_as_ubyte(np.clip(img_ref_deformed.img, 0, 1)), cv2.COLOR_RGB2BGR
        ),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )

if False:
    # Determine the displacement in metric units on pixel level.
    displacement = compaction_analysis.displacement()

    # Separate into the single components
    displacement_x_vector = displacement[:, 0]
    displacement_y_vector = displacement[:, 1]

    # Convert into mesh format
    Ny, Nx = labels_ref.img.shape[:2]
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
