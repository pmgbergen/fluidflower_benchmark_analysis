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

# Image after sand filling
img_zero_npy = np.load("corrected/zero.npy")
img_zero = darsia.Image(img_zero_npy, width=2.8, height=1.5)
water_zero = np.load("blackened/zero_water_and_fault.npy")
water_only_zero = np.load("blackened/zero_water.npy")
water_zero = cv2.resize(
    water_zero.astype(int),
    tuple(reversed(img_zero.img.shape[:2])),
    interpolation=cv2.INTER_NEAREST,
).astype(bool)
water_only_zero = cv2.resize(
    water_only_zero.astype(int),
    tuple(reversed(img_zero.img.shape[:2])),
    interpolation=cv2.INTER_NEAREST,
).astype(bool)
labels_zero = np.load("blackened/zero_labels.npy").astype(int)
plt.imshow(labels_zero)
plt.show()


# # Image before running C1
img_dst_npy = np.load("corrected/dst.npy")
img_dst = darsia.Image(img_dst_npy, width=2.8, height=1.5)
water_dst = np.load("blackened/dst_water_and_fault.npy")
# water_dst = np.load("blackened/dst_water.npy")
water_dst = cv2.resize(
    water_dst.astype(int),
    tuple(reversed(img_zero.img.shape[:2])),
    interpolation=cv2.INTER_NEAREST,
).astype(bool)

if False:
    plt.figure("zero")
    plt.imshow(img_zero.img)
    plt.figure("dst")
    plt.imshow(img_dst.img)
    plt.show()

# Fix a reference image
img_ref = img_zero.copy()
labels_ref = labels_zero.copy()
plt.imshow(labels_ref)
plt.show()

# Masks
mask_dst = darsia.Image(np.logical_not(water_dst), width=2.8, height=1.5)
mask_ref_0 = darsia.Image(np.logical_not(water_zero), width=2.8, height=1.5)
labels_ref = darsia.Image(labels_ref, width=2.8, height=1.5)
num_labels = len(np.unique(labels_ref.img))

# Reservoir
reservoir_ref = darsia.Image(np.logical_not(water_only_zero), width=2.8, height=1.5)

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
print("reduce the compaction to one number. what is it?")
print("gh")
# assert False

# Total compaction - initialize
config_compaction = copy.deepcopy(base_config_compaction)
config_compaction["N_patches"] = [64, 32]
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
print("1. iteration")
_, compaction_1 = iteration(img_dst, img_ref, mask_dst, mask_ref_0, [4, 2])
mask_ref_1 = compaction_1.apply(mask_ref_0)
compaction_analysis.deduct(compaction_1)
img_ref_1 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_1.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_1.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("2. iteration")
_, compaction_2 = iteration(img_dst, img_ref_1, mask_dst, mask_ref_1, [8, 4])
mask_ref_2 = compaction_2.apply(mask_ref_1)
compaction_analysis.add(compaction_2)
img_ref_2 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_2.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_2.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("3. iteration")
_, compaction_3 = iteration(img_dst, img_ref_2, mask_dst, mask_ref_2, [16, 8])
mask_ref_3 = compaction_3.apply(mask_ref_2)
compaction_analysis.add(compaction_3)
img_ref_3 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_3.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_3.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("4. iteration")
_, compaction_4 = iteration(img_dst, img_ref_3, mask_dst, mask_ref_3, [32, 16])
mask_ref_4 = compaction_3.apply(mask_ref_3)
compaction_analysis.add(compaction_4)
img_ref_4 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_4.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_4.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("5. iteration")
_, compaction_5 = iteration(img_dst, img_ref_4, mask_dst, mask_ref_4, [64, 32])
mask_ref_5 = compaction_5.apply(mask_ref_4)
compaction_analysis.add(compaction_5)
img_ref_5 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_5.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_5.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

if True:
    compaction_analysis.plot()

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
