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
# water_zero = np.load("blackened/zero_water.npy")
water_zero = cv2.resize(
    water_zero.astype(int),
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

# Masks
mask_dst = darsia.Image(np.logical_not(water_dst), width=2.8, height=1.5)
mask_ref_0 = darsia.Image(np.logical_not(water_zero), width=2.8, height=1.5)
labels_ref = darsia.Image(labels_ref, width=2.8, height=1.5)
num_labels = len(np.unique(labels_ref.img))
print(num_labels)

plt.imshow(labels_ref.img)
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
compaction_analysis.add_correctly(compaction_2)
img_ref_2 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_2.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_2.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("3. iteration")
_, compaction_3 = iteration(img_dst, img_ref_2, mask_dst, mask_ref_2, [16, 8])
mask_ref_3 = compaction_3.apply(mask_ref_2)
compaction_analysis.add_correctly(compaction_3)
img_ref_3 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_3.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_3.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("4. iteration")
_, compaction_4 = iteration(img_dst, img_ref_3, mask_dst, mask_ref_3, [32, 16])
mask_ref_4 = compaction_3.apply(mask_ref_3)
compaction_analysis.add_correctly(compaction_4)
img_ref_4 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_4.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_4.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("5. iteration")
_, compaction_5 = iteration(img_dst, img_ref_4, mask_dst, mask_ref_4, [64, 32])
mask_ref_5 = compaction_5.apply(mask_ref_4)
compaction_analysis.add_correctly(compaction_5)
img_ref_5 = compaction_analysis.apply(img_ref)

cv2.imwrite(
    "compaction_corrected/ref_5.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_ref_5.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

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

# Plot the differences between the two original images and after the transformation.
if False:
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

    plt.show()

    # Store compaction corrected image
    cv2.imwrite(
        "compaction_corrected/ref_reverse.jpg",
        cv2.cvtColor(
            skimage.img_as_ubyte(np.clip(img_ref_deformed.img, 0, 1)), cv2.COLOR_RGB2BGR
        ),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )


if True:
    # Determine the displacement in metric units on pixel level.
    displacement = compaction_analysis.displacement()

    plt.figure("displacement x")
    plt.imshow(displacement[:, :, 0])
    plt.figure("displacement y")
    plt.imshow(displacement[:, :, 1])
    plt.show()

    np.save("results/displacement.npy", displacement)
else:
    displacement = np.load("results/displacement.npy")

# ! ---- Analysis tools

# Analysis of deformed labels


def area_analysis(labels, thresh=10000):
    regions = skimage.measure.regionprops(labels)

    # Area
    area = []
    label_values = []
    for l in range(len(regions)):
        if regions[l].area > thresh:
            area.append(regions[l].area)
            label_values.append(regions[l].label)

    return area, label_values


# Mean displacement per label
def centroid_displacement_analysis(labels, labels_ref, thresh=10000):
    regions = skimage.measure.regionprops(labels)
    regions_ref = skimage.measure.regionprops(labels_ref)

    # Collect centroids
    centroids = []
    label_values = []
    for l in range(len(regions)):
        if regions[l].area > thresh:
            centroids.append(regions[l].centroid)
            label_values.append(regions[l].label)
    centroids_ref = []
    label_values_ref = []
    for l in range(len(regions_ref)):
        if regions_ref[l].area > thresh:
            centroids_ref.append(regions_ref[l].centroid)
            label_values_ref.append(regions_ref[l].label)

    assert len(centroids) == len(centroids_ref)
    assert all([label_values[i] == label_values_ref[i] for i in range(len(centroids))])

    displacement_row = []
    displacement_col = []
    for i in range(len(centroids)):
        displacement_row.append(centroids[i][0] - centroids_ref[i][0])
        displacement_col.append(centroids[i][1] - centroids_ref[i][1])

    return displacement_row, displacement_col, label_values


def mean_strain_analysis(labels_ref, displacement, thresh=10000):
    # Only 1d? TODO

    displacement_y = displacement[:, :, 1]
    strain_y = np.gradient(displacement_y, axis=0)

    mean_strain_y = []
    label_values = []
    for l in range(len(regions)):
        if regions[l].area > thresh:

            # Determine roi
            label_value = regions[l].label
            mask = labels_ref == label_value

            # Determine mean displacement in y direction
            single_mean_strain_y = np.sum(strain_y[mask]) / np.count_nonzero(mask)

            # Collect results
            mean_strain_y.append(single_mean_strain_y)
            label_values.append(label_value)

    return mean_strain_y, label_values


# ! ---- Actual analysis

# Volume/area base analysis
area_ref, label_values_ref = area_analysis(labels_ref.img)
area_deformed, label_values_deformed = area_analysis(labels_deformed.img)

print(label_values_ref)
print(label_values_deformed)

plt.figure("areas")
plt.plot(area_ref)
plt.plot(area_deformed)
plt.show()

plt.figure("relative deformation wrt ref")
ratio = area_deformed / np.maximum(area_ref, 1) - 1
plt.plot(ratio)
plt.show()

# Centroid displacement based analysis
(
    displacement_row,
    displacement_col,
    label_values_displacement,
) = centroid_displacement_analysis(labels_deformed.img, labels_ref.img)
plt.figure("row")
plt.plot(displacement_row)
plt.figure("col")
plt.plot(displacement_col)
plt.show()

print(label_values_displacement)

# Strain based analysis
strain_y, label_values_strain = mean_strain_analysis(labels_ref.img, displacement)

print(label_values_strain)

plt.figure("strain")
plt.plot(strain_y)
plt.show()

assert False

# Divergence integrated over the domain
divergence = compaction_analysis.divergence()

plt.figure("divergence")
plt.imshow(divergence)
plt.show()

assert False
