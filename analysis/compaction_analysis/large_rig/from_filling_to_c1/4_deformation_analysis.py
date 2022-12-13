"""
Quantify the deformation.
"""

import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import darsia

# Image after sand filling
img_zero = darsia.Image(
    np.load("corrected/zero.npy"),
    width=2.8,
    height=1.5
)

# Fix a reference image
labels_ref = darsia.Image(
    np.load("results/labels_ref.npy"),
    width=2.8,
    height=1.5
)
labels_deformed = darsia.Image(
    np.load("results/labels_deformed.npy"),
    width=2.8,
    height=1.5
)

# Displacement
displacement_x = np.load("results/displacement_x.npy")
displacement_y = np.load("results/displacement_y.npy")
rows, cols = displacement_x.shape[:2]
displacement = np.zeros((rows, cols, 2), dtype=float)
displacement[:,:,0] = displacement_x
displacement[:,:,1] = displacement_y

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

def mean_displacement_analysis(labels_ref, displacement, thresh = 10000):

    regions_ref = skimage.measure.regionprops(labels_ref)

    displacement_y = displacement[:,:,1]
    mean_displacement_y = []
    label_values = []
    for l in range(len(regions_ref)):
        if regions_ref[l].area > thresh:

            # Determine roi
            label_value = regions_ref[l].label
            mask = labels_ref == label_value

            # Determine mean displacement in y direction
            single_mean_displacment_y = np.sum(displacement_y[mask]) / np.count_nonzero(mask)

            # Collect results
            mean_displacement_y.append(single_mean_displacement_y)
            label_values.append(label_value)

    return displacement_y, label_values

def mean_strain_analysis(labels_ref, displacement, thresh=10000):
    # Only 1d? TODO
    regions_ref = skimage.measure.regionprops(labels_ref)

    displacement_y = displacement[:, :, 1]
    strain_y = np.gradient(displacement_y, axis=0)

    mean_strain_y = []
    label_values = []
    for l in range(len(regions_ref)):
        if regions_ref[l].area > thresh:

            # Determine roi
            label_value = regions_ref[l].label
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

# Centroid displacement based analysis
(
    displacement_row,
    displacement_col,
    label_values_displacement,
) = centroid_displacement_analysis(labels_deformed.img, labels_ref.img)

# Strain based analysis
strain_y, label_values_strain = mean_strain_analysis(labels_ref.img, displacement)

# Double check all labels
assert label_values_ref == label_values_deformed
assert label_values_ref == label_values_displacement
assert label_values_ref == label_values_strain

plt.figure("areas")
plt.plot(area_ref)
plt.plot(area_deformed)

plt.figure("relative deformation wrt ref")
ratio = area_deformed / np.maximum(area_ref, 1) - 1
plt.plot(ratio)

plt.figure("row pixel displacement - centroid")
plt.plot(displacement_row)
plt.figure("col pixel displacement - centroid")
plt.plot(displacement_col)

plt.figure("strain y")
plt.plot(strain_y)
plt.show()

# Spatial illustration of the results

label_ratio = np.zeros(labels_ref.img.shape[:2], dtype=float)
label_strain_y = np.zeros(labels_ref.img.shape[:2], dtype=float)
label_centroid_displacement = np.zeros(labels_ref.img.shape[:2], dtype=float)
for i, label in enumerate(label_values_ref):
    mask = labels_ref.img == label
    pixel_height = 1.5 / labels_ref.img.shape[0]
    label_ratio[mask] = ratio[i]
    label_strain_y[mask] = strain_y[i]
    label_centroid_displacement[mask] = -displacement_row[i] * pixel_height

plt.figure("relative deformation wrt ref, spatial")
plt.imshow(label_ratio)
plt.colorbar()
plt.figure("y displacement")
plt.imshow(label_centroid_displacement)
plt.colorbar()
plt.figure("strain y, spatial")
plt.imshow(label_strain_y)
plt.colorbar()
plt.show()

