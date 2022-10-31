from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as spsignal
import skimage

# ! ----  Auxiliary methods


def _labeled_mask_to_contour_mask(labeled_mask: np.ndarray, band_size) -> np.ndarray:
    """
    Starting from a boolean array identifying a region, find
    the contours with a user-defined bandwidth.

    Args:
        labeled_mask (np.ndarray): boolean array identifying a connected region.

    Returns:
        np.ndarray: boolean array identifying a band width of the contours
    """
    # Determine the contours of the labeled mask
    contours, _ = cv2.findContours(
        skimage.img_as_ubyte(labeled_mask),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
        # cv2.CHAIN_APPROX_SIMPLE, # TODO which one? test...
    )

    # Extract the contour as mask
    contour_mask = np.zeros(signal.shape[:2], dtype=bool)
    for c in contours:
        c = (c[:, 0, 1], c[:, 0, 0])
        contour_mask[c] = True

    # Dilate to generate a thick contour
    contour_mask = skimage.morphology.binary_dilation(
        contour_mask, np.ones((band_size, band_size), np.uint8)
    )

    # Convert to boolean mask
    contour_mask = skimage.img_as_bool(contour_mask)

    return contour_mask


# ! ----  Example

verbosity = False

labels = np.load("labels.npy")
signal = np.load("concentration_30.npy")
median_path = Path("median_30.npy")

plt.figure()
plt.imshow(labels)
plt.show()

# ESF: 1 (upper), 8, 9 (lower)
# C: 2, 3, 4
# D: 5. 6. 7
# Loewer: 10

# TODO use something like np.unique(labels) and exclude inactive labels (water)
all_labels = np.unique(labels)
label_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Find contours, contours bands and their pairwise intersections
print("Finding contours")
if False:

    # User parameter
    band_size = 50

    contour_mask = {}
    for label in label_set:
        contour_mask[label] = _labeled_mask_to_contour_mask(labels == label, band_size)
        np.save(f"contour_mask_{label}.npy", contour_mask[label])

else:
    contour_mask = {}
    for label in label_set:
        contour_mask[label] = np.load(f"contour_mask_{label}.npy")

# Define crucial coupling of interest. # TODO allow for restrictions - or make most general...
masked_labels = label_set

# Convert to array and sort
masked_labels = np.unique(masked_labels)

# TODO user defined parameter
overlap_threshold = 1000

# Find relevant couplings of masked labels.
print("Finding couplings")
couplings = []
for label1 in masked_labels:
    for label2 in masked_labels:

        # Consider directed pairs
        if label1 < label2:
            if verbosity:
                print(
                    "test overlap",
                    np.count_nonzero(
                        np.logical_and(contour_mask[label1], contour_mask[label2])
                    ),
                )

            # Check if labeled regions share significant part of contour
            if (
                np.count_nonzero(
                    np.logical_and(contour_mask[label1], contour_mask[label2])
                )
                > overlap_threshold
            ):
                couplings.append((label1, label2))

print("couplings", couplings)

# Prepare the layered median

median_disk_radius = 20

print("Find median")

if median_path.exists():
    median = np.load(median_path)

else:
    median = np.zeros(signal.shape[:2], dtype=np.uint8)
    for label in label_set:
        mask = labels == label
        median_label = skimage.filters.rank.median(
            signal, skimage.morphology.disk(median_disk_radius), mask=mask
        )
        median_label[~mask] = 0
        median = median + median_label

    # TODO rm later
    np.save(median_path, median)

# Find ratio of mean median values for each bondary.
# The idea will be that this ratio will be required
# for one-sided scaling in order to correct for
# discontinuities. Store the ratios in a dictionary.
# To keep track of orientation, consider sorted pairs
# of combinations. Also keep track which ratios are
# trustable or not (only if sufficient information
# provided).
print("Find ratios")

interface_ratio = {}
trustable = {}

# TODO include as user input
mean_thresh = 1

for coupling in couplings:

    # Fetch the single labels
    label1, label2 = coupling

    # Fetch the common boundary
    common_boundary = np.logical_and(contour_mask[label1], contour_mask[label2])

    # Restrict the common boundary to the separate subdomains
    roi1 = np.logical_and(common_boundary, labels == label1)
    roi2 = np.logical_and(common_boundary, labels == label2)

    # Consider the mean of the median of the signal on the separate bands
    mean1 = np.mean(median[roi1])
    mean2 = np.mean(median[roi2])

    # Check whether this result can be trusted - require sufficient signal.
    # TODO rm print statement
    print("debugging means", mean1, mean2)
    trustable[coupling] = min(mean1, mean2) >= mean_thresh

    # Define the ratio / later scaling - if value not trustable (for better usability) choose ratio equal to 1
    interface_ratio[coupling] = mean1 / mean2 if trustable[coupling] else 1

print(interface_ratio)
print(trustable)

# From interfaces to regions - find suitable scaling parameters
matrix = np.zeros((0, all_labels.shape[0]), dtype=float)
rhs = np.zeros((0, 1), dtype=float)
basis_vectors = np.eye

# Add inactive components and constraints.
for label in [0, 1]:
    basis_vector = np.zeros((1, all_labels.shape[0]), dtype=float)
    basis_vector[0, label] = 1
    matrix = np.vstack((matrix, basis_vector))
    rhs = np.vstack((rhs, np.array([[1]])))

# Add couplings.
for coupling in couplings:
    label1, label2 = coupling
    scaling_balance = np.zeros((1, all_labels.shape[0]), dtype=float)
    scaling_balance[0, label1] = interface_ratio[coupling]
    scaling_balance[0, label2] = -1
    matrix = np.vstack((matrix, scaling_balance))
    rhs = np.vstack((rhs, np.array([0])))

# Add info on similar sand types
for coupling in [(5, 10)]:
    label1, label2 = coupling
    similarity_balance = np.zeros((1, all_labels.shape[0]), dtype=float)
    similarity_balance[0, label1] = 1
    similarity_balance[0, label2] = -1
    matrix = np.vstack((matrix, similarity_balance))
    rhs = np.vstack((rhs, np.array([0])))

# Solve the over determined system
scaling = np.linalg.lstsq(matrix, np.ravel(rhs))[0]
print(scaling)

# Rescale signal using interface ratios
median = skimage.img_as_float(median)
rescaled_median = np.copy(median)
rescaled_signal = np.copy(signal)

for label in all_labels:
    rescaled_median[labels == label] *= scaling[label]
    rescaled_signal[labels == label] *= scaling[label]

median_from_rescaled_signal = skimage.img_as_float(
    skimage.filters.rank.median(
        rescaled_signal, skimage.morphology.disk(median_disk_radius)
    )
)
tvd_from_rescaled_signal = skimage.restoration.denoise_tv_bregman(
    rescaled_signal, weight=1e-2, eps=1e-4, max_num_iter=200
)

plt.figure("labels")
plt.imshow(labels)
plt.figure("signal")
plt.imshow(signal)
plt.figure("median")
plt.imshow(median)
plt.figure("rescaled signal")
plt.imshow(rescaled_signal)
plt.figure("rescaled median")
plt.imshow(rescaled_median)
plt.figure("median from rescaled signal")
plt.imshow(median_from_rescaled_signal)
plt.figure("tvd from rescaled signal")
plt.imshow(tvd_from_rescaled_signal)

plt.show()
