import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage

img = np.load("base_c1.npy")

# Remove shadow boundary
img = img[:, 100:-100]

# Deactivate colorchecker in the analysis
mask = np.ones(img.shape[:2], dtype=bool)
color_checker = (slice(0, 500), slice(0, 450))
mask[color_checker] = False

# Setup config for segmentation
config = {
    # Preprocessing
    "median disk radius": 20,
    "rescaling factor": 0.1,
    # Tuning parameters
    "marker_points": np.array(
        [
            [260, 3560],
            [1100, 3560],
            [1260, 5560],
            [1440, 5530],
            [1720, 5470],
            [2510, 5170],
            [3000, 4700],
            [3110, 310],
            [1330, 740],
            [1510, 740],
            [1730, 740],
            [2000, 740],
            [2220, 830],
            [2660, 420],
            [3160, 970],
            [1410, 3080],
            [1640, 2930],
            [1950, 3080],
            [2210, 3220],
            [3384, 900],
            [2378, 606],
            [2218, 1567],
            [1945, 5400],
            [2315, 5500],
            # [4110, 4330], # attempt for lowest zone
            [1450, 4055],  # attempt for open fault
            [1065, 2572],  # Atempt for sealed fault
        ]
    ),
    "region_size": 20,
    "mask scharr": mask,
    # Postprocessing
    "dilation size": 10,
    "boundary size": 80,  # TODO decrease to 0
    # Verbosity
    "verbosity": False,
}

# Segmentation
labels = daria.segment(
    img, markers_method="supervised", edges_method="scharr", **config
)

# Store the result
np.save("labels_geometry_segmentation.npy", labels)

plt.figure()
plt.imshow(img)
plt.imshow(labels, alpha=0.3)
plt.show()
