"""
Determine compaction of FluidFlower by comparing two different images.
"""
from pathlib import Path

import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage

from fluidflower import BenchmarkRig

# ! ---- Data from c1, already processed
base_c1_img = np.load("base_c1.npy")
base_c1 = daria.Image(base_c1_img, color_space="RGB", width=2.8, height=1.5)

# ! ---- c2
folder = Path("/home/jakub/images/ift/benchmark/c2")
baseline = Path("baseline")
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))
ff = BenchmarkRig(baseline_images, config="./config.json", update_setup=False)
base_c2 = ff.base

# Make compatible with TranslationEstimator
base_c1.img = skimage.img_as_ubyte(base_c1.img)
base_c2.img = skimage.img_as_ubyte(base_c2.img)

# Compute drift between setups c1 and c2 (after correction)
roi_cc = (slice(150, 550), slice(300, 550))
translation_estimator = daria.TranslationEstimator()
base_c1_img = base_c1.img
base_c2_img = base_c2.img
new_c1 = translation_estimator.match_roi(base_c1_img, base_c2_img, roi_cc)

plt.figure()
plt.imshow(base_c1_img)
plt.imshow(new_c1, alpha=0.3)
plt.figure()
plt.imshow(base_c2_img)
plt.imshow(new_c1, alpha=0.3)
plt.show()

# Compute compaction between translated c1 and original c2

# Define compaction analysis tool
config = {
    # Define the number of patches in x and y directions
    "N_patches": [20, 10],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

new_base_c1 = daria.Image(new_c1, color_space="RGB", width=2.8, height=1.5)
compaction_analysis = daria.CompactionAnalysis(base_c2, **config)

# Apply compaction analysis, providing the deformed image matching the baseline image.
# Also plot the deformation as vector field.
da_new_image = compaction_analysis(new_base_c1, plot=True)

# Plot the differences between the two original images and after the transformation.
fig, ax = plt.subplots(1, num=1)
ax.imshow(skimage.util.compare_images(base_c2.img, new_base_c1.img, method="blend"))
fig, ax = plt.subplots(1, num=2)
ax.imshow(skimage.util.compare_images(base_c2.img, da_new_image.img, method="blend"))
plt.show()

# Apply transformation to labels
labels_c1 = np.load("labels_c1.npy")
labels = daria.Image(labels_c1, color_space="RGB", width=2.8, height=1.5)

translation_1, _ = translation_estimator.find_effective_translation(
    base_c1_img, base_c2_img, roi_cc
)
(h, w) = base_c2.img.shape[:2]
labels.img = cv2.warpAffine(labels.img, translation_1, (w, h))

compacted_labels = compaction_analysis.apply(labels)

np.save("labels_c2.npy", compacted_labels.img)
plt.figure()
plt.imshow(base_c1_img)
plt.imshow(labels_c1, alpha=0.3)
plt.figure()
plt.imshow(base_c2_img)
plt.imshow(compacted_labels.img, alpha=0.3)
plt.show()
