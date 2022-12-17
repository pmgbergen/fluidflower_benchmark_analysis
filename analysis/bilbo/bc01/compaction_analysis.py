"""
Determine compaction of FluidFlower by comparing two different images.

The images correpsond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, compaction/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
"""

"""
Analysis of FluidFlower Benchmark Run BC01.
"""
from benchmark.standardsetups.mediumco2analysis import MediumCO2Analysis
from benchmark.utils.misc import read_paths_from_user_data
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


# Read user-defined paths to images, number of baseline images, and config file
images, baseline, config, results = read_paths_from_user_data("user_data.json")

# Define FluidFlower based on a full set of basline images
analysis = MediumCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results = results, # path to results directory
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=True,  # print intermediate results to screen
)

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest. NOTE: These images are not part of the GH repo.
path_src = images[0]
path_dst = images[3]

analysis.load_and_process_image(path_src)
img_src = analysis.img.copy()

analysis.load_and_process_image(path_dst)
img_dst = analysis.img.copy()

# ! ----- Actual analysis: Determine the compaction between img_dst and aligned_img_src

# Define compaction analysis tool
config_compaction = {
    # Define the number of patches in x and y directions
    "N_patches": [20, 10],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
compaction_analysis = darsia.CompactionAnalysis(img_src, **config_compaction)

# Apply compaction analysis, providing the deformed image matching the baseline image,
# as well as the required translations on each patch, characterizing the total
# deformation. Also plot the deformation as vector field.
new_image, patch_translation = compaction_analysis(
    img_dst, plot_patch_translation=True, return_patch_translation=True
)

print("The centers of the 20 x 10 patches are at most by translated:")
print(np.max(np.absolute(patch_translation[:,:,1])))

# Plot the differences between the two original images and after the transformation.
fig, ax = plt.subplots(1, num=1)
ax.imshow(skimage.util.compare_images(img_src.img, img_dst.img, method="blend"))
fig, ax = plt.subplots(1, num=2)
ax.imshow(skimage.util.compare_images(img_src.img, new_image.img, method="blend"))
plt.show()

# Do image analysis for corrected image.
analysis.single_image_analysis(new_image, plot_contours=True)
