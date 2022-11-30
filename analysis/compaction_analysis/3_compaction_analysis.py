"""
Determine compaction of FluidFlower by comparing two different images.

The images correpsond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, compaction/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
from pathlib import Path

from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.standardsetups.benchmarkco2analysis import BenchmarkCO2Analysis
import darsia
import cv2

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two corrected images of interest - with black water.
path_zero = Path("blackened/zero.png")
path_src = Path("blackened/src.png")
path_dst = Path("blackened/dst.png")

# Utilize the co2 analysis class with the earlier image as baseline image
fluidflower_zero = LargeFluidFlower(path_zero, "./config_compaction_zero.json", False)
labels_zero = fluidflower_zero.labels.copy()
fluidflower_src = LargeFluidFlower(path_src, "./config_compaction_src.json", False)
labels_src = fluidflower_src.labels.copy()
fluidflower_dst = LargeFluidFlower(path_dst, "./config_compaction_dst.json", False)
labels_dst = fluidflower_dst.labels.copy()

print(
    "make sure that the segmentation is for the src which should be deformed. or ist it dst which is deformed? not required before analyzing the deformation for each sandlayer"
)

# Now have path_src and path_dst as darsia Images accesible via
# analysis.base and analysis.img respectively.
img_zero = fluidflower_zero.base
img_src = fluidflower_src.base
img_dst = fluidflower_dst.base

if False:
    plt.figure()
    plt.imshow(img_zero.img)
    plt.figure()
    plt.imshow(img_src.img)
    plt.figure()
    plt.imshow(img_dst.img)
    plt.figure()
    plt.imshow(labels_zero)
    plt.figure()
    plt.imshow(labels_src)
    plt.figure()
    plt.imshow(labels_dst)
    plt.show()

# ! ---- 1. Iteration
print("1. iteration")

# Define compaction analysis tool
config_compaction = {
    # Define the number of patches in x and y directions
    "N_patches": [10, 3],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

img_ref = img_zero.copy()

sub1_compaction_analysis = darsia.CompactionAnalysis(img_ref, **config_compaction)

plt.figure("0th iteration")
plt.imshow(skimage.util.compare_images(img_dst.img, img_ref.img, method="blend"))
plt.show()

# Apply compaction analysis, providing the deformed image matching the baseline image,
# as well as the required translations on each patch, characterizing the total
# deformation. Also plot the deformation as vector field.
new_img, patch_translation = sub1_compaction_analysis(
    img_dst, plot_patch_translation=True, return_patch_translation=True
)

plt.figure("1st iteration - reverse")
plt.imshow(skimage.util.compare_images(img_dst.img, new_img.img, method="blend"))
plt.show()

# ! ---- 2. Iteration
print("2. iteration")

# Define compaction analysis tool
config_compaction = {
    # Define the number of patches in x and y directions
    # "N_patches": [20, 10],
    "N_patches": [20, 10],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
sub2_compaction_analysis = darsia.CompactionAnalysis(img_ref, **config_compaction)

new_img_2, patch_translation = sub2_compaction_analysis(
    new_img, plot_patch_translation=True, return_patch_translation=True
)

plt.figure("2nd iteration - reverse")
plt.imshow(skimage.util.compare_images(new_img_2.img, new_img.img, method="blend"))
plt.show()

# ! ---- 3. Iteration
print("3. iteration")

# Define compaction analysis tool
config_compaction = {
    # Define the number of patches in x and y directions
    # "N_patches": [20, 10],
    "N_patches": [20, 10],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
sub3_compaction_analysis = darsia.CompactionAnalysis(img_ref, **config_compaction)

new_img_3, patch_translation = sub3_compaction_analysis(
    new_img_2, plot_patch_translation=True, return_patch_translation=True
)

plt.figure("3rd iteration - reverse")
plt.imshow(skimage.util.compare_images(new_img_3.img, new_img_2.img, method="blend"))
plt.show()

# ! ---- 4. Iteration
print("4. iteration")

# Define compaction analysis tool
config_compaction = {
    # Define the number of patches in x and y directions
    "N_patches": [30, 15],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
sub4_compaction_analysis = darsia.CompactionAnalysis(img_ref, **config_compaction)

new_img_4, patch_translation = sub4_compaction_analysis(
    new_img_3, plot_patch_translation=True, return_patch_translation=True
)

plt.figure("4th iteration - reverse")
plt.imshow(skimage.util.compare_images(new_img_4.img, new_img_3.img, method="blend"))
plt.show()

# ! ---- 3. Post analysis

# Sum up all compaction analysis
config_compaction = {
    # Define the number of patches in x and y directions
    "N_patches": [30, 15],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
compaction_analysis = darsia.CompactionAnalysis(img_ref, **config_compaction)
compaction_analysis.add(sub1_compaction_analysis)
compaction_analysis.plot()
compaction_analysis.add(sub2_compaction_analysis)
compaction_analysis.plot()
compaction_analysis.add(sub3_compaction_analysis)
compaction_analysis.plot()
compaction_analysis.add(sub4_compaction_analysis)
compaction_analysis.plot()

img_ref_deformed = compaction_analysis.apply(img_ref)

# Plot the differences between the two original images and after the transformation.
plt.figure("Comparison of deformed ref and dst")
plt.imshow(
    skimage.util.compare_images(img_dst.img, img_ref_deformed.img, method="blend")
)
plt.show()

# Store compaction corrected image
Path("compaction_corrected").mkdir(exist_ok=True)
cv2.imwrite(
    "compaction_corrected/ref.jpg",
    img_ref_deformed.img,
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

# Plot results / arrows.

# Divergence integrated over the domain
divergence = compaction_analysis.divergence()

plt.figure("divergence")
plt.imshow(divergence)
plt.show()

assert False

#
## It is also possible to evaluate the compaction approximation in arbitrary points.
## For instance, consider 4 points in metric coordinates (provided in x, y format):
# pts = np.array(
#    [
#        [0.2, 1.4],
#        [0.5, 0.5],
#        [0.5, 1.2],
#        [1.2, 0.75],
#        [2.3, 1.1],
#    ]
# )
# print("Consider the points:")
# print(pts)
#
# deformation = compaction_analysis.evaluate(pts)
# print("Deformation evaluated:")
# print(deformation)
#
## One can also use a patched ROI and evaluate the deformation in the patch centers.
## For this, we start extracting a roi, here we choose a box, which in the end
## corresponds to box B from the benchmark analysis. This it is sufficient to
## define two corner points of the box:
# box_B = np.array([[0.0, 1.2], [1.1, 0.6]])
#
## and extract the corresponding ROI as darsia.Image (based on da_img_src):
# img_box_B = darsia.extractROI(da_img_src, box_B)
#
## To double check the box, we plot the resulting box.
# plt.figure("Box B")
# plt.imshow(img_box_B.img)
# plt.show()
#
## Now we patch box B, the number of patches is arbitrary (here chosen to be 5 x 3):
# patches_box_B = darsia.Patches(img_box_B, 5, 3)
#
## The patch centers can be accessed:
# patch_centers_box_B = patches_box_B.global_centers_cartesian
#
## The deformation in the centers of these points can be obtained by evaluating the
## deformation map in a similar fashion as above. The results comes in a matrix.
## Entry [row,col] is associated to patch with coordinate [row,col] using the
## conventional matrix indexing.
# deformation_patch_centers_box_B = compaction_analysis.evaluate(patch_centers_box_B)
# print("The deformation in the centers of the patches of Box B:")
# print(deformation_patch_centers_box_B)
