"""
Determine compaction of FluidFlower by comparing two different images.

Reference image is taken just after filling the rig with sand.
Test image is just before C1.

"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
from pathlib import Path

from benchmark.rigs.largefluidflower import LargeFluidFlower
import darsia
import cv2
import copy

# ! ----- Preliminaries - prepare two images for compaction analysis

# Image after sand filling
path_zero = Path("blackened/zero.png")
fluidflower_zero = LargeFluidFlower(path_zero, "./config_compaction_zero.json", False)
labels_zero = fluidflower_zero.labels.copy()
img_zero = fluidflower_zero.base

# # Some later image
# path_src = Path("blackened/src.png")
# fluidflower_src = LargeFluidFlower(path_src, "./config_compaction_src.json", False)
# labels_src = fluidflower_src.labels.copy()
# img_src = fluidflower_src.base

# Image before running C1
path_dst = Path("blackened/dst.png")
fluidflower_dst = LargeFluidFlower(path_dst, "./config_compaction_dst.json", False)
labels_dst = fluidflower_dst.labels.copy()
img_dst = fluidflower_dst.base

print(
    "make sure that the segmentation is for the src which should be deformed. or ist it dst which is deformed? not required before analyzing the deformation for each sandlayer"
)

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

# Fix a reference image
img_ref = img_zero.copy()

# Preliminaries: Define base config for compaction analysis
base_config_compaction = {
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

# One iteration of multiscale compaction analysis
def iteration(img_dst, img, patches, plot = False) -> tuple[darsia.Image, darsia.ReversedCompactionAnalysis]:
    # Define compaction analysis tool
    config_compaction = copy.deepcopy(base_config_compaction)
    config_compaction["N_patches"] = patches
    compaction_analysis = darsia.ReversedCompactionAnalysis(img_dst, **config_compaction)
    
    transformed_img, patch_translation = compaction_analysis(
        img, plot_patch_translation=plot, return_patch_translation=True
    )
   
    if plot:
        plt.figure("comparison")
        plt.imshow(skimage.util.compare_images(img_dst.img, transformed_img.img, method="blend"))
        plt.show()

    return transformed_img, compaction_analysis

# ! ---- 0. Iteration
print("0. iteration")
plt.figure("Initial comparison")
plt.imshow(skimage.util.compare_images(img_dst.img, img_ref.img, method="blend"))
plt.show()

# ! ---- Multiscale analysis
print("1. iteration")
img_ref_1, compaction_1 = iteration(img_dst, img_ref, [4,2])

print("2. iteration")
img_ref_2, compaction_2 = iteration(img_dst, img_ref_1, [8, 4])

print("3. iteration")
img_ref_3, compaction_3 = iteration(img_dst, img_ref_2, [16, 8])

print("4. iteration")
img_ref_4, compaction_4 = iteration(img_dst, img_ref_3, [32, 16])

print("5. iteration")
img_ref_5, compaction_5 = iteration(img_dst, img_ref_4, [64, 32])

# ! ---- 3. Post analysis

# Sum up all compaction analysis
print("post analysis")
config_compaction = copy.deepcopy(base_config_compaction)
config_compaction["N_patches"] = [64,32]
compaction_analysis = darsia.ReversedCompactionAnalysis(img_ref, **config_compaction)
compaction_analysis.deduct(compaction_1)
compaction_analysis.plot()
compaction_analysis.add(compaction_2)
compaction_analysis.plot()
compaction_analysis.add(compaction_3)
compaction_analysis.plot()
compaction_analysis.add(compaction_4)
compaction_analysis.plot()
compaction_analysis.add(compaction_5)
compaction_analysis.plot()

print("finished analysis")

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
    "compaction_corrected/ref_reverse.jpg",
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
