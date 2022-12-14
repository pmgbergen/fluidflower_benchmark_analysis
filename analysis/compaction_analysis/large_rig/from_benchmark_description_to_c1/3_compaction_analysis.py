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

# Image of benchmark description
img_src_arr = np.load("corrected/src.npy")
mask_src_arr = np.load("blackened/src_mask.npy")

# Image before running C1
img_dst_arr = np.load("corrected/dst.npy")
mask_dst_arr = np.load("blackened/dst_mask.npy")

# Darsia Image variants
img_src = darsia.Image(img_src_arr, width=2.8, height=1.5)
img_dst = darsia.Image(img_dst_arr, width=2.8, height=1.5)
mask_dst = darsia.Image(np.logical_not(mask_dst_arr), width=2.8, height=1.5)
mask_src = darsia.Image(np.logical_not(mask_src_arr), width=2.8, height=1.5)

# Verbosity
if True:
    plt.figure("src")
    plt.imshow(img_src.img)
    plt.figure("dst")
    plt.imshow(img_dst.img)
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

# Total compaction - initialize multiscale approach
config_compaction = copy.deepcopy(base_config_compaction)
config_compaction["N_patches"] = [64, 32]
compaction_analysis = darsia.ReversedCompactionAnalysis(img_src, **config_compaction)

# Initial approximation also of the mask
img_src_0 = img_src.copy()
mask_src_0 = mask_src.copy()

# One iteration of multiscale compaction analysis
def iteration(
    img_dst, img_src, mask_dst, mask_src, patches, plot=False
) -> tuple[darsia.Image, darsia.ReversedCompactionAnalysis]:
    # Define compaction analysis tool
    config_compaction = copy.deepcopy(base_config_compaction)
    config_compaction["N_patches"] = patches
    compaction_analysis = darsia.ReversedCompactionAnalysis(
        img_dst, mask=mask_dst, **config_compaction
    )

    transformed_img, patch_translation = compaction_analysis(
        img_src, plot_patch_translation=plot, return_patch_translation=True, mask=mask_src
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
    plt.imshow(skimage.util.compare_images(img_dst.img, img_src.img, method="blend"))
    plt.show()
Path("compaction_corrected").mkdir(exist_ok=True)
cv2.imwrite(
    "compaction_corrected/src_0.jpg",
    img_src.img,
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

# ! ---- Multiscale analysis
print("1. iteration")
_, compaction_1 = iteration(img_dst, img_src, mask_dst, mask_src_0, [4, 2])
mask_src_1 = compaction_1.apply(mask_src_0)
compaction_analysis.deduct(compaction_1)
img_src_1 = compaction_analysis.apply(img_src)

cv2.imwrite(
    "compaction_corrected/src_1.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_src_1.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("2. iteration")
_, compaction_2 = iteration(img_dst, img_src_1, mask_dst, mask_src_1, [8, 4])
mask_src_2 = compaction_2.apply(mask_src_1)
compaction_analysis.add(compaction_2)
img_src_2 = compaction_analysis.apply(img_src)

cv2.imwrite(
    "compaction_corrected/src_2.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_src_2.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("3. iteration")
_, compaction_3 = iteration(img_dst, img_src_2, mask_dst, mask_src_2, [16, 8])
mask_src_3 = compaction_3.apply(mask_src_2)
compaction_analysis.add(compaction_3)
img_src_3 = compaction_analysis.apply(img_src)

cv2.imwrite(
    "compaction_corrected/src_3.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_src_3.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("4. iteration")
_, compaction_4 = iteration(img_dst, img_src_3, mask_dst, mask_src_3, [32, 16])
mask_src_4 = compaction_3.apply(mask_src_3)
compaction_analysis.add(compaction_4)
img_src_4 = compaction_analysis.apply(img_src)

cv2.imwrite(
    "compaction_corrected/src_4.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_src_4.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

print("5. iteration")
_, compaction_5 = iteration(img_dst, img_src_4, mask_dst, mask_src_4, [64, 32])
mask_src_5 = compaction_5.apply(mask_src_4)
compaction_analysis.add(compaction_5)
img_src_5 = compaction_analysis.apply(img_src)

cv2.imwrite(
    "compaction_corrected/src_5.jpg",
    cv2.cvtColor(skimage.img_as_ubyte(np.clip(img_src_5.img, 0, 1)), cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)

if True:
    compaction_analysis.plot()

if False:
    plt.figure("0")
    plt.imshow(mask_src_0.img)
    plt.figure("1")
    plt.imshow(mask_src_1.img)
    plt.figure("2")
    plt.imshow(mask_src_2.img)
    plt.figure("3")
    plt.imshow(mask_src_3.img)
    plt.figure("4")
    plt.imshow(mask_src_4.img)
    plt.figure("5")
    plt.imshow(mask_src_5.img)
    plt.show()

# ! ---- 3. Post analysis

# Apply the total deformation
img_src_deformed = compaction_analysis.apply(img_src)

# Plot the differences between the two original images and after the transformation.
if True:
    plt.figure("Comparison of deformed src and dst")
    plt.imshow(
        skimage.util.compare_images(img_dst.img, img_src_deformed.img, method="blend")
    )

    print(img_src_5.img.shape, img_src_deformed.img.shape)

    plt.figure("Comparison of deformed src x 2")
    plt.imshow(
        skimage.util.compare_images(img_src_5.img, img_src_deformed.img, method="blend")
    )

    plt.show()

    # Store compaction corrected image
    cv2.imwrite(
        "compaction_corrected/src_reverse.jpg",
        cv2.cvtColor(
            skimage.img_as_ubyte(np.clip(img_src_deformed.img, 0, 1)), cv2.COLOR_RGB2BGR
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
    Ny, Nx = labels_src.img.shape[:2]
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
