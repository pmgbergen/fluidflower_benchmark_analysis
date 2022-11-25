import cv2
import darsia as da
import numpy as np

# Create segmentation object (It is necessary to provide the number of segmentations that you want to compare)
# If no colors are provided, default colors will be chosen. NOTE: Number of colors must match the number of segmentated images.
segmentationComparison = da.SegmentationComparison(
    number_of_segmented_images=2,
    component_names=["CO2(w)", "CO2(g)"],
    segmentation_names=["BC01", "Lluis"],
    colors=np.array([[160, 0, 0], [150, 20, 200]]),
)


# Get segmentation path I do not know which time t3 corresonds to in Lluis images, 
# so Im just taking a random one. Please choose the correct one before running script.
segmentation_path = "C:/Users/Erlend/src/darsia/images/bc01_segmentation_malin/220507_time212200_DSC00745_segmentation.npy"

# Read segmentation
segmentation1 = np.load(segmentation_path)

"""
read lluis images and create segmentation
"""

# Paths to images
path_c = "C:/Users/Erlend/src/darsia/images/lluis_segmentations/C/C_t3.png"
path_s = "C:/Users/Erlend/src/darsia/images/lluis_segmentations/S/Sg_t3.png"

# ROI where the actual images are
ROI_im = (slice(276,1536),slice(371,2586))
# Tolerance used for the segmentation (this can definitely be tuned if you like)
tol = 64

# Read images from Lluis in grayscale
c_im = cv2.imread(path_c,0)
s_im = cv2.imread(path_s,0)

# Create segmentation array
segmentation_lluis = np.zeros_like(c_im[ROI_im])

# Fill segmentation array depending on information from Lluis' images
segmentation_lluis[c_im[ROI_im]>tol] = 1
segmentation_lluis[s_im[ROI_im]>tol] = 2


# Resize segmentation_1 to match the size of the segmentation from Lluis
segmentation1 = cv2.resize(segmentation1, (segmentation_lluis.shape[1], segmentation_lluis.shape[0]), interpolation= cv2.INTER_NEAREST)

# Create the comparison array (Here as many segmentations as desirable can be provided)
comparison = segmentationComparison(segmentation1, segmentation_lluis)

# Read baseline image for overlay plot
base = cv2.imread("C:/Users/Erlend/src/darsia/images/bc02_corrected_baseline.JPG")
base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

# Plot overlay image
segmentationComparison.plot_overlay_segmentation(
    comparison_image=comparison, base_image=base, opacity=0.6, legend_anchor=(0.6, 1),
)
