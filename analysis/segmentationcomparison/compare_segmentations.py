import cv2
import darsia as da
import numpy as np

# Create segmentation object (It is necessary to provide the number of segmentations that you want to compare)
# If no colors are provided, default colors will be chosen. NOTE: Number of colors must match the number of segmentated images.
segmentationComparison = da.SegmentationComparison(
    number_of_segmented_images=2,
    component_names=["CO2(w)", "CO2(g)"],
    segmentation_names=["BC01", "BC02"],
    colors=np.array([[160, 0, 0], [150, 20, 200]]),
)


# Get image paths
segmentation_path_1 = "C:/Users/Erlend/src/darsia/images/bc01_segment/220507_time194700_DSC00555_segmentation.npy"
segmentation_path_2 = "C:/Users/Erlend/src/darsia/images/bc02_segment/220523_time155600_DSC01433_segmentation.npy"

# Read segmentations
segmentation1 = np.load(segmentation_path_1)
segmentation2 = np.load(segmentation_path_2)

# Create the comparison array (Here as many segmentations as desirable can be provided)
comparison = segmentationComparison(segmentation1, segmentation2)


# Read baseline image for overlay plot
base = cv2.imread("C:/Users/Erlend/src/darsia/images/bc02_corrected_baseline.JPG")
base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

# Plot overlay image
segmentationComparison.plot_overlay_segmentation(
    comparison_image=comparison, base_image=base, opacity=0.6, legend_anchor=(0.6, 1)
)
