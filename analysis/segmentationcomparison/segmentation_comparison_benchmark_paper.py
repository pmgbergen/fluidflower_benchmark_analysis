import cv2
import darsia as da
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib
import os
from datetime import datetime
from benchmark.utils.misc import read_time_from_path
import pandas as pd

matplotlib.use("Agg")


def whole_img(segmentation_path, depth_map, plot_name, plot=False):
    # Create segmentation object (It is necessary to provide the number of segmentations that you want to compare)
    # If no colors are provided, default colors will be chosen. NOTE: Number of colors must match the number of segmentated images.
    segmentationComparison = da.SegmentationComparison()

    # Create the comparison array (Here as many segmentations as desirable can be provided)
    comparison = segmentationComparison.compare_segmentations_binary_array(
        np.load(segmentation_path[0]),
        np.load(segmentation_path[1]),
        np.load(segmentation_path[2]),
        np.load(segmentation_path[3]),
        np.load(segmentation_path[4]),
    )

    # Create color palette
    sns_palette = np.array(sns.color_palette())
    gray = np.array([[0.3, 0.3, 0.3], [1, 1, 1]])
    palette = np.append(sns_palette[:6], gray, axis=0)

    # Image to be filled with colors depending on segmentation overlap
    colored_comparison = np.zeros(
        comparison.shape[:2] + (3,),
        dtype=np.uint8,
    )

    # List of unique combinations
    unique_combinations = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]

    # List of combinations including c2, c3, and c4 (the 1, 2, 3 is for the relative
    # position in the presence lists, i.e., we will get all combinations of the form [x,1,1,1,x],
    # it is of course only 4 different ones in this specific case, but this is a general function)
    comb_c2_c3_c4 = segmentationComparison.get_combinations(
        1, 2, 3, num_segmentations=5
    )

    # Get bolean array of true-values for all combinations of c2, c3, and c4.
    c_2_c3_c4_bool = np.zeros(comparison.shape[:2], dtype=bool)
    for combination in comb_c2_c3_c4:
        c_2_c3_c4_bool += np.all(comparison == combination, axis=2)

    # -------------------------------------#
    # Get bolean array of true-values for all combinations between two runs for c2-c3-c4.
    spes_bool = np.zeros(comparison.shape[:2], dtype=bool)

    comb_c2_c3 = segmentationComparison.get_combinations(1, 2, num_segmentations=5)
    for combination in comb_c2_c3:
        spes_bool += np.all(comparison == combination, axis=2)

    comb_c3_c4 = segmentationComparison.get_combinations(2, 3, num_segmentations=5)

    for combination in comb_c3_c4:
        spes_bool += np.all(comparison == combination, axis=2)

    comb_c2_c4 = segmentationComparison.get_combinations(1, 3, num_segmentations=5)

    for combination in comb_c2_c4:
        spes_bool += np.all(comparison == combination, axis=2)
    # -------------------------------------#

    # Fill the colored image. Start by coloring all pixels that have any segmentation present at all with the light gray color
    colored_comparison[np.any(comparison != [0, 0, 0, 0, 0], axis=2)] = (
        palette[7] * 255
    ).astype(np.uint8)

    # Fill the colored image with the spesial combinations
    colored_comparison[spes_bool] = (palette[5] * 255).astype(np.uint8)

    # Color the unique combinations
    for c, i in enumerate(unique_combinations):
        colored_comparison[np.all(comparison == i, axis=2)] = (palette[c] * 255).astype(
            np.uint8
        )

    # Color the combination where c2, c3 and c4 are included
    colored_comparison[c_2_c3_c4_bool] = (palette[6] * 255).astype(np.uint8)

    # NOTE: If several of these computations are to be done on images of the same size, save the depth_map and feed it to the color_fractions() method instead of depth_measurements. That will result in a HUGE time save.
    (
        weighted_colors,
        color_fractions,
        colors,
        total_color,
        depth_map,
    ) = segmentationComparison.color_fractions(
        colored_comparison,
        colors=(palette * 255).astype(np.uint8),
        depth_map=depth_map,
        width=2.8,
        height=1.5,
    )

    if plot:
        figure, axes = plt.subplots(figsize=(20, 10))
        # create legend paches
        labels = ["C1", "C2", "C3", "C4", "C5", "spesial_comb", "C2+C3+C4", "Other"]

        for i in range(len(labels)):
            labels[i] = labels[i] + " " + str(round(weighted_colors[i], 2))
        patch = []
        for i in range(8):
            patch.append(mpatches.Patch(color=palette[i], label=labels[i]))

        # Read baseline image for overlay plot
        base = cv2.imread(
            "E:/Git/fluidflower_benchmark_analysis/analysis/segmentationcomparison_new_method/211124_time082740_DSC00067.TIF"
        )
        # Resize image to same size as colored comparison
        base = cv2.resize(
            base, (colored_comparison.shape[1], colored_comparison.shape[0])
        )
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

        # Process the comparison image
        processed_comparison_image = segmentationComparison._post_process_image(
            colored_comparison, unique_colors=colors, opacity=0.6, contour_thickness=10
        )

        plt.imshow(base)
        plt.imshow(processed_comparison_image)
        plt.legend(handles=patch, bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.0)

        boxes = [
            np.array([[1.1, 0.6], [2.8, 0.0]]),  # boxA
            np.array([[0.0, 1.2], [1.1, 0.0]]),  # boxB*
            np.array([[1.1, 1.2], [2.8, 0.6]]),
        ]  # boxD

        colors = [[0.2, 0.8, 0.2], [1, 1, 1], [1, 1, 0]]
        # for c,i in enumerate(boxes):
        #     _, roi_box = da.extractROI(da.Image(base,width=2.8,height=1.5), i, return_roi=True)
        #     x0,y0 = roi_box[1].start,roi_box[0].start
        #     width,height = roi_box[1].stop-roi_box[1].start,roi_box[0].stop-roi_box[0].start
        #     rec = plt.Rectangle((x0,y0), width, height,fill = False,linewidth=1.5,ls = "--",color = colors[c])
        #     axes.add_artist(rec)
        plt.axis("off")
        gs = plt.tight_layout()
        plt.savefig(plot_name, dpi=100)

    try:
        return True, weighted_colors, total_color
    except:
        False, None


# Main script
Results_path = (
    "E:/Git/fluidflower_benchmark_analysis/analysis/Results/fine_segmentation/"
)

seg_folders = [Results_path + i + "/" for i in os.listdir(Results_path)]
segmentations_path = []


# remove segmentations from 68 to 71, because c5 missing those images/segmentations
for i in seg_folders:
    s = [i + seg for seg in os.listdir(i) if seg.endswith(".npy")]
    if len(s) > 123:
        s = s[:68] + s[72:]
    segmentations_path.append(s)

inj_start = [
    datetime(2021, 11, 24, 8, 31, 0),  # c1
    datetime(2021, 12, 4, 10, 1, 0),  # c2
    datetime(2021, 12, 14, 11, 20, 0),  # c3
    datetime(2021, 12, 24, 9, 0, 0),  # c4
    datetime(2022, 1, 4, 11, 0, 0),  # c5
]

time = []
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
spesial_comb = []
c2_c3_c4_overlap = []
other = []
total = []

meas_dir = "E:/Git/fluidflower_benchmark_analysis/analysis/depths/"

depth_measurements = (
    np.load(meas_dir + "x_measures.npy"),
    np.load(meas_dir + "y_measures.npy"),
    np.load(meas_dir + "depth_measures.npy"),
)

seg_da = da.Image(np.load(segmentations_path[0][0]), width=2.8, height=1.5)

depth_map = da.compute_depth_map(seg_da, depth_measurements=depth_measurements)

for i in range(46, len(segmentations_path[0])):
    timestamp1 = read_time_from_path(segmentations_path[0][i])
    timestamp2 = read_time_from_path(segmentations_path[1][i])
    timestamp3 = read_time_from_path(segmentations_path[2][i])
    timestamp4 = read_time_from_path(segmentations_path[3][i])
    timestamp5 = read_time_from_path(segmentations_path[4][i])
    t1 = (timestamp1 - inj_start[0]).total_seconds() / 60  # minutes
    t2 = (timestamp2 - inj_start[1]).total_seconds() / 60  # minutes
    t3 = (timestamp3 - inj_start[2]).total_seconds() / 60  # minutes
    t4 = (timestamp4 - inj_start[3]).total_seconds() / 60  # minutes
    t5 = (timestamp5 - inj_start[4]).total_seconds() / 60  # minutes
    plot_name = (
        "comp_images_weighted/"
        + str(round(t1, 1))
        + "_"
        + str(round(t2, 1))
        + "_"
        + str(round(t3, 1))
        + "_"
        + str(round(t4, 1))
        + "_"
        + str(round(t5, 1))
        + "_min.png"
    )

    ans, values, tot = whole_img(
        [segmentations_path[j][i] for j in range(5)],
        depth_map,
        plot_name,
        plot=True,
    )

    if ans:
        c1.append(values[0])
        c2.append(values[1])
        c3.append(values[2])
        c4.append(values[3])
        c5.append(values[4])
        spesial_comb.append(values[5])
        c2_c3_c4_overlap.append(values[6])
        other.append(values[7])
        total.append(tot)
        time.append(t1)
        print(sum(values))
    else:
        None

df = pd.DataFrame()

df["Time [min]"] = time
df["C1"] = c1
df["C2"] = c2
df["C3"] = c3
df["C4"] = c4
df["C5"] = c5
df["spesial_comb"] = spesial_comb
df["c2_c3_c4_overlap"] = c2_c3_c4_overlap
df["other"] = other
df["total"] = total

df.to_excel("fine_segmentation_whole_FL_weighted_colors.xlsx", index=False)