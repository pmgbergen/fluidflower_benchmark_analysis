import darsia as da
import numpy as np
import seaborn as sns


def zones(segmentation_path, depth_map, subregion):
    # Create segmentation object (It is necessary to provide the number of segmentations that you want to compare)
    # If no colors are provided, default colors will be chosen. NOTE: Number of colors must match the number of segmentated images.
    segmentationComparison = da.SegmentationComparison()

    # Create the comparison array (Here as many segmentations as desirable can be provided)
    comparison = segmentationComparison.compare_segmentations_binary_array(
        da.extractROI(
            da.Image(
                np.load(segmentation_path[0]), width=2.8, height=1.5, color_space="GRAY"
            ),
            subregion,
        ),
        da.extractROI(
            da.Image(
                np.load(segmentation_path[1]), width=2.8, height=1.5, color_space="GRAY"
            ),
            subregion,
        ),
        da.extractROI(
            da.Image(
                np.load(segmentation_path[2]), width=2.8, height=1.5, color_space="GRAY"
            ),
            subregion,
        ),
        da.extractROI(
            da.Image(
                np.load(segmentation_path[3]), width=2.8, height=1.5, color_space="GRAY"
            ),
            subregion,
        ),
        da.extractROI(
            da.Image(
                np.load(segmentation_path[4]), width=2.8, height=1.5, color_space="GRAY"
            ),
            subregion,
        ),
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
        depth_map=depth_map.img,
        width=depth_map.width,
        height=depth_map.height,
    )

    try:
        return True, weighted_colors, total_color
    except:
        False, None

