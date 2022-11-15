"""
Module containing the standardized CO2 analysis applicable for C1, ..., C5.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.utils.misc import read_time_from_path

# TODO major cleanup required!

## Define specific concentration analysis class to detect mobile CO2
# class CO2MaskAnalysis_layered(darsia.LayeredBinaryConcentrationAnalysis):
#    """
#    Binary concentration analysis based on a multichromatic HSV comparison
#    and further analysis on the third component, i.e., V, identifying signal
#    strength.
#    """
#
#    def __init__(self, base, color, labels, labels_legend, **kwargs) -> None:
#        super().__init__(base, color, labels, **kwargs)
#
#        # Fetch parameters for HUE based thresholding
#        self.hue_low_threshold = {}
#        self.hue_high_threshold = {}
#        self.hue_low_threshold["esf_sand"] = kwargs.pop("threshold min hue esf", 0.0)
#        self.hue_high_threshold["esf_sand"] = kwargs.pop("threshold max hue esf", 360)
#
#        self.hue_low_threshold["c_sand"] = kwargs.pop("threshold min hue c", 0.0)
#        self.hue_high_threshold["c_sand"] = kwargs.pop("threshold max hue c", 360)
#
#        self.hue_low_threshold["rest"] = kwargs.pop("threshold min hue rest", 0.0)
#        self.hue_high_threshold["rest"] = kwargs.pop("threshold max hue rest", 360)
#
#        # Heterogeneous thresholding
#        self.threshold_value = np.zeros(self.base.img.shape[:2], dtype=float)
#        # Water - discard
#        water = labels == labels_legend["water"]
#        self.threshold_value[water] = 1.0
#        # ESF-specific thresholding
#        esf_sand = labels == labels_legend["esf_sand"]
#        self.threshold_value[esf_sand] = kwargs.pop("threshold value esf")
#        # C-specific thresholding
#        c_sand = labels == labels_legend["c_sand"]
#        self.threshold_value[c_sand] = kwargs.pop("threshold value c")
#        # The rest
#        rest = labels == labels_legend["rest"]
#        self.threshold_value[rest] = kwargs.pop("threshold value rest")
#
#        # Store input
#        self.labels = labels
#        self.labels_legend = labels_legend
#
#    def _extract_scalar_information(self, img: darsia.Image) -> None:
#        """Transform to HSV.
#
#        Args:
#            img (darsia.Image): Input image which shall be modified.
#        """
#        img.img = cv2.cvtColor(img.img.astype(np.float32), cv2.COLOR_RGB2HSV)
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        """Return 3rd component of HSV (value), identifying signal strength.
#
#        Args:
#            img (np.ndarray): trichromatic image in HSV color space.
#
#        Returns:
#            np.ndarray: monochromatic image
#        """
#
#        # Clip values in hue - from calibration.
#        h_img = img[:, :, 0]
#        v_img = img[:, :, 2]
#
#        if self.verbosity:
#            plt.figure("H component")
#            plt.imshow(h_img)
#            plt.show()
#
#        # Investigate OTSU on each layer - aside of water
#        esf_sand = self.labels == self.labels_legend["esf_sand"]
#        c_sand = self.labels == self.labels_legend["c_sand"]
#        rest = self.labels == self.labels_legend["rest"]
#
#        # esf_values = np.ravel(h_img)[np.ravel(esf_sand)]
#        # c_values = np.ravel(h_img)[np.ravel(c_sand)]
#        # rest_values = np.ravel(h_img)[np.ravel(rest)]
#
#        ## Find automatic threshold value using OTSU
#        # esf_thresh = skimage.filters.threshold_otsu(esf_values)
#        # c_thresh = skimage.filters.threshold_otsu(c_values)
#        # rest_thresh = skimage.filters.threshold_otsu(rest_values)
#
#        # print("OTSU threshold h esf value", esf_thresh)
#        # print("OTSU threshold h c value", c_thresh)
#        # print("OTSU threshold h rest value", rest_thresh)
#
#        # esf_values = np.ravel(v_img)[np.ravel(esf_sand)]
#        # c_values = np.ravel(v_img)[np.ravel(c_sand)]
#        # rest_values = np.ravel(v_img)[np.ravel(rest)]
#
#        ## Find automatic threshold value using OTSU
#        # esf_thresh = skimage.filters.threshold_otsu(esf_values)
#        # c_thresh = skimage.filters.threshold_otsu(c_values)
#        # rest_thresh = skimage.filters.threshold_otsu(rest_values)
#
#        # print("OTSU threshold v esf value", esf_thresh)
#        # print("OTSU threshold v c value", c_thresh)
#        # print("OTSU threshold v rest value", rest_thresh)
#
#        # Go through all layers and restrict to the considered values
#
#        # Water - discard
#        # TODO
#
#        # ESF sand
#        threshold_mask_esf_sand = skimage.filters.apply_hysteresis_threshold(
#            h_img,
#            self.hue_low_threshold["esf_sand"],
#            self.hue_high_threshold["esf_sand"],
#        )
#        img[np.logical_and(esf_sand, ~threshold_mask_esf_sand)] = 0
#
#        # C sand
#        threshold_mask_c_sand = skimage.filters.apply_hysteresis_threshold(
#            h_img, self.hue_low_threshold["c_sand"], self.hue_high_threshold["c_sand"]
#        )
#        img[np.logical_and(c_sand, ~threshold_mask_c_sand)] = 0
#
#        # Rest
#        threshold_mask_rest = skimage.filters.apply_hysteresis_threshold(
#            h_img, self.hue_low_threshold["rest"], self.hue_high_threshold["rest"]
#        )
#        img[np.logical_and(rest, ~threshold_mask_rest)] = 0
#
#        plt.figure("thresholded h image")
#        plt.imshow(img[:, :, 0])
#        plt.figure("thresholded v image")
#        plt.imshow(img[:, :, 2])
#        plt.show()
#
#        # Consider Value (3rd component from HSV) to detect signal strength.
#        return img[:, :, 2]


## Define specific concentration analysis class to detect mobile CO2
# class CO2MaskAnalysis_old(darsia.BinaryConcentrationAnalysis):
#    """
#    Binary concentration analysis based on a multichromatic HSV comparison
#    and further analysis on the third component, i.e., V, identifying signal
#    strength.
#    """
#
#    def __init__(self, base, color, esf, **kwargs) -> None:
#        super().__init__(base, color, **kwargs)
#
#        # Heterogeneous thresholding
#        self.threshold_value = np.zeros(self.base.img.shape[:2], dtype=float)
#        self.threshold_value[esf] = kwargs.pop("threshold value esf")
#        self.threshold_value[~esf] = kwargs.pop("threshold value non-esf")
#
#        # Fetch parameters for HUE based thresholding
#        self.hue_low_threshold = kwargs.pop("threshold min hue", 0.0)
#        self.hue_high_threshold = kwargs.pop("threshold max hue", 360)
#
#    def _extract_scalar_information(self, img: darsia.Image) -> None:
#        """Transform to HSV.
#
#        Args:
#            img (darsia.Image): Input image which shall be modified.
#        """
#        img.img = cv2.cvtColor(img.img.astype(np.float32), cv2.COLOR_RGB2HSV)
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        """Return 3rd component of HSV (value), identifying signal strength.
#
#        Args:
#            img (np.ndarray): trichromatic image in HSV color space.
#
#        Returns:
#            np.ndarray: monochromatic image
#        """
#
#        # Clip values in hue - from calibration.
#        h_img = img[:, :, 0]
#
#        mask = skimage.filters.apply_hysteresis_threshold(
#            h_img, self.hue_low_threshold, self.hue_high_threshold
#        )
#
#        # Restrict to co2 mask
#        img[~mask] = 0
#
#        # Consider Value (3rd component from HSV) to detect signal strength.
#        return img[:, :, 2]
#
#
# class CO2MaskAnalysis(darsia.BinaryConcentrationAnalysis):
#    """
#    Binary concentration analysis based on a multichromatic HSV comparison
#    and further analysis on the third component, i.e., V, identifying signal
#    strength.
#    """
#
#    def __init__(self, base, color, **kwargs) -> None:
#        super().__init__(base, color, **kwargs)
#
#        # Fetch parameters for HUE based thresholding
#        self.hue_low_threshold = kwargs.pop("threshold min hue", 0.0)
#        self.hue_high_threshold = kwargs.pop("threshold max hue", 360)
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        """Return 3rd component of HSV (value), identifying signal strength.
#
#        Args:
#            img (np.ndarray): trichromatic image in HSV color space.
#
#        Returns:
#            np.ndarray: monochromatic image
#        """
#
#        mask = skimage.filters.apply_hysteresis_threshold(
#            img, self.hue_low_threshold, self.hue_high_threshold
#        )
#
#        return skimage.img_as_float(mask)


class BenchmarkCO2Analysis(LargeFluidFlower, darsia.CO2Analysis):
    """
    Class for managing the FluidFlower benchmark analysis.
    Identifiers for CO2 dissolved in water and CO2(g) are
    tailored to the benchmark set of the large rig.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = False,
    ) -> None:
        """
        Constructor for Benchmark rig.

        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the
                post-analysis are printed to screen; default is False.
        """
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        darsia.CO2Analysis.__init__(self, baseline, config, update_setup)

        # The above constructors provide access to the config via self.config.
        # Determine the injection start from the config file. Expect format
        # complying with "%y%m%d %H%M%D", e.g., "211127 083412"
        self.injection_start: datetime = datetime.strptime(
            self.config["injection_start"], "%y%m%d %H%M%S"
        )

        # Add possibility to apply compaction correction for each image
        if "compaction" in self.config.keys():
            self.apply_compaction_analysis = self.config["compaction"].get(
                "apply", False
            )
            if self.apply_compaction_analysis:
                self.compaction_analysis = darsia.CompactionAnalysis(
                    self.base, **self.config["compaction"]
                )
        else:
            self.apply_compaction_analysis = False

        # Initialize results dictionary for post-analysis
        self.results: dict = {}

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)

        # Store verbosity
        self.verbosity = verbosity

    # ! ---- Setup tools
    def load_and_process_image(self, path: Union[str, Path]) -> None:
        """
        Load image as before and read time from the title in addition.

        Args:
            path (str or Path): path to image
        """
        # Read image via parent class
        darsia.CO2Analysis.load_and_process_image(self, path)

        # Ammend image with time, reading from title assuming title
        # of format */yyMMdd_timeHHmmss*.
        self.img.timestamp = read_time_from_path(path)

    # ! ---- Analysis tools for detecting the different CO2 phases

    def define_co2_analysis(self) -> darsia.BinaryConcentrationAnalysis:
        """
        Identify CO2 using a heterogeneous HSV thresholding scheme.
        """
        if self.config["co2"].get("segmented", False):
            co2_analysis = darsia.SegmentedBinaryConcentrationAnalysis(
                self.base, self.labels, **self.config["co2"]
            )
        else:
            co2_analysis = darsia.BinaryConcentrationAnalysis(
                self.base, **self.config["co2"]
            )

        return co2_analysis

    def define_co2_gas_analysis(self) -> darsia.BinaryConcentrationAnalysis:
        """
        Identify CO2(g) using a thresholding scheme on the blue color channel,
        controlled from external config file.
        """
        if self.config["co2(g)"].get("segmented", False):
            co2_gas_analysis = darsia.SegmentedBinaryConcentrationAnalysis(
                self.base, self.labels, **self.config["co2(g)"]
            )
        else:
            co2_gas_analysis = darsia.BinaryConcentrationAnalysis(
                self.base, **self.config["co2(g)"]
            )

        return co2_gas_analysis

    # ! ----- Analysis tools

    def determine_co2_mask(self) -> darsia.Image:
        """Determine CO2.

        Returns:
            darsia.Image: boolean image detecting CO2.
        """
        # Extract co2 from analysis
        co2 = super().determine_co2()

        # Add expert knowledge. Turn of any signal in the water zone
        co2.img[self.water] = 0

        return co2

    def determine_co2_gas_mask(self, co2: darsia.Image) -> darsia.Image:
        """Determine CO2.

        Args:
            co2 (darsia.Image): boolean image detecting all co2.

        Returns:
            darsia.Image: boolean image detecting CO2(g).
        """
        # Extract co2 from analysis
        co2_gas = super().determine_co2_gas()

        # Add expert knowledge. Turn of any signal outside the presence of co2.
        # And turn off any signal in the ESF layer.
        co2_gas.img[~co2.img] = 0
        co2_gas.img[self.esf_sand] = 0

        # Remove small objects which are created through adding expert knowledge.
        # TODO include this in here? as adding epxert knowledge is very specific
        # for this analysis, and not the general concentrationanalysis.
        co2_gas.img = self.co2_gas_analysis.clean_mask(co2_gas.img)

        return co2_gas

    def single_image_analysis(
        self, img: Union[Path, darsia.Image], **kwargs
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Standard workflow to analyze CO2 phases.

        Args:
            image (Path or Image): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: co2 mask
            np.ndarray: co2(g) mask
            dict: dictinary with all stored results from the post-analysis.
        """

        # Load the current image
        if isinstance(img, darsia.Image):
            self.img = img.copy()
        else:
            self.load_and_process_image(img)

        # Perform compaction analysis
        if self.apply_compaction_analysis:
            # Apply compaction analysis, providing the deformed image matching the baseline image,
            # as well as the required translations on each patch, characterizing the total
            # deformation. Also plot the deformation as vector field.
            self.img = self.compaction_analysis(self.img)

        # Determine binary mask detecting any(!) CO2
        co2 = self.determine_co2_mask()

        # Determine binary mask detecting mobile CO2.
        co2_gas = self.determine_co2_gas_mask(co2)

        # ! ---- Post-analysis

        # Define some general data first:
        # Crop folder and ending from path - required for writing to file.
        img_id = Path(img.name).with_suffix("")

        # Determine the time increment (in terms of hours),
        # referring to injection start, in hours.
        SECONDS_TO_HOURS = 1.0 / 3600
        relative_time = (
            self.img.timestamp - self.injection_start
        ).total_seconds() * SECONDS_TO_HOURS

        # Fingering analysis in Box C
        fingering_analysis_box_C = kwargs.pop("fingering_analysis_box_C", False)
        if fingering_analysis_box_C:

            # Fingers in box C are defines as contour of the CO2. To evaluate the length
            # of the fingers, determine the length of the internal contour of the CO2 mask.
            # Apply Venn diagram concept to exclude the boundary of box C. For this consider
            # the CO2, the water, and the entire box, and take the correct difference.

            contour_length_co2 = darsia.contour_length(co2, roi=self.box_C)
            contour_length_water = darsia.contour_length(
                co2, roi=self.box_C, values_of_interest=[False]
            )
            boundary_box_C = darsia.perimeter(self.box_C)

            # Venn diagram concept to determine the internal contour of the CO2.
            # Cut values at 0, slightly negative values can occur e.g., when
            # contour_length_co2 is very small due to a different evaluation
            # routine for the contour length and the permiter of a box.
            length_finger = max(
                0.5 * (contour_length_co2 + contour_length_water - boundary_box_C), 0.0
            )

            # Keep track of result and store internally
            if "fingering_analysis_box_C" not in self.results:
                self.results["fingering_analysis_box_C"] = []
            self.results["fingering_analysis_box_C"].append(
                [relative_time, length_finger]
            )

            if self.verbosity:
                print(
                    f"Total finger length in Box C at {relative_time} hours: {length_finger} m."
                )
                print(contour_length_co2, contour_length_water, boundary_box_C)

        # Plot and store image with contours
        plot_contours = kwargs.pop("plot_contours", False)
        write_contours_to_file = kwargs.pop("write_contours_to_file", False)

        if plot_contours or write_contours_to_file:

            # Start with the original image
            original_img = np.copy(self.img.img)
            original_img = skimage.img_as_ubyte(original_img)

            # Overlay the original image with contours for CO2
            contours_co2, _ = cv2.findContours(
                skimage.img_as_ubyte(co2.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(original_img, contours_co2, -1, (0, 255, 0), 3)

            # Overlay the original image with contours for CO2(g)
            contours_co2_gas, _ = cv2.findContours(
                skimage.img_as_ubyte(co2_gas.img),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(original_img, contours_co2_gas, -1, (255, 255, 0), 3)

            # Overlay the original image with contours of Box A
            contours_box_A, _ = cv2.findContours(
                skimage.img_as_ubyte(self.mask_box_A),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(original_img, contours_box_A, -1, (180, 180, 180), 3)

            # Overlay the original image with contours of Box B
            contours_box_B, _ = cv2.findContours(
                skimage.img_as_ubyte(self.mask_box_B),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(original_img, contours_box_B, -1, (180, 180, 180), 3)

            # Overlay the original image with contours of Box C
            contours_box_C, _ = cv2.findContours(
                skimage.img_as_ubyte(self.mask_box_C),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(original_img, contours_box_C, -1, (180, 180, 180), 3)

            # Plot
            if plot_contours:
                plt.figure("Image with contours of CO2 segmentation")
                plt.imshow(original_img)
                plt.show()

            # Write corrected image with contours to file
            if write_contours_to_file:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(self.path_to_results / Path(f"{img_id}_with_contours.jpg")),
                    original_img,
                )

        # Write segmentation to file
        write_segmentation_to_file = kwargs.pop("write_segmentation_to_file", False)
        write_coarse_segmentation_to_file = kwargs.pop(
            "write_coarse_segmentation_to_file", False
        )

        if write_segmentation_to_file or write_coarse_segmentation_to_file:

            # Generate segmentation with codes:
            # 0 - water
            # 1 - dissolved CO2
            # 2 - gaseous CO2
            segmentation = np.zeros(self.img.img.shape[:2], dtype=int)
            segmentation[co2.img] += 1
            segmentation[co2_gas.img] += 1

            # Store fine scale segmentation
            if write_segmentation_to_file:
                np.save(
                    self.path_to_results / Path(f"{img_id}_segmentation.npy"),
                    segmentation,
                )

            # Store coarse scale segmentation
            if write_coarse_segmentation_to_file:
                coarse_shape = (150, 280)
                coarse_segmentation = np.zeros(coarse_shape, dtype=int)
                co2_coarse = skimage.img_as_bool(
                    skimage.transform.resize(co2.img, coarse_shape)
                )
                co2_gas_coarse = skimage.img_as_bool(
                    skimage.transform.resize(co2_gas.img, coarse_shape)
                )
                coarse_segmentation[co2_coarse] += 1
                coarse_segmentation[co2_gas_coarse] += 1
                np.save(
                    self.path_to_results / Path(f"{img_id}_coarse_segmentation.npy"),
                    coarse_segmentation,
                )

        return co2, co2_gas, self.results

    def batch_analysis(self, images: list[Path], **kwargs) -> dict:
        """
        Standard batch analysis for C1, ..., C5.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments:
                plot_contours (bool): flag controlling whether the original image
                    is plotted with contours of the two CO2 phases; default False.
                write_contours_to_file (bool): flag controlling whether the plot from
                    plot_contours is written to file; default False.
                fingering_analysis_box_C (bool): flag controlling whether the length
                    of the finger is performed in Box C; default is False.
                write_segmentation_to_file (bool): flag controlling whether the
                    CO2 segmentation is written to file, where water, dissolved CO2
                    and CO2(g) get decoded 0, 1, 2, respectively; default False.
                write_coarse_segmentation_to_file (bool): flag controlling whether
                    a coarse (280 x 150) representation of the CO2 segmentation from
                    write_segmentation_to_file is written to file; default False.

        Returns:
            dict: dictinary with all stored results from the post-analysis.
        """

        for num, img in enumerate(images):

            tic = time.time()

            # Determine binary mask detecting any(!) CO2, and CO2(g), and apply post-analysis.
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")

        return self.results

    def return_results(self) -> dict:
        """
        Return all results collected throughout any analysis performed.
        """
        # TODO restrict to specific list of keywords.

        return self.results

    def write_results_to_file(self) -> None:
        """
        Write results in separate files.

        Args:
            folder (Path): folder where the results are stored.
        """

        # Write results of fingering analysis to file
        if "fingering_analysis_box_C" in self.results.keys():
            # Convert list to a numpy array
            finger_length_c = np.array(self.results["fingering_analysis_box_C"])

            # Store to file as npy file
            np.save(self.path_to_results / Path("finger_length_c.npy"), finger_length_c)

            # Store to file as cvs file
            np.savetxt(
                self.path_to_results / Path("finger_lenght_c.csv"),
                finger_length_c,
                delimiter=",",
            )
