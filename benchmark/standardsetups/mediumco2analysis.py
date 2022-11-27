"""
Module containing the standardized CO2 analysis applicable for the medium rig.
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
from benchmark.rigs.bilbo import Bilbo
from benchmark.utils.misc import array_to_csv, read_time_from_path


class MediumCO2Analysis(Bilbo, darsia.CO2Analysis):
    """
    Class for managing the FluidFlower benchmark analysis.
    Identifiers for CO2 dissolved in water and CO2(g) are
    tailored to the benchmark set of the medium rig.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = True,
    ) -> None:
        """
        Constructor for the analysis of the medium rig (Bilbo).

        Sets up fixed config file required for preprocessing.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the
                post-analysis are printed to screen; default is False.
        """
        Bilbo.__init__(self, baseline, config, update_setup)
        darsia.CO2Analysis.__init__(self, baseline, config, update_setup)

        # The above constructors provide access to the config via self.config.
        # Determine the injection start from the config file. Expect format
        # complying with "%y%m%d %H%M%D", e.g., "211127 083412"
        # self.injection_start: datetime = datetime.strptime(
        #    self.config["injection_start"], "%y%m%d %H%M%S"
        # )

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

    # ! ---- Segementation tools for detecting the different CO2 phases

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

    def determine_co2_mask(self) -> darsia.Image:
        """Determine CO2.

        Returns:
            darsia.Image: boolean image detecting CO2.
        """
        # Extract co2 from analysis
        co2 = super().determine_co2()

        # Add expert knowledge. Turn of any signal in the water zone.
        co2.img[self.extended_water] = 0

        return co2

    def determine_co2_gas_mask(self, co2: darsia.Image) -> darsia.Image:
        """Determine CO2.

        Args:
            co2 (darsia.Image): boolean image detecting all co2.

        Returns:
            darsia.Image: boolean image detecting CO2(g).
        """
        # Add expert knowledge - do not expect any CO2(g) outside
        # of the CO2, and neither in ESF nor C. Include this in
        # the analysis.
        expert_knowledge = np.logical_and(
            co2.img, np.logical_not(np.logical_or(self.esf_sand, self.c_sand))
        )
        # Also exclude CO2 from the bottom of the rig - mostly due to shaking images.
        expert_knowledge = np.logical_and(expert_knowledge, np.logical_not(self.bottom_zone))
        self.co2_gas_analysis.update_mask(expert_knowledge)

        # Extract co2 from analysis - restrict the analysis to areas with CO2.
        co2_gas = super().determine_co2_gas()

        # Add expert knowledge. Turn of any signal outside the presence of co2.
        # And turn off any signal in the ESF layer and C layer.
        co2_gas.img[~expert_knowledge] = 0

        # Clean the results once more after adding expert knowledge.
        co2_gas.img = self.co2_gas_analysis.clean_mask(co2_gas.img)

        return co2_gas

    def single_image_segmentation(
        self, img: Union[Path, darsia.Image], **kwargs
    ) -> None:
        """
        Standard workflow to analyze CO2 phases.

        Args:
            image (Path or Image): path to single image.
            kwargs: optional keyword arguments, see batch_segmentation.
        """
        # ! ----  Pre-processing

        # Load the current image
        if isinstance(img, darsia.Image):
            self.img = img.copy()
        else:
            self.load_and_process_image(img)

        # Compaction correction
        if self.apply_compaction_analysis:
            # Apply compaction analysis, providing the deformed image matching the baseline image,
            # as well as the required translations on each patch, characterizing the total
            # deformation. Also plot the deformation as vector field.
            self.img = self.compaction_analysis(self.img)

        # ! ----  Segmentation

        # Determine binary mask detecting any(!) CO2
        co2 = self.determine_co2_mask()

        # Determine binary mask detecting mobile CO2.
        co2_gas = self.determine_co2_gas_mask(co2)

        # ! ---- Storage of segmentation

        # Define some general data first:
        # Crop folder and ending from path - required for writing to file.
        img_id = Path(img.name).with_suffix("")

        # Plot and store image with contours
        plot_contours = kwargs.pop("plot_contours", False)
        write_contours_to_file = kwargs.pop("write_contours_to_file", True)

        if plot_contours or write_contours_to_file:

            # Start with the original image
            original_img = np.clip(np.copy(self.img.img), 0, 1)
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

            # Plot
            if plot_contours:
                plt.figure("Image with contours of CO2 segmentation")
                plt.imshow(original_img)
                plt.show()

            # Write corrected image with contours to file
            if write_contours_to_file:
                (self.path_to_results / Path("contour_plots")).mkdir(
                    parents=True, exist_ok=True
                )
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(
                        self.path_to_results
                        / Path("contour_plots")
                        / Path(f"{img_id}_with_contours.jpg")
                    ),
                    original_img,
                )

        # Write segmentation to file
        write_segmentation_to_file = kwargs.pop("write_segmentation_to_file", True)
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
                (self.path_to_results / Path("npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("npy_segmentation")
                    / Path(f"{img_id}_segmentation.npy"),
                    segmentation,
                )

            # Store coarse scale segmentation
            if write_coarse_segmentation_to_file:
                # TODO hardcoded - try to use base.dimensions and use 1cm grids
                coarse_shape = (150, 280)
                coarse_shape_reversed = tuple(reversed(coarse_shape))

                co2_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )
                co2_gas_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2_gas.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )

                coarse_segmentation = np.zeros(coarse_shape, dtype=int)
                coarse_segmentation[co2_coarse] += 1
                coarse_segmentation[co2_gas_coarse] += 1

                # Store segmentation as npy array
                (self.path_to_results / Path("coarse_npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("coarse_npy_segmentation")
                    / Path(f"{img_id}_coarse_segmentation.npy"),
                    coarse_segmentation,
                )

                # Store segmentation as csv file
                (self.path_to_results / Path("coarse_csv_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                array_to_csv(
                    self.path_to_results
                    / Path("coarse_csv_segmentation")
                    / Path(f"{img_id}_coarse_segmentation.csv"),
                    coarse_segmentation,
                    img.name,
                )

    def batch_segmentation(self, images: list[Path], **kwargs) -> None:
        """
        Standard batch segmentation.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments:
                plot_contours (bool): flag controlling whether the original image
                    is plotted with contours of the two CO2 phases; default False.
                write_contours_to_file (bool): flag controlling whether the plot from
                    plot_contours is written to file; default False.
                write_segmentation_to_file (bool): flag controlling whether the
                    CO2 segmentation is written to file, where water, dissolved CO2
                    and CO2(g) get decoded 0, 1, 2, respectively; default False.
                write_coarse_segmentation_to_file (bool): flag controlling whether
                    a coarse (280 x 150) representation of the CO2 segmentation from
                    write_segmentation_to_file is written to file; default False.

        """
        # TODO 280, 150 shapes are hardcoded for large rig

        for num, img in enumerate(images):

            tic = time.time()

            # Determine binary mask detecting any(!) CO2, and CO2(g)
            self.single_image_segmentation(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")

        return self.results

    # ! ---- Post-analysis

    # TODO this section needs revision. The segmentation can of course also be performed in parallel.
    # However, due to the large time consumption on the segmentation, a split between segmentation
    # And post-analysis seems most practical and is suggested here as well.

    # TODO allow to segment (without storing, by providing the right keyword), if storing is requested,
    # calling of single_image_segmentation is possible as well

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

            # Plot
            if plot_contours:
                plt.figure("Image with contours of CO2 segmentation")
                plt.imshow(original_img)
                plt.show()

            # Write corrected image with contours to file
            if write_contours_to_file:
                (self.path_to_results / Path("contour_plots")).mkdir(
                    parents=True, exist_ok=True
                )
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(
                        self.path_to_results
                        / Path("contour_plots")
                        / Path(f"{img_id}_with_contours.jpg")
                    ),
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
                (self.path_to_results / Path("npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("npy_segmentation")
                    / Path(f"{img_id}_segmentation.npy"),
                    segmentation,
                )

            # Store coarse scale segmentation
            if write_coarse_segmentation_to_file:
                # TODO shape
                coarse_shape = (150, 280)
                coarse_shape_reversed = tuple(reversed(coarse_shape))

                co2_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )
                co2_gas_coarse = skimage.img_as_bool(
                    cv2.resize(
                        skimage.img_as_ubyte(
                            co2_gas.img,
                        ),
                        coarse_shape_reversed,
                        interpolation=cv2.INTER_AREA,
                    )
                )

                coarse_segmentation = np.zeros(coarse_shape, dtype=int)
                coarse_segmentation[co2_coarse] += 1
                coarse_segmentation[co2_gas_coarse] += 1

                # Store segmentation as npy array
                (self.path_to_results / Path("coarse_npy_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                np.save(
                    self.path_to_results
                    / Path("coarse_npy_segmentation")
                    / Path(f"{img_id}_coarse_segmentation.npy"),
                    coarse_segmentation,
                )

                # Store segmentation as csv file
                (self.path_to_results / Path("coarse_csv_segmentation")).mkdir(
                    parents=True, exist_ok=True
                )
                array_to_csv(
                    self.path_to_results
                    / Path("coarse_csv_segmentation")
                    / Path(f"{img_id}_coarse_segmentation.csv"),
                    coarse_segmentation,
                    img.name,
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
