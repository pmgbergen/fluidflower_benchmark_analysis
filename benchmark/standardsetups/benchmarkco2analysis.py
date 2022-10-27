"""
Module containing the standardized CO2 analysis applicable for C1, ..., C5.
"""
import time
from pathlib import Path
from typing import Union

import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower


# Define specific concentration analysis class to detect mobile CO2
class CO2MaskAnalysis(daria.BinaryConcentrationAnalysis):
    """
    Binary concentration analysis based on a multichromatic HSV comparison
    and further analysis on the third component, i.e., V, identifying signal
    strength.
    """

    def __init__(self, base, color, esf, **kwargs) -> None:
        super().__init__(base, color, **kwargs)

        # Heterogeneous thresholding
        self.threshold_value = np.zeros(self.base.img.shape[:2], dtype=float)
        self.threshold_value[esf] = kwargs.pop("threshold value esf")
        self.threshold_value[~esf] = kwargs.pop("threshold value non-esf")

        # Fetch parameters for HUE based thresholding
        self.hue_low_threshold = kwargs.pop("threshold min hue", 0.0)
        self.hue_high_threshold = kwargs.pop("threshold max hue", 360)

    def _extract_scalar_information(self, img: daria.Image) -> None:
        """Transform to HSV.

        Args:
            img (daria.Image): Input image which shall be modified.
        """
        img.img = cv2.cvtColor(img.img.astype(np.float32), cv2.COLOR_RGB2HSV)

    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
        """Return 3rd component of HSV (value), identifying signal strength.

        Args:
            img (np.ndarray): trichromatic image in HSV color space.

        Returns:
            np.ndarray: monochromatic image
        """

        # Clip values in hue - from calibration.
        h_img = img[:, :, 0]

        mask = skimage.filters.apply_hysteresis_threshold(
            h_img, self.hue_low_threshold, self.hue_high_threshold
        )

        # Restrict to co2 mask
        img[~mask] = 0

        # Consider Value (3rd component from HSV) to detect signal strength.
        return img[:, :, 2]


class BenchmarkCO2Analysis(LargeFluidFlower, daria.CO2Analysis):
    """
    Class for managing the FluidFlower benchmark analysis.
    Identifiers for CO2 dissolved in water and CO2(g) are
    tailored to the benchmark set of the large rig.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for Benchmark rig.

        Sets up fixed config file required for preprocessing.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        daria.CO2Analysis.__init__(self, baseline, config, update_setup)

    # ! ---- Analysis tools for detecting the different CO2 phases

    def define_co2_analysis(self) -> daria.BinaryConcentrationAnalysis:
        """
        Identify CO2 using a heterogeneous HSV thresholding scheme.
        """
        co2_analysis = CO2MaskAnalysis(
            self.base, color="", esf=self.esf, **self.config["co2"]
        )

        return co2_analysis

    def define_co2_gas_analysis(self) -> daria.BinaryConcentrationAnalysis:
        """
        Identify CO2(g) using a thresholding scheme on the blue color channel.
        """
        co2_gas_analysis = daria.BinaryConcentrationAnalysis(
            self.base, color="blue", **self.config["co2(g)"]
        )

        return co2_gas_analysis

    # ! ----- Analysis tools

    def determine_co2_mask(self) -> daria.Image:
        """Determine CO2.

        Returns:
            daria.Image: boolean image detecting CO2.
        """
        # Extract co2 from analysis
        co2 = super().determine_co2()

        # Add expert knowledge. Turn of any signal in the water zone
        co2.img[self.water] = 0

        return co2

    def determine_co2_gas_mask(self, co2: daria.Image) -> daria.Image:
        """Determine CO2.

        Args:
            co2 (daria.Image): boolean image detecting all co2.

        Returns:
            daria.Image: boolean image detecting CO2(g).
        """
        # Extract co2 from analysis
        co2_gas = super().determine_co2_gas()

        # Add expert knowledge. Turn of any signal outside the presence of co2.
        # And turn off any signal in the ESF layer.
        co2_gas.img[~co2.img] = 0
        co2_gas.img[self.esf] = 0

        return co2_gas

    def single_image_analysis(self, img: Path, **kwargs) -> tuple[np.ndarray]:
        """
        Standard workflow to analyze CO2 phases.

        Args:
            image (Path): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: co2 mask
            np.ndarray: co2(g) mask
        """
        # Load the current image
        self.load_and_process_image(img)

        # Determine binary mask detecting any(!) CO2
        co2 = self.determine_co2_mask()

        # Determine binary mask detecting mobile CO2.
        co2_gas = self.determine_co2_gas_mask(co2)

        # ! ---- Post-analysis

        # Plot and store image with contours
        plot_contours = kwargs.pop("plot_contours", False)
        write_contours_to_file = kwargs.pop("write_contours_to_file", False)

        if plot_contours or write_contours_to_file:

            # Create image with contours on top
            contours_co2, _ = cv2.findContours(
                skimage.img_as_ubyte(co2.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_co2_gas, _ = cv2.findContours(
                skimage.img_as_ubyte(co2_gas.img),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            original_img = np.copy(self.img.img)
            original_img = skimage.img_as_ubyte(original_img)
            cv2.drawContours(original_img, contours_co2, -1, (0, 255, 0), 3)
            cv2.drawContours(original_img, contours_co2_gas, -1, (255, 255, 0), 3)

            # Plot
            if plot_contours:
                plt.figure("Image with contours of CO2 segmentation")
                plt.imshow(original_img)
                plt.show()

            # Write corrected image with contours to file
            if write_contours_to_file:
                img_id = Path(img.name).with_suffix("")
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"segmentation/{img_id}_with_contours.jpg", original_img)

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
                np.save(f"segmentation/{img_id}_segmentation.npy", segmentation)

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
                    f"segmentation/{img_id}_coarse_segmentation.npy",
                    coarse_segmentation,
                )

        return co2, co2_gas

    def batch_analysis(self, images: list[Path], **kwargs) -> None:
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
        """

        for num, img in enumerate(images):

            tic = time.time()

            # Determine binary mask detecting any(!) CO2, and CO2(g), and apply post-analysis.
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            print(f"Elapsed time for {img.name}: {time.time()- tic}.")
