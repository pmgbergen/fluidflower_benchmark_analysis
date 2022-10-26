"""
Module containing the setup for the fluidflower rig, used for the PoroTwin1 optimal control project.

Applicable for uncorrected images (of C1).
"""
from pathlib import Path
from typing import Union

import cv2
import daria
import numpy as np
import skimage

# TODO use ..rigs
from largefluidflower import LargeFluidFlower


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
