"""
Module containing the setup for the fluidflower rig, used for the PoroTwin1 optimal control project.

Identical to that of C1, but without curvature correction, as input images already corrected.
"""
import json
from pathlib import Path
from typing import Optional, Union

import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from scipy.optimize import bisect, minimize


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
        img[~mask] = 0

        # Consider Value (3rd component from HSV) to detect signal strength.
        return img[:, :, 2]


class BenchmarkRig:
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
        # Read general config file
        f = open(config, "r")
        self.config = json.load(f)
        f.close()

        # Define set of baseline images and initiate object for caching
        # processed baseline images.
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]
        self.processed_baseline_images = None

        # Define correction objects
        self.drift_correction = daria.DriftCorrection(
            base=reference_base, roi=self.config["drift"]["roi"]
        )
        self.color_correction = daria.ColorCorrection(roi=self.config["color"]["roi"])
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Define baseline image as corrected daria Image - if exists use cached image
        self.base = self._read(reference_base)

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry(update=update_setup)

        # Neutralize the water zone in the baseline image
        self.base_with_clean_water = self._neutralize_water_zone(self.base)

        # Determine effective volumes, required for calibration, determining total mass etc.
        self._determine_effective_volumes()

        # ! ---- Analysis tools for detecting the different CO2 phases

        # ! ---- Concentration analysis to detect (mobile and dissolved) CO2. Hue serves as basis for the analysis.

        # TODO instead of using heterogeneous thresholding, the goal could be also (as for tracers)
        # to heterogeneously scale the signal prior to further analysis.

        # Works - very similar to value based. But uses a heterogeneous thresholding scheme.
        self.co2_mask_analysis = CO2MaskAnalysis(
            self.base_with_clean_water, color="", esf=self.esf, **self.config["co2"]
        )

        print("Warning: Concentration analysis is not calibrated.")

        self._setup_concentration_analysis(
            self.co2_mask_analysis,
            self.config["co2"]["cleaning_filter"],
            baseline,
            update_setup,
        )

        # ! ---- Mobile phase

        # TODO/NOTE: For C1, more tvd could be a possibility, as well as a higher threshold in the upper part, than lower.
        # However, there exist only adhoc justification for why the upper zone should be treated differently than the lower.
        # The lower has larger grain sizes, which however, also are represented in the upper zone, yet not the top layer of
        # the upper zone.

        self.mobile_co2_analysis = daria.BinaryConcentrationAnalysis(
            self.base_with_clean_water, color="blue", **self.config["mobile_co2"]
        )

        self._setup_concentration_analysis(
            self.mobile_co2_analysis,
            self.config["mobile_co2"]["cleaning_filter"],
            baseline,
            update_setup,
        )

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update: bool = False) -> None:
        """
        Use watershed segmentation and some cleaning to segment
        the geometry. Note that not all sand layers are detected
        by this approach.

        Args:
            update (bool): flag whether
        """

        # Fetch or generate and store labels
        if Path(self.config["segmentation"]["labels"]).exists() and not update:
            labels = np.load(self.config["segmentation"]["labels"])
            # TODO: Think of some general strategy - compaction analysis?
            labels = cv2.resize(labels, tuple(reversed(self.base.img.shape[:2])))
        else:
            labels = daria.segment(self.base.img, **self.config["segmentation"])
            np.save(self.config["segmentation"]["labels"], labels)

        plt.figure()
        plt.imshow(labels)
        plt.show()

        # Hardcoded: Identify water layer
        self.water = np.zeros(labels.shape[:2], dtype=bool)
        for i in [0, 1]:
            self.water = np.logical_or(self.water, labels == i)

        # Hardcoded: Identify ESF layer with ids 1, 4, 6
        self.esf = np.zeros(labels.shape[:2], dtype=bool)
        for i in [2, 5, 7]:
            self.esf = np.logical_or(self.esf, labels == i)

    def _determine_effective_volumes(self) -> None:
        """
        Compute constant voxel volume, based on the reported depth map.
        Use constant porosity for now; this should be changed, when a proper
        segmentation is provided.
        """
        # TODO use self.img here? Then we can use this on boxes as well! It takes approx.
        # The setup takes approx. 30 seconds. So the cost is bareable when not switiching
        # between different boxes too often.

        # Determine number of voxels in each dimension - assume 2d image
        Ny, Nx = self.base.img.shape[:2]
        Nz = 1
        x = np.arange(Nx)
        y = np.arange(Ny)
        X_pixel, Y_pixel = np.meshgrid(x, y)
        pixel_vector = np.transpose(np.vstack((np.ravel(X_pixel), np.ravel(Y_pixel))))
        coords_vector = self.base.coordinatesystem.pixelToCoordinate(pixel_vector)

        # TODO use the updated depth map.

        # Fetch physical dimensions
        width = self.config["physical_asset"]["dimensions"]["width"]
        height = self.config["physical_asset"]["dimensions"]["height"]

        # Depth of the rig, measured in discrete points, taking into account expansion.
        # Values taken from the benchmark description.
        x_coords_measurements, y_coords_measurements = np.meshgrid(
            np.array([0.0, 0.4, 0.7, 1.4, 2.8]),
            np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5]),
            # np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]),
            # np.array([
        )
        depth_measurements = np.array(
            [
                [0.019, 0.020, 0.019, 0.020, 0.019],
                [0.019, 0.024, 0.014, 0.024, 0.019],
                [0.019, 0.026, 0.028, 0.026, 0.019],
                [0.019, 0.027, 0.025, 0.025, 0.019],
                [0.019, 0.026, 0.026, 0.024, 0.019],
                [0.019, 0.019, 0.023, 0.020, 0.019],
            ]
        )
        ## Correct measurements
        # depth_measurements = np.array([20.72,20.7,20.9,21.2,21.4,21.3,21.3,20.5,20.1,21.1,20.8,20.0,19.4,19,20,
        # 20.43,23.1,24.4,25.8,25.7,25.8,25.6,25.1,25,25.6,25.3,23.8,21.7,19.7,20.2,
        # 20.9,23.8,27.1,28.9,27.9,28.4,27.8,27.1,27.3,27.4,27.6,25.6,22.7,20.5,20,
        # 20.7,24.5,28,29.3,29,29.6,29.1,27.9,28.6,28.4,28.1,26.9,23.2,20.9,19.7,
        # 20.7,23.6,27,28.8,29.6,29.8,28.5,27.7,28.7,28.9,27.5,27.5,22.7,20.7,20,
        # 20.7,22.4,25.3,27.8,29.3,29.2,28.4,27,28,28.4,26.8,26,22.4,19.9,20,
        # 20.7,21.5,24.2,26.3,27.2,27.4,27.5,26,26.8,27.7,26.8,25.2,22.4,19.9,20])

        depth_interpolator = RBFInterpolator(
            np.transpose(
                np.vstack(
                    (
                        np.ravel(x_coords_measurements),
                        np.ravel(y_coords_measurements),
                    )
                )
            ),
            np.ravel(depth_measurements),
        )

        # Evaluate depth function to determine depth map
        depth_vector = depth_interpolator(coords_vector)
        depth = depth_vector.reshape((Ny, Nx))

        # Porosity # TODO actually heterogeneous (though with very little differences)
        porosity = self.config["physical_asset"]["parameters"]["porosity"]

        # Compute effective volume per porous voxel
        self.effective_volumes = porosity * width * height * depth / (Nx * Ny * Nz)

        # TODO store depth map

    def _setup_concentration_analysis(
        self,
        concentration_analysis: daria.ConcentrationAnalysis,
        cleaning_filter: Union[str, Path],
        baseline_images: list[Union[str, Path]],
        update: bool = False,
    ) -> None:
        """
        Wrapper to find cleaning filter of the concentration analysis.

        Args:
            concentration_analysis (daria.ConcentrationAnalysis): concentration analysis
                to be set up.
            cleaning_filter (str or Path): path to cleaning filter array.
            baseline_images (list of str or Path): paths to baseline images.
            update (bool): flag controlling whether the calibration and setup should
                be updated.
        """
        # Set volume information
        concentration_analysis.update_volumes(self.effective_volumes)

        # Fetch or generate cleaning filter
        if not update and Path(cleaning_filter).exists():
            concentration_analysis.read_cleaning_filter_from_file(cleaning_filter)
        else:
            # Process baseline images used for setting up the cleaning filter
            if self.processed_baseline_images is None:
                self.processed_baseline_images = [
                    self._read(path) for path in baseline_images
                ]

            # Construct the concentration analysis specific cleaning filter
            concentration_analysis.find_cleaning_filter(self.processed_baseline_images)

            # Store the cleaning filter to file for later reuse.
            concentration_analysis.write_cleaning_filter_to_file(cleaning_filter)

        # TODO take care of calibration! and update config accordingly

    # ! ----- I/O

    def _read(self, path: Union[str, Path]) -> daria.Image:
        """
        Auxiliary reading methods for daria Images.

        Args:
            path (str or Path): path to file.

        Returns:
            daria.Image: image corrected for curvature and color.
        """

        print(f"Reading image from {str(path)}.")

        return daria.Image(
            img=path,
            drift_correction=self.drift_correction,
            curvature_correction=self.curvature_correction,
            color_correction=self.color_correction,
            # get_timestamp=True,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> None:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image
        """

        # Read and process
        self.img = self._read(path)

        # TODO read time from name
        # print(
        #    f"Image {str(path)} is considered, with rel. time {(self.img.timestamp - self.base.timestamp).total_seconds() / 3600.} hours."
        # )

        # Keep the original, processed image for visualization
        self.img_with_colorchecker = self.img.copy()

        # Neutralize water
        self.img = self._neutralize_water_zone(self.img)

    # ! ----- Cleaning routines

    def _neutralize_water_zone(self, img: daria.Image) -> daria.Image:
        """
        Deactivate water zone in the provided image.

        Args:
            img (daria.Image): image

        Returns:
            daria.Image: cleaned image
        """
        img_copy = img.copy()
        img_copy.img[self.water] = 0
        return img_copy

    # ! ----- Analysis tools

    def determine_co2_mask(self) -> daria.Image:
        """Segment domain into CO2 (mobile or dissolved) and water.

        Returns:
            daria.Image: boolean image detecting any co2.
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map
        co2 = self.co2_mask_analysis(img)

        return co2

    def determine_mobile_co2_mask(self, co2: daria.Image) -> daria.Image:
        """Determine mobile CO2.

        Args:
            co2 (daria.Image): boolean image detecting all co2.

        Returns:
            daria.Image: boolean image detecting mobile co2.
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Expect mobile CO2 only among CO2 but not in the ESF layer
        self.mobile_co2_analysis.update_mask(
            np.logical_and(co2.img, np.logical_not(self.esf))
        )

        # Determine mobile CO2 mask
        mobile_co2 = self.mobile_co2_analysis(img)

        return mobile_co2
