"""
Module containing the setup for the fluidflower rig, used for the PoroTwin1 optimal control project.
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

        # Restrict to co2 mask
        img[~mask] = 0

        # Consider Value (3rd component from HSV) to detect signal strength.
        return img[:, :, 2]


class BenchmarkRig:
    """
    Class for managing the analysis of C1.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config_source: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for Benchmark rig.

        Sets up fixed config file required for preprocessing.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config_source (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        # TODO
        # fetch paths from config files. Replace for instance tmp.

        # Read general config file
        f = open(config_source, "r")
        self.config = json.load(f)
        f.close()

        # Some hardcoded config data (incl. not JSON serializable data)
        self.roi = {
            "color": (slice(0, 600, None), slice(0, 600, None)),
            "sealed fault top": (slice(1080, 1320), slice(2650, 2900)),
            "color_checker_marks": np.array(
                [
                    [377, 504],
                    [560, 511],
                    [562, 251],
                    [380, 242],
                ]
            ),
        }

        # Define set of baseline images and initiate object for caching
        # processed baseline images.
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]
        self.processed_baseline_images = None

        # Define correction objects
        self.drift_correction = daria.DriftCorrection(
            base=reference_base, roi=self.roi["sealed fault top"]
        )
        self.color_correction = daria.ColorCorrection(
            roi=self.roi["color_checker_marks"]
        )
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Define baseline image as corrected daria Image - if exists use cached image
        if Path("tmp/processed_baseline.npy").exists() and not update_setup and False:
            processed_base = np.load("tmp/processed_baseline.npy")
            self.base = daria.Image(
                processed_base,
                width=self.config["physical_asset"]["dimensions"]["width"],
                height=self.config["physical_asset"]["dimensions"]["height"],
            )
        else:
            print("Init: Read baseline image.")
            self.base = self._read(reference_base)
            np.save("tmp/processed_baseline.npy", self.base.img)
            print("Init: Finish processing baseline image.")

            plt.imshow(self.base.img)
            plt.show()

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry()

        # Neutralize the water zone in the baseline image
        self.base_with_clean_water = self._neutralize_water_zone(self.base)

        # Determine effective volumes, required for calibration, determining total mass etc.
        self._determine_effective_volumes()

        # ! ---- Analysis tools

        # Concentration analysis to detect (mobile and dissolved) CO2. Hue serves as basis for the analysis.

        # TODO instead of using heterogeneous thresholding, the goal could be also (as for tracers)
        # to heterogeneously scale the signal prior to further analysis.

        # Works - very similar to value based. But uses a heterogeneous thresholding scheme.
        config_co2_analysis = {
            # Color spectrum of interest - hue
            "threshold min hue": 14,
            "threshold max hue": 70,
            # Presmoothing - acting on a signal (be careful with the weight, rather expect low tolerances)
            "presmoothing": True,
            "presmoothing resize": 0.5,
            "presmoothing weight": 5,
            "presmoothing eps": 1e-4,
            "presmoothing max_num_iter": 200,
            # Heterogeneous thresholding
            "threshold value esf": 0.01,
            "threshold value non-esf": 0.04,
            # Light postsmoothing
            "postsmoothing": True,
            "postsmoothing resize": 0.5,
            "postsmoothing weight": 5,
            "postsmoothing eps": 1e-4,
            "postsmoothing max_num_iter": 100,
        }
        self.co2_mask_analysis = CO2MaskAnalysis(
            self.base_with_clean_water,
            color="",
            esf=self.esf,
            verbosity=True,
            **config_co2_analysis,
        )

        print("Warning: Concentration analysis is not calibrated.")

        self._setup_concentration_analysis(
            self.co2_mask_analysis,
            "tmp/co2_mask_cleaning_filter_hsv2.npy",
            baseline,
            update_setup,
        )

        # ! ---- Mobile phase

        # Blue based
        config_mobile_co2_analysis = {
            # Presmoothing
            "presmoothing": True,
            "presmoothing resize": 0.5,
            "presmoothing weight": 5,
            "presmoothing eps": 1e-4,
            "presmoothing max_num_iter": 100,
            # Thresholding
            "threshold value": 0.04,
            # Presmoothing
            "postsmoothing": True,
            "postsmoothing resize": 0.5,
            "postsmoothing weight": 5,
            "postsmoothing eps": 1e-4,
            "postsmoothing max_num_iter": 100,
            # Posterior thresholding
            "posterior": True,
            "threshold posterior gradient modulus": 0.002,
        }

        self.mobile_co2_analysis = daria.BinaryConcentrationAnalysis(
            self.base_with_clean_water,
            color="blue",
            # verbosity = True, # Set to True to tune the parameters
            **config_mobile_co2_analysis,
        )

        self._setup_concentration_analysis(
            self.mobile_co2_analysis,
            "tmp/co2_mask_cleaning_filter_blue.npy",
            baseline,
            update_setup,
        )

        ## Red based
        # config_mobile_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 5,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    # Thresholding
        #    "threshold value": 0.24,
        #    ## Presmoothing
        #    #"postsmoothing": True,
        #    #"postsmoothing resize": 0.5,
        #    #"postsmoothing weight": 5,
        #    #"postsmoothing eps": 1e-4,
        #    #"postsmoothing max_num_iter": 100,
        # }

        # self.mobile_co2_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="red",
        #    **config_mobile_co2_analysis,
        # )

        # self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "co2_mask_cleaning_filter_red.npy",
        #    baseline,
        #    update_setup,
        # )

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update: bool = False) -> None:
        """
        Use watershed segmentation and some cleaning to segment
        the geometry. Note that not all sand layers are detected
        by this approach.

        Args:
            update (bool): flag whether
        """

        # Fetch or generate labels
        if Path("tmp/labels.npy").exists() and not update:
            labels = np.load("tmp/labels.npy")
        else:
            # Require scalar representation - work with grayscale image. Alternatives exist, but with little difference.
            basis = cv2.cvtColor(self.base.img, cv2.COLOR_RGB2GRAY)

            # Smooth the image to get rid of sand grains
            denoised = skimage.filters.rank.median(basis, skimage.morphology.disk(20))

            # Resize image
            denoised = skimage.util.img_as_ubyte(
                skimage.transform.rescale(denoised, 0.1, anti_aliasing=False)
            )

            # Find continuous region, i.e., areas with low local gradient
            markers_basis = skimage.filters.rank.gradient(
                denoised, skimage.morphology.disk(10)
            )
            markers = markers_basis < 20  # hardcoded
            markers = ndi.label(markers)[0]

            # Find edges
            gradient = skimage.filters.rank.gradient(
                denoised, skimage.morphology.disk(2)
            )

            # Process the watershed and resize to the original size
            labels_rescaled = skimage.util.img_as_ubyte(
                skimage.segmentation.watershed(gradient, markers)
            )
            labels = skimage.util.img_as_ubyte(
                skimage.transform.resize(labels_rescaled, self.base.img.shape[:2])
            )

            # NOTE: Segmentation needs some cleaning, as some areas are just small,
            # tiny lines, etc. Define some auxiliary methods for this.

            # Make labels increasing in steps of 1 starting from zero
            def _reset(labels):
                pre_labels = np.unique(labels)
                for i, label in enumerate(pre_labels):
                    mask = labels == label
                    labels[mask] = i
                return labels

            # Fill holes
            def _fill_holes(labels):
                pre_labels = np.unique(labels)
                for label in pre_labels:
                    mask = labels == label
                    mask = ndi.binary_fill_holes(mask).astype(bool)
                    labels[mask] = label
                return labels

            # Dilate objects
            def _dilate_by_size(labels, footprint, decreasing_order):
                # Determine sizes of all marked areas
                pre_labels = np.unique(labels)
                sizes = [np.count_nonzero(labels == label) for label in pre_labels]
                # Sort from small to large
                labels_sorted_sizes = np.argsort(sizes)
                if decreasing_order:
                    labels_sorted_sizes = np.flip(labels_sorted_sizes)
                # Erode for each label if still existent
                for label in labels_sorted_sizes:
                    mask = labels == label
                    mask = skimage.morphology.binary_dilation(
                        mask, skimage.morphology.disk(footprint)
                    )
                    labels[mask] = label
                return labels

            # Extend internals to the boundary
            def _boundary(labels, thickness=10):
                # Top
                labels[:thickness, :] = labels[thickness : 2 * thickness, :]
                # Bottom
                labels[-thickness - 1 : -1, :] = labels[-2 * thickness : -thickness, :]
                # Left
                labels[:, :thickness] = labels[:, thickness : 2 * thickness]
                # Right
                labels[:, -thickness - 1 : -1] = labels[:, -2 * thickness : -thickness]
                return labels

            # Simplify the segmentation drastically by removing small entities, and correct for boundary effects.
            labels = _reset(labels)
            labels = _dilate_by_size(labels, 10, False)
            labels = _reset(labels)
            labels = _fill_holes(labels)
            labels = _reset(labels)
            labels = _dilate_by_size(labels, 10, True)
            labels = _reset(labels)
            labels = _boundary(labels)
            labels = _boundary(labels, 55)

            # Hardcoded: Remove area in water zone
            labels[labels == 1] = 0
            labels = _reset(labels)

            # Store to file
            np.save("tmp/labels.npy", labels)

        # Hardcoded: Identify ESF layer with ids 1, 4, 6
        self.esf = np.zeros(labels.shape[:2], dtype=bool)
        for i in [1, 4, 6]:
            self.esf = np.logical_or(self.esf, labels == i)

        # Hardcoded: Identify water layer
        self.water = labels == 0

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
            get_timestamp=True,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> None:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image
        """

        # Read and process
        self.img = self._read(path)

        print(
            f"Image {str(path)} is considered, with rel. time {(self.img.timestamp - self.base.timestamp).total_seconds() / 3600.} hours."
        )

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

    # Routine for hue based co2 analysis

    def determine_co2_mask(
        self, presmoothing: bool = False, convexification: bool = False
    ) -> daria.Image:
        """Segment domain into CO2 (mobile or dissolved) and water.

        Returns:
            daria.Image: image array with segmentation
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map
        co2 = self.co2_mask_analysis(img)

        return co2

    def determine_mobile_co2_mask(self, co2: daria.Image) -> daria.Image:
        """Segment domain into mobile CO2 and rest.

        Returns:
            daria.Image: image array with segmentation
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Determine mobile CO2 mask
        mobile_co2 = self.mobile_co2_analysis(img)

        return mobile_co2
