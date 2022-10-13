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
class MobileCO2Analysis(daria.BinaryConcentrationAnalysis):
    def _extract_scalar_information(self, img: daria.Image) -> None:
        pass

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        blue = signal[:, :, 0]
        # blue = signal[:, :, 2]
        # signal = skimage.util.img_as_ubyte(signal)
        # blue = skimage.util.img_as_float(cv2.cvtColor(signal, cv2.COLOR_RGB2GRAY))
        postprocessed_signal = super().postprocess_signal(blue)
        return postprocessed_signal


class MobileCO2AnalysisNew(daria.BinaryConcentrationAnalysis):
    def _extract_scalar_information(self, img: daria.Image) -> None:
        pass

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        # Convert to HSV color space
        signal = skimage.util.img_as_ubyte(signal)
        signal = cv2.cvtColor(signal, cv2.COLOR_RGB2HSV)

        # Threshold between two colors
        low = (1, 1, 50)
        high = (255, 200, 255)
        mask = skimage.util.img_as_bool(cv2.inRange(signal, low, high))
        signal[~mask] = 0

        # Consider the Value component
        value = signal[:, :, 2]
        plt.figure()
        plt.imshow(value)
        plt.show()
        postprocessed_signal = super().postprocess_signal(value)
        return postprocessed_signal


class BenchmarkRig:
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
        # Read general config file
        f = open(config_source, "r")
        self.config = json.load(f)
        f.close()

        # Some hardcoded config data (incl. not JSON serializable data)
        self.roi = {
            "color": (slice(0, 600, None), slice(0, 600, None)),
        }

        # Define correction objects
        self.color_correction = daria.ColorCorrection(roi=self.roi["color"])
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Store the baseline image
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]
        self.base = self._read(reference_base)

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry()

        # Neutralize the water zone in the baseline image
        self.base_with_clean_water = self._neutralize_water_zone(self.base)

        # Determine effective volumes, required for calibration, determining total mass etc.
        self._determine_effective_volumes()

        # ! ---- Analysis tools

        # Translation estimator to match different images such that color checker
        # is at same location
        self.translation_estimator = daria.TranslationEstimator()

        # Concentration analysis to detect (mobile and dissolved) CO2. Hue serves as basis for the analysis.
        def hue(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]

        config_co2_analysis = {
            # Presmoothing
            "presmoothing": True,
            # "presmoothing resize": 1.,
            # "presmoothing weight": 10,
            # "presmoothing eps": 1e-5,
            # "presmoothing max_num_iter": 1000,
            "presmoothing resize": 0.25,
            "presmoothing weight": 0.5,
            "presmoothing eps": 1e-5,
            "presmoothing max_num_iter": 100,
            "presmoothing method": "chambolle",
            # Thresholding
            "threshold value": 0.02,
            # Hole filling
            "max hole size": 20**2,
            # Local convex cover
            "local convex cover patch size": 10,
            # Postsmoothing
            "postsmoothing": True,
            "postsmoothing resize": 0.25,
            "postsmoothing weight": 1.0,
            "postsmoothing eps": 1e-5,
            "postsmoothing max_num_iter": 100,
            "postsmoothing method": "chambolle",
        }

        self.co2_mask_analysis = daria.BinaryConcentrationAnalysis(
            self.base_with_clean_water,
            color=hue,
            **config_co2_analysis,
        )

        print("Warning: Concentration analysis is not calibrated.")

        self._setup_concentration_analysis(
            self.co2_mask_analysis,
            "co2_mask_cleaning_filter.npy",
            baseline,
            update_setup,
        )

        ## Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        # config_mobile_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 1.,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 100,
        #    "presmoothing method": "chambolle",

        #    # Thresholding
        #    "threshold value": 0.048, # for blue
        #    #"threshold value": 0.25, # for red

        #    # Hole filling
        #    "max hole size": 20**2,

        #    # Local convex cover
        #    "local convex cover patch size": 10,

        #    # Postsmoothing
        #    "postsmoothing": True,
        #    "postsmoothing resize": 0.25,
        #    "postsmoothing weight": 4, # 4 if resize=0.25
        #    "postsmoothing eps": 1e-5,
        #    "postsmoothing max_num_iter": 100,
        #    "postsmoothing method": "chambolle"
        # }
        #
        # self.mobile_co2_analysis = MobileCO2Analysis(
        #    self.base_with_clean_water,
        #    color="empty",
        #    **config_mobile_co2_analysis
        # )

        # Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        config_mobile_co2_analysis = {
            # Presmoothing
            "presmoothing": True,
            "presmoothing resize": 0.5,
            "presmoothing weight": 0.02,
            "presmoothing eps": 1e-5,
            "presmoothing max_num_iter": 1000,
            "presmoothing method": "anisotropic bregman",
            # Thresholding
            "threshold value": 10,
            # Hole filling
            "max hole size": 20**2,
            # Local convex cover
            "local convex cover patch size": 10,
            # Postsmoothing
            "postsmoothing": False,
            # "postsmoothing resize": 0.25,
            # "postsmoothing weight": 4, # 4 if resize=0.25
            # "postsmoothing eps": 1e-5,
            # "postsmoothing max_num_iter": 100,
            # "postsmoothing method": "chambolle"
        }

        self.mobile_co2_analysis = MobileCO2AnalysisNew(
            self.base_with_clean_water, color="empty", **config_mobile_co2_analysis
        )

        print("Warning: Concentration analysis is not calibrated.")

        self._setup_concentration_analysis(
            self.mobile_co2_analysis,
            "mobile_co2_cleaning_filter_new.npy",
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

        # Fetch or generate labels
        if Path("labels.npy").exists() and not update:
            labels = np.load("labels.npy")
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
            np.save("labels.npy", labels)

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

        # Fetch physical dimensions
        width = self.config["physical_asset"]["dimensions"]["width"]
        height = self.config["physical_asset"]["dimensions"]["height"]

        # Depth of the rig, measured in discrete points, taking into account expansion.
        # Values taken from the benchmark description.
        # TODO: Have these values been updated after opening disassembling the FluidFlower?
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

    # TODO split methods and move to daria.ConcentrationAnalysis
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
        if (
            not update
            and "calibration" in self.config
            and Path(cleaning_filter).exists()
        ):
            # TODO take care of calibration!
            concentration_analysis.read_calibration_from_file(
                self.config["calibration"], cleaning_filter
            )
        else:
            # Construct and store the cleaning filter.
            images = [self._read(path) for path in baseline_images]
            concentration_analysis.find_cleaning_filter(images)
            # TODO take care of calibration!
            concentration_analysis.write_calibration_to_file("tmp", cleaning_filter)

            # TODO separate cleaning from calibration
            # TODO perform actual calibration to obtain scaling.
            # TODO update config

    # ! ----- I/O

    def _read(self, path: Union[str, Path]) -> daria.Image:
        """
        Auxiliary reading methods for daria Images.

        Args:
            path (str or Path): path to file.

        Returns:
            daria.Image: image corrected for curvature and color.
        """
        return daria.Image(
            img=path,
            curvature_correction=self.curvature_correction,
            color_correction=self.color_correction,
        )

    def load_and_process_image(self, path: Union[str, Path]) -> None:
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image
        """
        # Read and process
        self.img = self._read(path)

        # Align with base image
        self.translation_estimator.match_roi(
            img_src=self.img,
            img_dst=self.base,
            roi_src=self.roi["color"],
            roi_dst=self.roi["color"],
        )

        # Neutralize water
        self.img = self._neutralize_water_zone(self.img)

    def store(
        self,
        img: daria.Image,
        path: Union[str, Path],
        cartesian_indexing: bool = True,
        store_image: bool = False,
    ) -> bool:
        """Convert to correct format (use Cartesian indexing by default)
        and store to file.

        Args:
            img (daria.Image): image
            path (str or Path): path for storage
            cartesian_indexing (bool): flag whether the stored numpy array should
                use cartesian indexing instead of matrix indexing.
            store_image (bool): flag whether a low-resolution image should be
                stored as well.
        """
        # Remove the suffix from the provided path
        plain_path = path.with_suffix("")

        # Store the lowe-resolution image
        if store_image:
            cv2.imwrite(
                str(plain_path) + "_img.jpg",
                skimage.util.img_as_ubyte(img.img),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )

        # Store array
        np.save(
            str(plain_path) + "_array.npy",
            daria.matrixToCartesianIndexing(img.img) if cartesian_indexing else img.img,
        )

        return True

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

    ## NOTE: Latest version!
    # def determine_co2_mask(
    #    self, presmoothing: bool = False, convexification: bool = False
    # ) -> daria.Image:
    #    """Segment domain into CO2 (mobile or dissolved) and water.

    #    Returns:
    #        daria.Image: image array with segmentation
    #    """
    #    # Make a copy of the current image
    #    img = self.img.copy()

    #    # Extract concentration map
    #    co2 = self.co2_mask_analysis(img)

    #    ## TODO try using aggressive smoothing here.
    #    #if presmoothing:
    #    #    co2.img = cv2.resize(co2.img.astype(np.float32), None, fx = 0.25, fy = 0.25)
    #    #    co2.img = skimage.restoration.denoise_tv_chambolle(co2.img, 0.05, eps = 1e-5, max_num_iter = 100)
    #    #    co2.img = cv2.resize(co2.img, tuple(reversed(original.shape[:2])))
    #    co2.img = skimage.restoration.denoise_tv_chambolle(
    #        co2.img, 10., eps=1e-5, max_num_iter=1000
    #    )

    #    # Apply thresholding
    #    thresh = skimage.filters.threshold_otsu(co2.img)
    #    print("co2", thresh)

    #    # TODO find through calibration!
    #    thresh = 0.02
    #    mask = co2.img > thresh

    #    # Fill holes
    #    mask = skimage.morphology.remove_small_holes(mask, area_threshold=20**2)

    #    # Loop through patches and fill up
    #    convex_mask = np.zeros(mask.shape[:2], dtype=bool)
    #    size = 10
    #    Ny, Nx = mask.shape[:2]
    #    for row in range(int(Ny / size) + 1):
    #        for col in range(int(Nx / size) + 1):
    #            roi = (
    #                slice(row * size, (row + 1) * size),
    #                slice(col * size, (col + 1) * size),
    #            )
    #            convex_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])

    #    # Clean up by resizing and denoising
    #    convex_mask = cv2.resize(convex_mask.astype(np.float32), None, fx=0.25, fy=0.25)
    #    convex_mask = skimage.restoration.denoise_tv_chambolle(
    #        convex_mask, 1, eps=1e-5, max_num_iter=100
    #    )

    #    # Extract mask from smooth image
    #    thresh = 0.5
    #    mask = convex_mask > thresh

    #    # Resize to original size
    #    mask = skimage.util.img_as_bool(
    #        cv2.resize(mask.astype(np.float32), tuple(reversed(self.img.img.shape[:2])))
    #    )

    #    # Define final result
    #    co2_mask = mask

    #    return daria.Image(img=co2_mask, metadata=self.img.metadata)

    # def determine_co2_mask(self) -> daria.Image:
    #    """Segment domain into CO2 (mobile or dissolved) and water.

    #    Returns:
    #        daria.Image: image array with segmentation
    #    """
    #    # Make a copy of the current image
    #    img = self.img.copy()

    #    # Extract concentration map
    #    co2 = self.co2_mask_analysis(img)

    #    img = cv2.resize(co2.img.astype(np.float32), None, fx = 0.25, fy = 0.25)
    #    img = skimage.restoration.denoise_tv_chambolle(img, 0.05, eps = 1e-5, max_num_iter = 100)
    #    img = cv2.resize(img, tuple(reversed(self.img.img.shape[:2])))

    #    # Apply thresholding
    #    thresh = skimage.filters.threshold_otsu(img)
    #    print("co2", thresh)

    #    # TODO find through calibration!
    #    thresh = 0.020
    #    mask = img > thresh

    #    # Define final result
    #    co2_mask = mask

    #    return daria.Image(img=co2_mask, metadata=self.img.metadata)

    # Routine for CO2Analysis (multichannel diff and red)

    # def determine_co2_mask(self) -> daria.Image:
    #    """Segment domain into CO2 (mobile or dissolved) and water.

    #    Returns:
    #        daria.Image: image array with segmentation
    #    """
    #    # Make a copy of the current image
    #    img = self.img.copy()

    #    # Extract concentration map
    #    co2 = self.co2_mask_analysis(img)

    #    # Apply thresholding
    #    thresh = skimage.filters.threshold_otsu(co2.img)
    #    print("co2", thresh)

    #    # TODO find through calibration!
    #    #thresh = 0.129
    #    thresh = 0.078
    #    mask = co2.img > thresh
    #
    #    # Fill holes
    #    mask = skimage.morphology.remove_small_holes(mask, area_threshold = 20**2)
    #
    #    # Loop through patches and fill up
    #    convex_mask = np.zeros(mask.shape[:2], dtype=bool)
    #    size = 10
    #    Ny, Nx = mask.shape[:2]
    #    for row in range(int(Ny / size) + 1):
    #        for col in range(int(Nx / size) + 1):
    #            roi = (slice(row * size, (row+1) * size), slice(col * size, (col+1) * size))
    #            convex_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])
    #
    #    # Clean up by resizing and denoising
    #    convex_mask = cv2.resize(convex_mask.astype(np.float32), None, fx = 0.25, fy = 0.25)
    #    convex_mask = skimage.restoration.denoise_tv_chambolle(convex_mask, 1, eps = 1e-5, max_num_iter = 100)
    #
    #    # Extract mask from smooth image
    #    #thresh = skimage.filters.threshold_otsu(convex_mask)
    #    thresh = 0.5
    #    mask = convex_mask > thresh

    #    # Resize to original size
    #    mask = skimage.util.img_as_bool(cv2.resize(mask.astype(np.float32), tuple(reversed(self.img.img.shape[:2]))))

    #    # Define final result
    #    co2_mask = mask

    #    return daria.Image(img=co2_mask, metadata=self.img.metadata)

    def determine_mobile_co2_mask(self, co2: daria.Image) -> daria.Image:
        """Segment domain into mobile CO2 and rest.

        Returns:
            daria.Image: image array with segmentation
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Mark co2 as active set, but turn off esf
        self.mobile_co2_analysis.update_active_roi(
            np.logical_and(
                co2.img,
                np.logical_not(self.esf),
            )
        )

        # Extract concentration map
        mobile_co2 = self.mobile_co2_analysis(img)

        return mobile_co2

    ## NOTE: Latest version!
    # def determine_mobile_co2_mask_(
    #    self, co2: daria.Image, presmoothing: bool = True, convexification: bool = True
    # ) -> daria.Image:
    #    """Segment domain into mobile CO2 and rest.

    #    Returns:
    #        daria.Image: image array with segmentation
    #    """
    #    # Make a copy of the current image
    #    img = self.img.copy()

    #    # Extract concentration map
    #    mobile_co2 = self.mobile_co2_analysis(img)

    #    # Turn off signals from ESF (a priori knowledge)
    #    co2_mask = skimage.util.img_as_bool(co2.img)
    #    esf = self.esf
    #    active = np.logical_and(co2_mask, ~esf)
    #    mobile_co2.img[~active] = 0

    #    plt.figure()
    #    plt.imshow(mobile_co2.img)

    #    # Smooth tiny bit
    #    if presmoothing:
    #        mobile_co2.img = cv2.resize(mobile_co2.img, None, fx=0.25, fy=0.25)
    #        # TODO standardize regularization parameter
    #        #mobile_co2.img = skimage.restoration.denoise_tv_chambolle(
    #        #    mobile_co2.img, 0.05, eps=1e-5, max_num_iter=100
    #        #)
    #        plt.figure()
    #        plt.imshow(mobile_co2.img)

    #        mobile_co2.img = skimage.restoration.denoise_tv_chambolle(
    #            mobile_co2.img, 0.5, eps=1e-5, max_num_iter=100
    #        )
    #        plt.figure()
    #        plt.imshow(mobile_co2.img)
    #        mobile_co2.img = cv2.resize(
    #            mobile_co2.img, tuple(reversed(self.img.img.shape[:2]))
    #        )

    #        #mobile_co2.img = cv2.resize(mobile_co2.img, None, fx=0.25, fy=0.25)
    #        # TODO standardize regularization parameter
    #        #mobile_co2.img = skimage.restoration.denoise_tv_chambolle(
    #        #    mobile_co2.img, 0.05, eps=1e-5, max_num_iter=100
    #        #)
    #        #plt.figure()
    #        #plt.imshow(mobile_co2.img)

    #        #mobile_co2.img = skimage.restoration.denoise_tv_chambolle(
    #        #    mobile_co2.img, 10., eps=1e-5, max_num_iter=1000
    #        #)
    #        #plt.figure()
    #        #plt.imshow(mobile_co2.img)
    #        #mobile_co2.img = cv2.resize(
    #        #    mobile_co2.img, tuple(reversed(self.img.img.shape[:2]))
    #        #)

    #    plt.figure()
    #    plt.imshow(mobile_co2.img)

    #    # Apply thresholding
    #    active_img = np.ravel(mobile_co2.img)[np.ravel(active)]
    #    thresh = skimage.filters.threshold_otsu(active_img)
    #    print("mobile co2", thresh)

    #    # TODO calibration - depends most likely on presmoothing!
    #    # When coarsening and using simple smoothing
    #    #thresh = 0.02
    #    # No coarsening and some smoothing
    #    #thresh = 0.043
    #    thresh = 0.048

    #    mask = mobile_co2.img > thresh

    #    plt.figure()
    #    plt.imshow(mask)
    #    plt.show()

    #    # Fill holes
    #    mask = skimage.morphology.remove_small_holes(mask, area_threshold=20**2)

    #    if convexification:
    #        # Loop through patches and fill up
    #        convex_mask = np.zeros(mask.shape[:2], dtype=bool)
    #        size = 10
    #        Ny, Nx = mask.shape[:2]
    #        for row in range(int(Ny / size) + 1):
    #            for col in range(int(Nx / size) + 1):
    #                roi = (
    #                    slice(row * size, (row + 1) * size),
    #                    slice(col * size, (col + 1) * size),
    #                )
    #                convex_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])

    #        # Define final result
    #        co2_mask = convex_mask
    #    else:
    #        co2_mask = mask

    #    return daria.Image(img=co2_mask, metadata=self.img.metadata)

    # OLD
    # def determine_mobile_co2_mask(self, co2: daria.Image) -> daria.Image:
    #    """Segment domain into mobile CO2 and rest.

    #    Returns:
    #        daria.Image: image array with segmentation
    #    """
    #    # Make a copy of the current image
    #    img = self.img.copy()

    #    # Fetch masks
    #    co2_mask = skimage.util.img_as_bool(co2.img)
    #    esf = self.esf

    #    # Extract concentration map
    #    mobile_co2 = self.mobile_co2_analysis(img)

    #    plt.figure()
    #    plt.imshow(mobile_co2.img)
    #    plt.show()

    #    # Turn off signals from ESF (a priori knowledge)
    #    active = np.logical_and(co2_mask, ~esf)
    #    mobile_co2.img[~active] = 0

    #    # Cut off low signals - results in mask
    #    thresh = skimage.filters.threshold_otsu(
    #        np.ravel(mobile_co2.img)[np.ravel(active)]
    #    )
    #    print("mobile", thresh)

    #    # When using BLUE:
    #    ## thresh = 0.065
    #    ## thresh = 0.045 # Works really bad in the end
    #    # thresh = 0.05

    #    # When using RED:
    #    thresh = 0.24  # * 1.1

    #    # To identify nitrogen: Try
    #    thresh = 0.26

    #    # Define mask
    #    mask = skimage.util.img_as_float(mobile_co2.img > thresh)

    #    # Clean signal
    #    mask = skimage.filters.rank.median(mask, skimage.morphology.disk(5))
    #    mask = skimage.morphology.binary_closing(
    #        skimage.util.img_as_bool(mask), footprint=np.ones((10, 10))
    #    )
    #    mask = skimage.morphology.remove_small_holes(mask, 50**2)
    #    mask = ndi.morphology.binary_opening(mask, structure=np.ones((5, 5)))
    #    mobile_co2_mask = mask

    #    return daria.Image(img=mobile_co2_mask, metadata=self.img.metadata)
