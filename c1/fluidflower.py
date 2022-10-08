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


class MobileCO2Analysis(daria.ConcentrationAnalysis):
    def set_esf(self, esf: np.ndarray) -> None:
        self.esf = esf

    def set_co2(self, co2: np.ndarray) -> None:
        self.co2 = co2

    def _extract_scalar_information(self, img: daria.Image) -> None:
        pass

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        blue = signal[:, :, 2]
        return blue


class TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def __init__(self, base, color, resize_factor, **kwargs) -> None:
        super().__init__(base, color, **kwargs)
        self.resize_factor = resize_factor

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = cv2.resize(
            signal,
            None,
            fx=self.resize_factor,
            fy=self.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        signal = skimage.restoration.denoise_tv_chambolle(signal, 0.1)
        signal = np.atleast_3d(signal)
        return super().postprocess_signal(signal)


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

        # Concentration analysis to detect CO2. Hue serves as basis for the analysis.
        def hue(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]

        self.co2_mask_analysis = daria.ConcentrationAnalysis(
            self.base_with_clean_water,
            color=hue,
        )
        print("Warning: Concentration analysis is not calibrated.")
        self._setup_concentration_analysis(
            self.co2_mask_analysis,
            "co2_mask_cleaning_filter.npy",
            baseline,
            update_setup,
        )

        self.mobile_co2_analysis = MobileCO2Analysis(
            self.base_with_clean_water,
            color="empty",
        )
        print("Warning: Concentration analysis is not calibrated.")
        self._setup_concentration_analysis(
            self.mobile_co2_analysis,
            "mobile_co2_cleaning_filter.npy",
            baseline,
            update_setup,
        )

        # Forward volumes to the concentration analysis, required for calibration
        self.concentration_analysis.update_volumes(self.effective_volumes)

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
            np.array([0.0, 0.7, 1.3, 2.1, 2.8]),
            np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5]),
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

    # ! ---- Auxiliary calibration routines

    def _calibrate_concentration_analysis(
        self,
        paths: list[Path],
        injection_rate: float,
        initial_guess: tuple[float],
        tol: float,
    ) -> None:
        """
        Manager for calibration of concentration analysis, based on a
        constant injection rate.

        Args:
            images (list of daria.Image): images used for the calibration
            injection_rate (float): constant injection rate, assumed for the images
        """
        # Read in images
        images = [self._read(path) for path in paths]

        # Calibrate concentration analysis
        self.concentration_analysis.calibrate(
            injection_rate, images, initial_guess, tol
        )

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

    def determine_co2_mask(self) -> daria.Image:
        """Segment domain into CO2 (mobile or dissolved) and water.

        Returns:
            daria.Image: image array with segmentation
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map
        co2 = self.co2_mask_analysis(img)

        # Apply aggresive coarsening to make TVD feasible
        factor = 4
        co2_coarse = cv2.resize(
            co2.img, (factor * 280, factor * 150), interpolation=cv2.INTER_AREA
        )

        # Apply TVD to get rid of grains
        co2_denoised = skimage.restoration.denoise_tv_chambolle(co2_coarse, 0.0001)

        # Apply dynamic thresholding
        thresh = skimage.filters.threshold_otsu(co2_denoised)
        mask = co2_denoised > 0.6 * thresh

        # Remove small objects
        clean_mask = ndi.morphology.binary_opening(mask, structure=np.ones((2, 2)))

        # Resize to original size
        shape = self.base.img.shape[:2]
        co2_mask = skimage.img_as_ubyte(
            cv2.resize(clean_mask.astype(float), tuple(reversed(shape))).astype(bool)
        )

        return daria.Image(img=co2_mask, metadata=self.img.metadata)

    def determine_mobile_co2_mask(self, co2: daria.Image) -> daria.Image:
        """Segment domain into mobile CO2 and rest.

        Returns:
            daria.Image: image array with segmentation
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Fetch masks
        co2_mask = skimage.util.img_as_bool(co2.img)
        esf = self.esf

        # Extract concentration map
        mobile_co2 = self.mobile_co2_analysis(img)

        # Turn off signals from ESF (a priori knowledge)
        active = np.logical_and(co2_mask, ~esf)
        mobile_co2.img[~active] = 0

        # Cut off low signals - results in mask
        thresh = skimage.filters.threshold_otsu(
            np.ravel(mobile_co2.img)[np.ravel(active)]
        )
        print(thresh)
        # thresh = 0.065
        # thresh = 0.045 # Works really bad in the end
        thresh = 0.05
        mask = skimage.util.img_as_float(mobile_co2.img > thresh)

        # Clean signal
        mask = skimage.filters.rank.median(mask, skimage.morphology.disk(5))
        mask = skimage.morphology.binary_closing(
            skimage.util.img_as_bool(mask), footprint=np.ones((10, 10))
        )
        mask = skimage.morphology.remove_small_holes(mask, 50**2)
        mask = ndi.morphology.binary_opening(mask, structure=np.ones((5, 5)))
        mobile_co2_mask = mask

        return daria.Image(img=mobile_co2_mask, metadata=self.img.metadata)
