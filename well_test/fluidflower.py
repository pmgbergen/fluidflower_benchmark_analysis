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


class TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def __init__(self, base, color, resize_factor, **kwargs) -> None:
        super().__init__(base, color, **kwargs)
        self.resize_factor = resize_factor

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        # signal = cv2.resize(
        #    signal,
        #    None,
        #    fx=self.resize_factor,
        #    fy=self.resize_factor,
        #    interpolation=cv2.INTER_AREA,
        # )
        np.save("signal.npy", signal)
        signal = skimage.restoration.denoise_tv_chambolle(
            signal, weight=20, eps=1e-4, max_num_iter=100
        )
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
            "color": np.array(
                [
                    [375, 493],
                    [558, 501],
                    [560, 240],
                    [378, 230],
                ]
            )
        }

        # Define a set of baseline images
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]

        # Define correction objects
        self.drift_correction = daria.DriftCorrection(
            base=reference_base, roi=self.roi["color"]
        )
        self.color_correction = daria.ColorCorrection(roi=self.roi["color"])
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Define baseline image as corrected daria Image - if exists use cached image
        print("Init: Read baseline image.")
        self.base = self._read(reference_base)
        print("Init: Finish processing baseline image.")

        # Segment the baseline image; identidy water and esf layer.
        print("segment")
        self._segment_geometry()
        print("segment")

        # Neutralize the water zone in the baseline image
        self.base_with_clean_water = self._neutralize_water_zone(self.base)

        # Determine effective volumes, required for calibration, determining total mass etc.
        print("Find effective vol")
        self._determine_effective_volumes()
        print("Find effective vol")

        # ! ---- Analysis tools

        self.concentration_analysis = TailoredConcentrationAnalysis(
            self.base_with_clean_water, color="gray", resize_factor=1
        )
        print("concentration")
        self._setup_concentration_analysis(
            self.concentration_analysis,
            "cleaning_filter.npy",
            baseline,
            update_setup,
        )

        print("Init: Finished setup.")

    # ! ---- Auxiliary setup routines

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
        #        concentration_analysis.update_volumes(self.effective_volumes)

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
            labels = _boundary(labels, 75)

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

        if not Path("volumes.npy").exists():
            # Determine number of voxels in each dimension - assume 2d image
            Ny, Nx = self.base.img.shape[:2]
            Nz = 1
            x = np.arange(Nx)
            y = np.arange(Ny)
            X_pixel, Y_pixel = np.meshgrid(x, y)
            pixel_vector = np.transpose(
                np.vstack((np.ravel(X_pixel), np.ravel(Y_pixel)))
            )
            coords_vector = self.base.coordinatesystem.pixelToCoordinate(pixel_vector)

            # Fetch physical dimensions
            width = self.config["physical_asset"]["dimensions"]["width"]
            height = self.config["physical_asset"]["dimensions"]["height"]

            # Depth of the rig, measured in discrete points, taking into account expansion.
            # Values taken from the benchmark description.

            # Coordinates at which depth measurements have been taken.
            # Note that the y-coordinate differs depending on the x-coordinate,
            # which dissallows use of np.meshgrid. Instead, the meshgrid is
            # constructed by hand.
            x_measurements = np.array(
                [
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                    1.6,
                    1.8,
                    2.0,
                    2.2,
                    2.4,
                    2.6,
                    2.8,
                ]
                * 7
            )
            y_measurements = np.array(
                [0.0] * 15
                + [0.2] * 15
                + [0.4] * 15
                + [0.6] * 4
                + [0.61, 0.63]
                + [0.64] * 6
                + [0.6] * 3
                + [0.8] * 15
                + [1.0] * 15
                + [1.2] * 15
            )

            # Measurements in mm, including the thickness of the equipment used to measure
            # the depth (1.5 mm).
            depth_measurements = np.array(
                [
                    20.72,
                    20.7,
                    20.9,
                    21.2,
                    21.4,
                    21.3,
                    21.3,
                    20.5,
                    20.1,
                    21.1,
                    20.8,
                    20.0,
                    19.4,
                    19,
                    20,
                    20.43,
                    23.1,
                    24.4,
                    25.8,
                    25.7,
                    25.8,
                    25.6,
                    25.1,
                    25,
                    25.6,
                    25.3,
                    23.8,
                    21.7,
                    19.7,
                    20.2,
                    20.9,
                    23.8,
                    27.1,
                    28.9,
                    27.9,
                    28.4,
                    27.8,
                    27.1,
                    27.3,
                    27.4,
                    27.6,
                    25.6,
                    22.7,
                    20.5,
                    20,
                    20.7,
                    24.5,
                    28,
                    29.3,
                    29,
                    29.6,
                    29.1,
                    27.9,
                    28.6,
                    28.4,
                    28.1,
                    26.9,
                    23.2,
                    20.9,
                    19.7,
                    20.7,
                    23.6,
                    27,
                    28.8,
                    29.6,
                    29.8,
                    28.5,
                    27.7,
                    28.7,
                    28.9,
                    27.5,
                    27.5,
                    22.7,
                    20.7,
                    20,
                    20.7,
                    22.4,
                    25.3,
                    27.8,
                    29.3,
                    29.2,
                    28.4,
                    27,
                    28,
                    28.4,
                    26.8,
                    26,
                    22.4,
                    19.9,
                    20,
                    20.7,
                    21.5,
                    24.2,
                    26.3,
                    27.2,
                    27.4,
                    27.5,
                    26,
                    26.8,
                    27.7,
                    26.8,
                    25.2,
                    22.4,
                    19.9,
                    20,
                ]
            )

            # TODO double check the measurements with Benyamine.

            # Correct for thickness of measurement equipment
            depth_measurements -= 1.5

            # Convert depth to meters
            depth_measurements *= 1e-3

            depth_interpolator = RBFInterpolator(
                np.transpose(
                    np.vstack(
                        (
                            x_measurements,
                            y_measurements,
                        )
                    )
                ),
                depth_measurements,
            )

            # Evaluate depth function to determine depth map
            depth_vector = depth_interpolator(coords_vector)
            depth = depth_vector.reshape((Ny, Nx))

            # TODO actually heterogeneous (though with very little differences)
            # Porosity
            porosity = self.config["physical_asset"]["parameters"]["porosity"]

            # Compute effective volume per porous voxel
            self.effective_volumes = porosity * width * height * depth / (Nx * Ny * Nz)

            np.save("volumes.npy", self.effective_volumes)

        else:
            self.effective_volumes = np.load("volumes.npy")

    # ! ---- Auxiliary calibration routines

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
            drift_correction=self.drift_correction,
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

        print(
            f"Image {str(path)} is considered, with rel. time {(self.img.timestamp - self.base.timestamp).total_seconds() / 3600.} hours."
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

    # ! ----- Concentration analysis

    def determine_concentration(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map - includes rescaling
        concentration = self.concentration_analysis(img)

        return concentration
