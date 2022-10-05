"""
Module containing the setup for the fluidflower rig, used for the PoroTwin1 optimal control project.
"""
import json
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.interpolate import RBFInterpolator


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
        path_to_cleaning_filter: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Constructor for Benchmark rig.

        Sets up fixed config file required for preprocessing.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config_source (str or Path): path to config dict
            path_to_cleaning_filter (str or Path, optional):
        """
        # Read general config file
        f = open(config_source, "r")
        self.config = json.load(f)
        f.close()

        # Some hardcoded config data (incl. not JSON serializable data)
        roi = {
            "color": (slice(0, 600, None), slice(0, 600, None)),
        }

        # Define correction objects
        self.color_correction = daria.ColorCorrection(roi=roi["color"])
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Store the baseline image
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]
        self.base = self._read(reference_base)

        # Determine effective volumes, required for calibration, determining total mass etc.
        self._determine_effective_volumes()

        # Define concentration analysis. To speed up significantly the process,
        # invoke resizing of signals within the concentration analysis.
        # Also use pre-calibrated information.
        self.concentration_analysis = TailoredConcentrationAnalysis(
            self.base, color="gray", resize_factor=0.2
        )
        if path_to_cleaning_filter is not None:
            self.concentration_analysis.read_calibration_from_file(
                self.config["calibration"],
                path_to_cleaning_filter,
            )
        elif "calibration" in self.config:
            self._setup_concentration_analysis(self.config["calibration"], baseline)
        else:
            print("Warning: Concentration analysis is not calibrated.")
            self._setup_concentration_analysis({}, baseline)

        # Forward volumes to the concentration analysis, required for calibration
        self.concentration_analysis.update_volumes(self.effective_volumes)

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self) -> None:
        pass

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
        X_coords = coords_vector[:, 0].reshape((Ny, Nx))
        Y_coords = coords_vector[:, 1].reshape((Ny, Nx))

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
        self.effective_volume = porosity * width * height * depth / (Nx * Ny * Nz)

    def _setup_concentration_analysis(
        self, config: dict, baseline_images: list[Union[str, Path]]
    ) -> None:
        """
        Wrapper to find cleaning filter of the concentration analysis.

        Args:
            config (dict): dictionary with scaling parameters
            baseline_images (list of str or Path): paths to baseline images.
        """
        # TODO uncomment!

        ## Define scaling factor
        # if "scaling" in config:
        #    self.concentration_analysis.scaling = config["scaling"]

        ## Find cleaning filter
        # images = [self._read(path) for path in baseline_images]
        # self.concentration_analysis.find_cleaning_filter(images)

        pass

    # ! ---- Auxiliary calibration routines

    def calibrate_concentration_analysis(
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
        self.img = self._read(path)

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

    # ! ----- Concentration analysis

    def determine_concentration(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map
        concentration = self.concentration_analysis(img)

        return concentration
