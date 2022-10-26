# TODO This is temporary, the respectve file in rigs should be used.
"""
Module containing the general setup for the large fluidflower rig. Valid for all runs. 
"""
from pathlib import Path
from typing import Union

import daria
import numpy as np
from scipy.interpolate import RBFInterpolator


class LargeFluidFlower(daria.AnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for well test analysis operating tailored to the well test
        included in the FluidFlower benchmark.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        daria.AnalysisBase.__init__(self, baseline, config, update_setup)

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry()

        # Determine effective volumes, required for calibration, determining total mass etc.
        self._determine_effective_volumes()

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update: bool = False) -> None:
        """
        Use watershed segmentation and some cleaning to segment
        the geometry. Note that not all sand layers are detected
        by this approach.

        Args:
            update (bool): flag whether segmentation has to be updated
                independent whether there already exists a file for provided
                path.
        """

        # Fetch or generate and store labels
        if not update and Path(self.config["segmentation"]["labels"]).exists():
            labels = np.load(self.config["segmentation"]["labels"])
        else:
            labels = daria.segment(self.base.img, **self.config["segmentation"])
            np.save(self.config["segmentation"]["labels"], labels)

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
