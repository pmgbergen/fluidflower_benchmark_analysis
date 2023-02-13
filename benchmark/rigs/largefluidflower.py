"""
Module containing the general setup for the large fluidflower rig.
Includes segmentation and depth map.
"""
from pathlib import Path
from typing import Union

import darsia
import numpy as np
from darsia.presets.fluidflower.fluidflowerrig import FluidFlowerRig
from scipy.interpolate import RBFInterpolator


class LargeFluidFlower(FluidFlowerRig):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for large FluidFlower rig specific data.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        super().__init__(baseline, config, update_setup)

        # Define boxes A, B, C, relevant for the benchmark analysis
        self._define_boxes()

        # Determine effective volumes, required for calibration, determining total mass etc.
        # self._determine_effective_volumes()

    # ! ---- Auxiliary setup routines

    def _define_boxes(self) -> None:
        """
        The benchmark analysis has identfified three specific boxes: box A, box B and box C.
        This method fixes the definition of these boxes in terms of rois and masks.
        """
        # Box A, B, C in metric coorddinates (left top, and right lower point)
        self.box_A = np.array([[1.1, 0.6], [2.8, 0.0]])
        self.box_B = np.array([[0.0, 1.2], [1.1, 0.6]])
        self.box_C = np.array([[1.1, 0.4], [2.6, 0.1]])

        # Box A, B, C in terms of pixels, adapted to the size of the base image
        self.box_A_roi = self.base.coordinatesystem.coordinateToPixel(self.box_A)
        self.box_B_roi = self.base.coordinatesystem.coordinateToPixel(self.box_B)
        self.box_C_roi = self.base.coordinatesystem.coordinateToPixel(self.box_C)

        # Boolean masks for boxes A, B, C, adapted to the size of the base image
        self.mask_box_A = np.zeros(self.base.img.shape[:2], dtype=bool)
        self.mask_box_B = np.zeros(self.base.img.shape[:2], dtype=bool)
        self.mask_box_C = np.zeros(self.base.img.shape[:2], dtype=bool)

        self.mask_box_A[darsia.bounding_box(self.box_A_roi)] = True
        self.mask_box_B[darsia.bounding_box(self.box_B_roi)] = True
        self.mask_box_C[darsia.bounding_box(self.box_C_roi)] = True

    def _segment_geometry(self, update_setup: bool = False) -> None:
        """
        See SegmentedFluidFlower.

        """
        super()._segment_geometry(update_setup)

        # Identify water layer
        self.water = self._labels_to_mask(self.config["segmentation"]["water"])

        # Identify ESF layer
        self.esf_sand = self._labels_to_mask(self.config["segmentation"]["esf"])

        # Identify C layer
        self.c_sand = self._labels_to_mask(self.config["segmentation"]["c"])

        # TODO rm
        # self.component_groups = [[1,10,11], [2,3,4]]

        # self.label_reduced = np.ones_like(self.labels)
        # self.label_reduced[self.water] = 0
        # self.label_reduced[self.esf_sand] = 1
        # self.label_reduced[self.c_sand] = 2

        ## Create new labeled image
        # self.labels = np.zeros(labels.shape[:2], dtype=np.uint8)
        # self.labels_legend = {
        #    "water": 0,
        #    "esf_sand": 1,
        #    "c_sand": 2,
        #    "rest": 3,
        # }
        ## Initiate all elements with the default parameter
        # self.labels[:, :] = self.labels_legend["rest"]
        ## Overwrite all specific segments
        # self.labels[self.water] = self.labels_legend["water"]
        # self.labels[self.esf_sand] = self.labels_legend["esf_sand"]
        # self.labels[self.c_sand] = self.labels_legend["c_sand"]

    def _determine_effective_volumes(self) -> None:
        """
        Compute constant voxel volume, based on the reported depth map.
        Use constant porosity for now; this should be changed, when a proper
        segmentation is provided.
        """

        # Fetch volumes from file if existent, otherwise generate.
        path = Path(self.config["physical_asset"]["volumes_path"])
        if path.exists():
            self.effective_volumes = np.load(path)

        else:
            # Determine number of voxels in each dimension - assume 2d image
            Ny, Nx = self.base.img.shape[:2]
            Nz = 1
            x = np.arange(Nx)
            y = np.arange(Ny)
            X_pixel, Y_pixel = np.meshgrid(x, y)
            pixel_vector = np.transpose(
                np.vstack((np.ravel(Y_pixel), np.ravel(X_pixel)))
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

            # Store in cache
            np.save(path, self.effective_volumes)
