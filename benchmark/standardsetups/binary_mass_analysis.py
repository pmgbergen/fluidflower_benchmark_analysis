"""
A class with utilities for computing mass based on segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from warnings import warn

import darsia as da
import numpy as np
from scipy.interpolate import RBFInterpolator


class BinaryMassAnalysis:
    """
    Class for for analysing mass of components in 2D segmented images

    Attributes:
        base (darsia.Image): A base darsia image. In principle only needed for size
            properties, like numer of pixels, height and width.
        pixelarea (float): product of dx and dy from base.
        porosity (np.ndarray): porosity map of equal size to base. Default is array
            of ones.
        depth_map (np.ndarray): depth map contain information about depth associated to
            each pixel. Default is an array of ones.
        height_map (np.ndarray): Associates each pixel with its distance from the top of
            the image.
    """

    def __init__(
        self,
        base: da.Image,
        depth_map: Optional[np.ndarray] = None,
        depth_measurements: Optional[tuple[np.ndarray, ...]] = None,
        porosity: Optional[Union[np.ndarray, float]] = None,
        cache: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initializer for mass analysis class.

        Arguments:
            base (darsia.Image): In principle only needed for size properties,
                like numer of pixels, height and width.
            depth_map (Optional[np.ndarray]): depth map with depth associated to
                    each pixel.
            depth_measurements (Optional[tuple[np.ndarray, ...]]): tuple of depth
                    measurements. Should be provided as horizontal coordinates and
                    vertical coordinates as a meshgrid, and depth cooridinates
                    corresponding to each horizontal and vertical entry.
            porosity (Optional[Union[np.ndarray, float]]): porosity map. Can be
                    provided as a uniform scalar value as well.
            cache (Optional[Union[str, Path]]): folder for caching auxiliary results.

        """
        self.base = base
        self.pixelarea: float = self.base.dx * self.base.dy

        # Define porosity map
        if porosity is not None:
            if isinstance(porosity, np.ndarray):
                self.porosity: np.ndarray = porosity
            else:
                self.porosity = porosity * np.ones(self.base.img.shape[:2])
        else:
            self.porosity = np.ones(self.base.img.shape[:2])
            warn("Please provide a porosity. Now it is assumed to be 1.")

        # Define depth map
        if depth_measurements is not None:
            if cache is not None and Path(cache).exists():
                self.depth_map = np.load(Path(cache))
            else:
                self.depth_map = self._compute_depth_map(depth_measurements)
                if cache is not None:
                    Path(cache).parents[0].mkdir(parents=True, exist_ok=True)
                    np.save(Path(cache), self.depth_map)
        elif depth_map is not None:
            self.depth_map = depth_map
        else:
            warn(
                "Please provide a depth_map (of at least the same size as "
                " the segmented images that are to be analyzed) or a tuple of "
                " depth measurements."
                " Now, a constant depth of 1 is assumed."
            )
            self.depth_map = np.ones(self.base.img.shape[:2])

        # Defines a height map associating each pixel with its physical
        # height (distance from top)
        self.height_map = np.linspace(0, self.base.height, self.base.img.shape[0])[
            :, None
        ] * np.ones_like(self.base.img)

    def volume_map(
        self,
        segmentation: np.ndarray,
        component: int,
        roi: Optional[Union[np.ndarray, list]] = None,
    ) -> np.ndarray:
        """
        Given segmentation and component, returns a volume map of the component
        from the segmentation where each pixel gives the volume in meters^3 associated
        to that pixel. If there is no compontent found at a pixel this map takes the value 0.

        Arguments:
            segmentation (np.ndarray): segmented image
            component (int): the value that the component of interest takes in the
                segmentation.
            roi (Optional[Union[list, np.ndarray]]): roi where free CO2 should be computed.
                Provided as lower left corner, upper right corner in [x,y] coordinates

        Returns:
            volume map (np.ndarray): volume associated to each pixel containing the prescribed
                component.
        """

        # Check that the depth map has the correct size.
        if self.depth_map.shape != segmentation.shape:
            raise Exception(
                f"The shape of the depthmap is {self.depth_map.shape} and the shape of the"
                f" segmentations is {segmentation.shape}."
                " Please make sure that the depthmap is at least as large as the"
                " segmentation. For optimal accuracy they should be equal in size."
            )
        else:
            # Compute volume map
            volume_map = (self.porosity * self.depth_map) * self.pixelarea

        # Extract only the correct volumes (depending on segmentation and chosen component)
        volume = np.zeros_like(volume_map)
        volume[segmentation == component] = volume_map[segmentation == component]

        if roi is not None:
            _, roi_box = da.extractROI(self.base, roi, return_roi=True)
            volume = volume[roi_box]

        return volume

    def volume(
        self,
        segmentation: np.ndarray,
        component: int,
        roi: Optional[Union[np.ndarray, list]] = None,
    ) -> float:
        """
        Given segmentation and component, returns the total volume of the component
        from the segmentation where each pixel gives the volume in meters^3 associated
        to that pixel. If there is no compontent found at a pixel this map takes the value 0.

        Arguments:
            segmentation (np.ndarray): segmented image
            component (int): the value that the component of interest takes in the
                segmentation.
            roi (Optional[Union[list, np.ndarray]]): roi where free CO2 should be computed.
                Provided as lower left corner, upper right corner in [x,y] coordinates

        Returns:
            volume (float): total volume containing the prescribed component.
        """
        return np.sum(self.volume_map(segmentation, component, roi))

    def external_pressure_to_density_co2(
        self, external_pressure: float, roi: Optional[Union[np.ndarray, list]] = None
    ) -> np.ndarray:
        """
        A conversion from pressure (in bar) to density of CO2 (in g/m^3)
        is given by the linear formula here. The formula is found by using
        linear regression on table values from NIST for standard conversion
        between pressure and density of CO2 at 23 Celsius. The extra added
        pressure due depth in the water is also corrected for, using the
        formula 0.1atm per meter, which amounts to 0.101325bar per meter.

        Arguments:
            external_pressure (float): external pressure, for example atmospheric pressure.
            roi (Optional[Union[list, np.ndarray]]): roi where free CO2 should be computed.
                Provided as lower left corner, upper right corner in [x,y] coordinates

        Returns:
            (np.ndarray): array associating each pixel with the CO2 density.
        """
        if roi is not None:
            _, roi_box = da.extractROI(self.base, roi, return_roi=True)
            height_map = self.height_map[roi_box]
        else:
            height_map = self.height_map
        return 1000 * (
            1.805726990977443 * (external_pressure + 0.101325 * height_map)
            - 0.009218969932330845
        )

    def free_co2_mass(
        self,
        segmentation: np.ndarray,
        external_pressure: float,
        co2_component: int = 2,
        roi: Optional[Union[list, np.ndarray]] = None,
    ) -> float:
        """
        Given a segmentation, external pressure and the number
        associated to free CO2 (default is 2), it returns the
        mass of free CO2 in grams.

        Arguments:
            segmentation (np.ndarray): segmented image
            external_pressure (float): external pressure, for example atmospheric pressure.
            co2_component (int): value of the component in segmentation that
                corresponds to co2.
            roi (Optional[Union[list, np.ndarray]]): roi where free CO2 should be computed.
                Provided as lower left corner, upper right corner in [x,y] coordinates

        Returns:
            (float): Mass of co2.
        """
        if roi is not None:
            volume = self.volume_map(segmentation, co2_component, roi)
            density_map = self.external_pressure_to_density_co2(external_pressure, roi)
        else:
            volume = self.volume_map(segmentation, co2_component)
            density_map = self.external_pressure_to_density_co2(external_pressure)
        return np.sum(volume * density_map)

    def _compute_depth_map(
        self, depth_measurements: tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """
        Compute depth map, based on the reported measurements.

        Arguments:
             depth_measurements (Optional[tuple[np.ndarray, ...]]): tuple of depth
                    measurements. Should be provided as horizontal coordinates,
                    vertical coordinates, and depth cooridinates corresponding
                    to each horizontal and vertical entry.
        Returns:
            (np.ndarray): depth map
        """

        # Determine number of voxels in each dimension - assume 2d image
        Ny, Nx = self.base.img.shape[:2]
        x = np.arange(Nx)
        y = np.arange(Ny)
        X_pixel, Y_pixel = np.meshgrid(x, y)
        pixel_vector = np.transpose(np.vstack((np.ravel(Y_pixel), np.ravel(X_pixel))))
        coords_vector = self.base.coordinatesystem.pixelToCoordinate(pixel_vector)
        # Fetch physical dimensions
        # Depth of the rig, measured in discrete points, taking into account expansion.
        # Values taken from the benchmark description.
        # Coordinates at which depth measurements have been taken.
        # Note that the y-coordinate differs depending on the x-coordinate,
        # which dissallows use of np.meshgrid. Instead, the meshgrid is
        # constructed by hand.
        depth_interpolator = RBFInterpolator(
            np.transpose(
                np.vstack(
                    (
                        depth_measurements[0],
                        depth_measurements[1],
                    )
                )
            ),
            depth_measurements[2],
        )
        # Evaluate depth function to determine depth map
        depth_vector = depth_interpolator(coords_vector)
        return depth_vector.reshape((Ny, Nx))
