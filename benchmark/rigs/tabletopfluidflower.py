"""
Module containing the general setup for a table top fluidflower rig.
Includes segmentation and depth map.
"""
from pathlib import Path
from typing import Union

import darsia
import numpy as np
from darsia.presets.fluidflower.fluidflowerrig import FluidFlowerRig


class TableTopFluidFlower(FluidFlowerRig):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for table top rig specific data.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        super().__init__(baseline, config, update_setup)

        # Determine effective volumes, required for calibration, determining total mass etc.
        # self._determine_effective_volumes()

    # ! ---- Auxiliary setup routines

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

        # Hardcoded: Define extended water zone, essentially upper zone
        # of the medium including some of the ESF sand, as strong
        # light fluctuations interfere with the top of the ESF layer.
        self.extended_water = np.zeros_like(self.water)
        self.extended_water[:530, :] = True

        # Hardcoded: Define bottom zone, essentially lower zone of the
        # medium. Since images can move, strange effects, especially at
        # boundaries of images can be identified. To exclude them manually
        # nitrocude a hardcoded zone.
        self.bottom_zone = np.zeros_like(self.water)
        self.bottom_zone[-250:, :] = True

    def _determine_effective_volumes(self) -> None:
        """
        Compute constant voxel volume, based on the reported depth map.
        Use constant porosity for now; this should be changed, when a proper
        segmentation is provided.
        """
        raise NotImplementedError("Effective volumes not known for Bilbo or Albus.")
