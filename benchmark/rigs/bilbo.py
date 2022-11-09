"""
Module containing the general setup for the Bilbo fluidflower rig.
Includes segmentation and depth map.
"""
from pathlib import Path
from typing import Union

import darsia
import numpy as np
from scipy.interpolate import RBFInterpolator


class Bilbo(darsia.AnalysisBase):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
    ) -> None:
        """
        Constructor for medium rig (Bilbo) specific data.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
        """
        darsia.AnalysisBase.__init__(self, baseline, config, update_setup)

        # Segment the baseline image; identidy water and esf layer.
        self._segment_geometry(update_setup=update_setup)

        # Determine effective volumes, required for calibration, determining total mass etc.
        # TODO has to be included?
        # self._determine_effective_volumes()

    # ! ---- Auxiliary setup routines

    def _segment_geometry(self, update_setup: bool = False) -> None:
        """
        Use watershed segmentation and some cleaning to segment
        the geometry. Note that not all sand layers are detected
        by this approach.

        Args:
            update_setup (bool): flag controlling whether the segmentation
                is performed even if a reference file exists; default is False.
        """

        # Fetch or generate and store labels
        if (
            Path(self.config["segmentation"]["labels_path"]).exists()
            and not update_setup
        ):
            labels = np.load(self.config["segmentation"]["labels_path"])
        else:
            labels = darsia.segment(
                self.base.img,
                markers_method="supervised",
                edges_method="scharr",
                verbosity=False,
                **self.config["segmentation"]
            )
            np.save(self.config["segmentation"]["labels_path"], labels)

        # Hardcoded: Identify water layer / Color checker (valid for BC02)
        self.water = np.zeros(labels.shape[:2], dtype=bool)
        for i in [0, 2]:
            self.water = np.logical_or(self.water, labels == i)

        # Hardcoded: Identify ESF layer with ids 2, 11 (valid for BC02)
        self.esf = np.zeros(labels.shape[:2], dtype=bool)
        for i in [2, 11]:
            self.esf = np.logical_or(self.esf, labels == i)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(self.base.img)
        plt.imshow(labels, alpha=0.3)
        plt.show()

    def _determine_effective_volumes(self) -> None:
        """
        Compute constant voxel volume, based on the reported depth map.
        Use constant porosity for now; this should be changed, when a proper
        segmentation is provided.
        """
        raise NotImplementedError("Effective volumes not known for Bilbo.")
