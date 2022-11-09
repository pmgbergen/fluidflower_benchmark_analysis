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
                **self.config["segmentation"]
            )
            labels_path = Path(self.config["segmentation"]["labels_path"])
            labels_path.parents[0].mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels)

        def _labels_to_mask(ids: list) -> np.ndarray:
            ids = ids if isinstance(ids, list) else [ids]
            mask = np.zeros(labels.shape[:2], dtype=bool)
            for i in ids:
                mask[labels==i] = True
            return mask

        # Hardcoded: Identify water layer / Color checker (valid for BC02)
        self.water = _labels_to_mask([0, 2])

        # Hardcoded: Define extended water zone, essentially upper zone
        # of the medium including some of the ESF sand, as strong
        # light fluctuations interfere with the top of the ESF layer.
        self.extended_water = np.zeros_like(self.water)
        self.extended_water[:450,:] = True

        # Hardcoded: Identify ESF layer (valid for BC02)
        self.esf_sand = _labels_to_mask([1, 10])

        # Hardcoded: Identify C sand layer (valid for BC02)
        self.c_sand = _labels_to_mask(4)

        self.labels = labels


    def _determine_effective_volumes(self) -> None:
        """
        Compute constant voxel volume, based on the reported depth map.
        Use constant porosity for now; this should be changed, when a proper
        segmentation is provided.
        """
        raise NotImplementedError("Effective volumes not known for Bilbo.")
