"""
Module containing the standardized CO2 analysis applicable for the table top rig.

"""
from pathlib import Path
from typing import Union

import numpy as np
from benchmark.rigs.tabletopfluidflower import TableTopFluidFlower

from .fluidflowerco2analysis import FluidFlowerCO2Analysis


class TableTopFluidFlowerCO2Analysis(TableTopFluidFlower, FluidFlowerCO2Analysis):
    """
    Class for managing the FluidFlower benchmark analysis.
    Identifiers for CO2 dissolved in water and CO2(g) are
    tailored to the benchmark set of the medium rig.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = True,
    ) -> None:
        TableTopFluidFlower.__init__(self, baseline, config, update_setup)
        FluidFlowerCO2Analysis.__init__(
            self, baseline, config, results, update_setup, verbosity
        )

    def _expert_knowledge_co2(self) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2.

        Returns:
            np.ndarray: mask with no CO2.

        """
        # Add expert knowledge. Turn of any signal in the water zone.
        return np.logical_not(self.extended_water)

    def _expert_knowledge_co2_gas(self, co2) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2(g).

        Args:
            co2 (darsia.Image): mask for CO2.

        Returns:
            np.ndarray: mask with no CO2(g)

        """
        # Add expert knowledge - do not expect any CO2(g) outside
        # of the CO2, and neither in ESF nor C. Include this in
        # the analysis.
        expert_knowledge = np.logical_and(
            co2.img, np.logical_not(np.logical_or(self.esf_sand, self.c_sand))
        )
        # Also exclude CO2 from the bottom of the rig - mostly due to shaking images.
        expert_knowledge = np.logical_and(
            expert_knowledge, np.logical_not(self.bottom_zone)
        )
        return expert_knowledge
