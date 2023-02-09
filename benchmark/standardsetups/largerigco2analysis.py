"""
Module containing the standardized CO2 analysis applicable for C1, ..., C5
of the International Benchmark initiative.

"""
from pathlib import Path
from typing import Union

import numpy as np
from benchmark.rigs.largefluidflower import LargeFluidFlower

from .fluidflowerco2analysis import FluidFlowerCO2Analysis


class LargeRigCO2Analysis(LargeFluidFlower, FluidFlowerCO2Analysis):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = True,
    ) -> None:
        """
        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the
                post-analysis are printed to screen; default is False.
        """
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        FluidFlowerCO2Analysis.__init__(
            self, baseline, config, results, update_setup, verbosity
        )

    def _expert_knowledge_co2_gas(self, co2) -> np.ndarray:
        """
        Retrieve expert knowledge, i.e., areas with possibility for CO2(g).

        Args:
            co2 (darsia.Image): mask for CO2.

        Returns:
            np.ndarray: mask with no CO2(g)

        """
        return np.logical_and(co2.img, np.logical_not(self.esf_sand))
