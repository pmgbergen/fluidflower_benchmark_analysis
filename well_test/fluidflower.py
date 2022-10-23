"""
Module containing the setup to analyze the well test in the large FluidFlower.
"""
from pathlib import Path
from typing import Union

import daria
import numpy as np
import skimage

# TODO use ..rigs.largefluidflower here.
from largefluidflower import LargeFluidFlower


class TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:

        # TODO try median as well
        # signal = skimage.restoration.denoise_tv_chambolle(
        #    signal, weight=20, eps=1e-4, max_num_iter=100
        # )

        signal = skimage.filters.rank.median(signal, skimage.morphology.disk(50))
        signal = skimage.img_as_float(signal)

        return super().postprocess_signal(signal)


class WellTestAnalysis(LargeFluidFlower, daria.TracerAnalysis):
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
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        daria.TracerAnalysis.__init__(self, baseline, config, update_setup)

    # ! ----- Concentration analysis

    def define_concentration_analysis(self) -> None:
        """
        Define self.concentration_analysis.
        """
        self.concentration_analysis = TailoredConcentrationAnalysis(
            self.base,
            color="gray",
        )

    def determine_concentration(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.
        Add expert knowledge, that there is no tracer in the water.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Extract concentration from the analysis
        concentration = super().determine_concentration()

        # Add expert knowledge: Turn of any signal in the water zone
        concentration.img[self.water] = 0

        return concentration
