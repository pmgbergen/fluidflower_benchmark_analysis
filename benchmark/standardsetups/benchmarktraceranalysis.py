"""
Module containing the standardized tracer concentration analysis applicable
for the well test performed in the large FluidFlower.
"""
import time
from pathlib import Path
from typing import Union

import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower

# from benchmark.utils.misc import read_time_from_path


class TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def __init__(
        self,
        base: Union[daria.Image, list[daria.Image]],
        color: Union[str, callable] = "gray",
        **kwargs,
    ) -> None:
        super().__init__(base, color)

        # TODO include in config
        self.disk_radius = kwargs.pop("median_disk_radius", 20)

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:

        # TODO try median as well
        # signal = skimage.restoration.denoise_tv_chambolle(
        #    signal, weight=20, eps=1e-4, max_num_iter=100
        # )
        signal = skimage.filters.rank.median(
            signal, skimage.morphology.disk(self.disk_radius)
        )
        signal = skimage.img_as_float(signal)

        return super().postprocess_signal(signal)


class BenchmarkTracerAnalysis(LargeFluidFlower, daria.TracerAnalysis):
    """
    Class for managing the well test of the FluidFlower benchmark.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        update_setup: bool = False,
        verbosity: bool = False,
    ) -> None:
        """
        Constructor for tracer analysis tailored to the benchmark
        geometry in the large FluidFlower.

        Sets up fixed config file required for preprocessing.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the post-analysis
                are printed to screen; default is False.
        """
        LargeFluidFlower.__init__(self, baseline, config, update_setup)
        daria.TracerAnalysis.__init__(self, baseline, config, update_setup)

        # The above constructors provide access to the config via self.config.
        # Determine the injection start from the config file. Expect format
        # complying with "%y%m%d %H%M%D", e.g., "211127 083412"
        # TODO as part of the calibration, this will be returned.
        # self.injection_start: datetime = datetime.strptime(
        #    self.config["injection_start"], "%y%m%d %H%M%S"
        # )
        # TODO temporarily...
        self.injection_start = self.base.timestamp

        # Initialize results dictionary for post-analysis
        self.results: dict = {}

        # Create folder for results if not existent
        self.path_to_results: Path = Path(self.config.get("results_path", "./results"))
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity
        self.verbosity = verbosity

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> daria.ConcentrationAnalysis:
        """
        Identify tracer concentration using a reduction to the garyscale space
        """
        tracer_analysis = TailoredConcentrationAnalysis(
            self.base,
            color="gray",
            **self.config["tracer"],
        )

        return tracer_analysis

    def determine_tracer(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.
        Add expert knowledge, that there is no tracer in the water.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Extract concentration from the analysis
        tracer_concentration = super().determine_tracer()

        # Add expert knowledge: Turn of any signal in the water zone
        tracer_concentration.img[self.water] = 0

        return tracer_concentration

    # ! ----- Analysis tools

    def single_image_analysis(self, img: Path, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Standard workflow to analyze the tracer concentration.

        Args:
            image (Path): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: tracer concentration map
            dict: dictinary with all stored results from the post-analysis.
        """
        # Load the current image
        self.load_and_process_image(img)

        # Determine tracer concentration
        tracer = self.determine_tracer()

        # ! ---- Post-analysis

        # Define some general data first:
        # Crop folder and ending from path - required for writing to file.
        img_id = Path(img.name).with_suffix("")

        # Determine the time increment (in terms of hours),
        # referring to injection start, in hours.
        SECONDS_TO_HOURS = 1.0 / 3600
        relative_time = (
            self.img.timestamp - self.injection_start
        ).total_seconds() * SECONDS_TO_HOURS

        # Plot and store image with contours
        plot_concentration = kwargs.pop("plot_concentration", False)
        write_concentration_to_file = kwargs.pop("write_concentration_to_file", False)

        if plot_concentration or write_concentration_to_file:

            # Plot
            if plot_concentration:
                plt.figure("Tracer concentration")
                plt.imshow(tracer.img)
                plt.show()

            # Write to file
            if write_concentration_to_file:
                cv2.imwrite(
                    str(self.path_to_results / Path(f"{img_id}_concentration.jpg")),
                    cv2.cvtColor(tracer.img, cv2.COLOR_RGB2BGR),
                )

        return tracer, self.results

    # TODO add to baseanalysis?

    def batch_analysis(self, images: list[Path], **kwargs) -> dict:
        """
        Standard batch analysis for the well test performed for the benchmark.

        Args:
            images (list of Path): paths to batch of images.
            kwargs: optional keyword arguments:
                plot_concentration (bool): flag controlling whether the concentration
                    profile is plotted; default False.
                write_concentration_to_file (bool): flag controlling whether the plot from
                    plot_concentration is written to file; default False.

        Returns:
            dict: dictinary with all stored results from the post-analysis.
        """

        for num, img in enumerate(images):

            tic = time.time()

            # Perform dedicated analysis for the current image
            self.single_image_analysis(img, **kwargs)

            # Information to the user
            if self.verbosity:
                print(f"Elapsed time for {img.name}: {time.time()- tic}.")

        return self.results

    def return_results(self) -> dict:
        """
        Return all results collected throughout any analysis performed.
        """
        # TODO restrict to specific list of keywords.

        return self.results

    def write_results_to_file(self, folder: Path) -> None:
        """
        Write results in separate files.

        Args:
            folder (Path): folder where the results are stored.
        """
        for keys in self.results.keys():
            pass
