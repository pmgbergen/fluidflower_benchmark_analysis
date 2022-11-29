"""
Analysis of well test for the FluidFlower Benchmark.
"""
from pathlib import Path

from benchmark.standardsetups.benchmarktraceranalysis import \
    BenchmarkTracerAnalysis

folder = Path("/media/jakub/Elements/Jakub/benchmark/data/well_test")
pt1 = Path("211002 well tests")
pt2 = Path("211002 well tests_overnight")
pt3 = Path("211003 well tests continued_1")
images_folder = folder / pt3
file_ending = "*.JPG"
config = "./config.json"

images = list(sorted((folder / pt3).glob(file_ending)))
baseline = list(sorted((folder / Path("baseline")).glob(file_ending)))[:20]

# Extract concentration.
images = [
    images[-1],
    images[0],
    images[10],
    images[20],
    images[30],
]

# Completely flushed rig for calibration
# img_flushed = folder / Path("220202_well_test") / Path("DSC04224.JPG")

tracer_analysis = BenchmarkTracerAnalysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=False,  # print intermediate results to screen
)

# Perform batch analysis for the entire images folder
tracer_analysis.batch_analysis(
    images=images,
    plot_concentration=True,
)
