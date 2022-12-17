"""
Analysis of well test for the FluidFlower Benchmark.
"""
from pathlib import Path

from benchmark.standardsetups.benchmarktraceranalysis import \
    BenchmarkTracerAnalysis

folder = Path("/media/jakub/Elements/Jakub/benchmark/data/well_test")
pt1 = Path("211002 well tests") # contains baseline: 4; phase 1.a: 5:35; break: 35:65; 1b: 65:95; break: 95:125; 1c: 125:155; break:155:185; phase 2.a 185:215; break: 215:245; phase 2b 245:275; break 275:305; phase 2c: 305:335; break: 335:end
pt2 = Path("211002 well tests_overnight")
pt3 = Path("211003 well tests continued_phase_3")
images_folder = folder / pt3
file_ending = "*.JPG"
config = "./config.json"

images_1 = list(sorted((folder / pt1).glob(file_ending)))
images_2 = list(sorted((folder / pt2).glob(file_ending)))
images_3 = list(sorted((folder / pt3).glob(file_ending)))
images = list(sorted((folder / pt3).glob(file_ending)))
baseline = list(sorted((folder / Path("baseline")).glob(file_ending)))[:20]

# Define the tracer analysis object
tracer_analysis = BenchmarkTracerAnalysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=False,  # print intermediate results to screen
)

# Calibrate the segmentation scaling in two steps - account for
# jumps in discontinuities, as well as - use images which cover sufficient
# areas of the geoemtry and result in connected labels after all.
images1 = list(sorted((folder / pt1).glob(file_ending))) # contains entire phase 1 and 2 including breaks
images2 = list(sorted((folder / pt2).glob(file_ending))) # contains overnight break between phase 2 and phase 3
images3 = list(sorted((folder / pt3).glob(file_ending))) # contains phase 3

# Phase 1c - steady injection (only one plume)
#segmentation_scaling_calibration_images = images1[125:155] # *260.jpg-*259.jpg

# Phase 2c - steady injection (two plumes)
segmentation_scaling_calibration_images = images1[340:370] # *445.jpg-*474.jpg

# TODO include scaling into config!

# Perform both steps of calibration.
tracer_analysis.calibrate(segmentation_scaling_calibration_images, injection_rate=2250)

# Perform batch analysis for the entire images folder
#test_images = images1[340:370] # *445.jpg-*474.jpg
#test_images = images1[340:341] # *445.jpg-*474.jpg
tracer_analysis.batch_analysis(
    images=images_2,
    plot_concentration=False,
    write_data_to_file = True,
)
