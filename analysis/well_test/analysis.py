"""
Analysis of well test for the FluidFlower Benchmark.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from benchmark.standardsetups.largerigtraceranalysis import \
    LargeRigTracerAnalysis

# ! ---- Data

folder = Path("/media/jakub/Elements/Jakub/benchmark/data/well_test")
pt1 = Path(
    "211002 well tests"
)  # contains baseline: 4; phase 1.a: 5:35; break: 35:65; 1b: 65:95; break: 95:125; 1c: 125:155; break:155:185; phase 2.a 185:215; break: 215:245; phase 2b 245:275; break 275:305; phase 2c: 305:335; break: 335:end
pt2 = Path("211002 well tests_overnight")
pt3 = Path("211003 well tests continued_phase_3")
images_folder = folder / pt3
file_ending = "*.JPG"
config = "./config.json"

images_1 = list(
    sorted((folder / pt1).glob(file_ending))
)  # contains entire phase 1 and 2 including breaks
images_2 = list(
    sorted((folder / pt2).glob(file_ending))
)  # contains overnight break between phase 2 and phase 3
images_3 = list(sorted((folder / pt3).glob(file_ending)))  # contains phase 3
images = list(sorted((folder / pt3).glob(file_ending)))
baseline = list(sorted((folder / Path("baseline")).glob(file_ending)))[:20]

# ! ---- Analysis object

# Define the tracer analysis object
tracer_analysis = LargeRigTracerAnalysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results="results-development",
    update_setup=False,  # flag controlling whether aux. data needs update
    verbosity=False,  # print intermediate results to screen
)

# ! ---- Balancing calibration

calibration_images_balancing = images_1[355:363]  # *445.jpg-*474.jpg
# tracer_analysis.calibrate_balancing(calibration_images_balancing)

# ! ---- Model calibration

# Calibrate the segmentation scaling in two steps - account for
# jumps in discontinuities, as well as - use images which cover sufficient
# areas of the geoemtry and result in connected labels after all.
# Phase 1c - steady injection (only one plume)
calibration_images_model = images_1[125:155]  # *260.jpg-*259.jpg

# Phase 2c - steady injection (two plumes)
# calibration_images_model = images_1[340:370] # *445.jpg-*474.jpg
# calibration_images_model = images_1[355:363:2] # *445.jpg-*474.jpg
# calibration_images_model = images_1[355:363]  # *445.jpg-*474.jpg
# tracer_analysis.calibrate_model(calibration_images_model)

# ! ---- Analysis

# random_indices_1 = np.unique((np.random.rand(5) * len(images_1)).astype(np.int32))
# random_images_1 = [images_1[i] for i in random_indices_1]
random_images_1 = images_1[::10]

# Perform batch analysis for the entire images folder
tracer_analysis.batch_analysis(
    images=random_images_1,
    plot_concentration=False,
    write_data_to_file=False,
)

print(tracer_analysis.times)
print(tracer_analysis.vols)

plt.plot(tracer_analysis.times, tracer_analysis.vols)
plt.show()
