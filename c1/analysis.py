"""
Analysis of FluidFlower Benchmark Run C1.
"""
from pathlib import Path

from fluidflower import BenchmarkRig

folder = Path("/home/jakub/images/ift/benchmark/c1")
baseline = Path("baseline")
processed = Path("processed")

# Define FluidFlower with first 10 baseline images
baseline_images = list(sorted((folder / baseline).glob("*.JPG")))[:10]
ff = BenchmarkRig(baseline_images, config_source="./config.json")

# Extract concentration.
images = list(sorted(folder.glob("*.JPG")))[30:32]
for img in images:
    ff.load_and_process_image(img)
    co2 = ff.determine_concentration()

    co2.plt_show()
