"""
Script for the analysis of quantity 5 in the benchmark:

    Integral over box C of the L1 norm of the gradient
    of the water concentration scaled by the inverse of
    the maximum concentration value.

The approach used here assumes constant water concentration
in the CO2 water zone. Using the segmentation of the domain
into water, CO2(w) and CO2(g), identifies
"""

from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from benchmark.rigs.largefluidflower import LargeFluidFlower
from benchmark.utils.misc import read_time_from_path

# ! ---- User controls

runs = ["c1", "c2", "c3", "c4", "c5"]

# Just plot the data from file.

rel_concentration_gradient_integral_1 = np.load(f"results/c1_qty_5.npy")
rel_concentration_gradient_integral_2 = np.load(f"results/c2_qty_5.npy")
rel_concentration_gradient_integral_3 = np.load(f"results/c3_qty_5.npy")
rel_concentration_gradient_integral_4 = np.load(f"results/c4_qty_5.npy")
rel_concentration_gradient_integral_5 = np.load(f"results/c5_qty_5.npy")

rel_time_1 = np.load(f"results/c1_time.npy")
rel_time_2 = np.load(f"results/c2_time.npy")
rel_time_3 = np.load(f"results/c3_time.npy")
rel_time_4 = np.load(f"results/c4_time.npy")
rel_time_5 = np.load(f"results/c5_time.npy")

# Unify times
mask_1 = np.ones(123, dtype=bool)
mask_2 = np.ones(123, dtype=bool)
mask_3 = np.ones(123, dtype=bool)
mask_4 = np.ones(124, dtype=bool)
mask_5 = np.ones(120, dtype=bool)

mask_1[68:72] = False
mask_2[68:72] = False
mask_3[68:72] = False
mask_4[68:72] = False
mask_4[-1] = False
mask_5[-1] = False

rel_time_1 = rel_time_1[mask_1]
rel_time_2 = rel_time_2[mask_2]
rel_time_3 = rel_time_3[mask_3]
rel_time_4 = rel_time_4[mask_4]
rel_time_5 = rel_time_5[mask_5]

# Choose unique time
time = np.copy(rel_time_1)

# Apply same unification to qty5
rel_concentration_gradient_integral_1 = rel_concentration_gradient_integral_1[mask_1]
rel_concentration_gradient_integral_2 = rel_concentration_gradient_integral_2[mask_2]
rel_concentration_gradient_integral_3 = rel_concentration_gradient_integral_3[mask_3]
rel_concentration_gradient_integral_4 = rel_concentration_gradient_integral_4[mask_4]
rel_concentration_gradient_integral_5 = rel_concentration_gradient_integral_5[mask_5]

# Make relative qty absolute
len_box_C = 1.5
abs_concentration_gradient_integral_1 = (
    rel_concentration_gradient_integral_1 * len_box_C
)
abs_concentration_gradient_integral_2 = (
    rel_concentration_gradient_integral_2 * len_box_C
)
abs_concentration_gradient_integral_3 = (
    rel_concentration_gradient_integral_3 * len_box_C
)
abs_concentration_gradient_integral_4 = (
    rel_concentration_gradient_integral_4 * len_box_C
)
abs_concentration_gradient_integral_5 = (
    rel_concentration_gradient_integral_5 * len_box_C
)

# Make plot.
plt.figure("QTY5")
plt.plot(time, abs_concentration_gradient_integral_1, label="c1")
plt.plot(time, abs_concentration_gradient_integral_2, label="c2")
plt.plot(time, abs_concentration_gradient_integral_3, label="c3")
plt.plot(time, abs_concentration_gradient_integral_4, label="c4")
plt.plot(time, abs_concentration_gradient_integral_5, label="c5")
plt.legend()

plt.figure("QTY5 / length box C")
plt.plot(time, rel_concentration_gradient_integral_1, label="c1")
plt.plot(time, rel_concentration_gradient_integral_2, label="c2")
plt.plot(time, rel_concentration_gradient_integral_3, label="c3")
plt.plot(time, rel_concentration_gradient_integral_4, label="c4")
plt.plot(time, rel_concentration_gradient_integral_5, label="c5")
plt.plot(time, np.ones(len(time)), linestyle="--", c="0", label="100%")
plt.plot(time, 1.1 * np.ones(len(time)), linestyle="--", c="0", label="110%")
plt.legend()

plt.show()

# Store as csv file.
data = np.zeros((len(time), 12), dtype=float)
data[:, 0] = time.copy()
data[:, 1] = time.copy() * 60
data[:, 2] = abs_concentration_gradient_integral_1
data[:, 3] = abs_concentration_gradient_integral_2
data[:, 4] = abs_concentration_gradient_integral_3
data[:, 5] = abs_concentration_gradient_integral_4
data[:, 6] = abs_concentration_gradient_integral_5
data[:, 7] = rel_concentration_gradient_integral_1
data[:, 8] = rel_concentration_gradient_integral_2
data[:, 9] = rel_concentration_gradient_integral_3
data[:, 10] = rel_concentration_gradient_integral_4
data[:, 11] = rel_concentration_gradient_integral_5

# Store to file
qty5_header = f"Time in minutes, time in seconds, qty 5 for C1, C2, C3, C4, C5, and qty5 divided by 1.5(= length of box C) for C1, C2, C3, C4, C5"
fmt = (
    "%d",
    "%d",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
    "%f",
)
np.savetxt(
    "convective_mixing_qty5.csv", data, fmt=fmt, delimiter=",", header=qty5_header
)

# Determine lower and upper bounds for when qty5 reaches 110% of length of C.
# As upper bound, choose the value provided by the computations.
# As lower bound use 100%.
time_110_1 = 215
time_110_2 = 235
time_110_3 = 248
time_110_4 = 257
time_110_5 = 288

# In seconds
time_110_1 = time_110_1 * 60
time_110_2 = time_110_2 * 60
time_110_3 = time_110_3 * 60
time_110_4 = time_110_4 * 60
time_110_5 = time_110_5 * 60

print("QTY5 (110%)- 1", time_110_1)
print("QTY5 (110%)- 2", time_110_2)
print("QTY5 (110%)- 3", time_110_3)
print("QTY5 (110%)- 4", time_110_4)
print("QTY5 (110%)- 5", time_110_5)

# Statistics
times_110 = np.array([time_110_1, time_110_2, time_110_3, time_110_4, time_110_5])
times_110_mean = np.mean(times_110)
times_110_std = np.std(times_110)
times_110_min = np.min(times_110)
times_110_max = np.max(times_110)

print(times_110_mean)
print(times_110_std)
print(times_110_min)
print(times_110_max)

# Times for 100%
time_100_1 = 204
time_100_2 = 207
time_100_3 = 214
time_100_4 = 221
time_100_5 = 211

# in seconds

time_100_1 = time_100_1 * 60
time_100_2 = time_100_2 * 60
time_100_3 = time_100_3 * 60
time_100_4 = time_100_4 * 60
time_100_5 = time_100_5 * 60

print("QTY5 (100%) - 1", time_100_1)
print("QTY5 (100%) - 2", time_100_2)
print("QTY5 (100%) - 3", time_100_3)
print("QTY5 (100%) - 4", time_100_4)
print("QTY5 (100%) - 5", time_100_5)

# Statistics
times_100 = np.array([time_100_1, time_100_2, time_100_3, time_100_4, time_100_5])
times_100_mean = np.mean(times_100)
times_100_std = np.std(times_100)
times_100_min = np.min(times_100)
times_100_max = np.max(times_100)

print(times_100_mean)
print(times_100_std)
print(times_100_min)
print(times_100_max)
