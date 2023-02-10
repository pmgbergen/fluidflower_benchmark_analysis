"""
Module computing the depth of the FluidFlower rig based on
the interpolation of the measured depths.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

# Coordinates at which depth measurements have been taken.
# Note that the y-coordinate differs depending on the x-coordinate,
# which dissallows use of np.meshgrid. Instead, the meshgrid is
# constructed by hand.
x_measurements = np.array(
    [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
    ]
    * 7
)
y_measurements = np.array(
    [0.0] * 15
    + [0.2] * 15
    + [0.4] * 15
    + [0.6] * 4
    + [0.61, 0.63]
    + [0.64] * 6
    + [0.6] * 3
    + [0.8] * 15
    + [1.0] * 15
    + [1.2] * 15
)

# Measurements in mm, including the thickness of the equipment used to measure
# the depth (1.5 mm).
depth_measurements = np.array(
    [
        20.72,
        20.7,
        20.9,
        21.2,
        21.4,
        21.3,
        21.3,
        20.5,
        20.1,
        21.1,
        20.8,
        20.0,
        19.4,
        19,
        20,
        20.43,
        23.1,
        24.4,
        25.8,
        25.7,
        25.8,
        25.6,
        25.1,
        25,
        25.6,
        25.3,
        23.8,
        21.7,
        19.7,
        20.2,
        20.9,
        23.8,
        27.1,
        28.9,
        27.9,
        28.4,
        27.8,
        27.1,
        27.3,
        27.4,
        27.6,
        25.6,
        22.7,
        20.5,
        20,
        20.7,
        24.5,
        28,
        29.3,
        29,
        29.6,
        29.1,
        27.9,
        28.6,
        28.4,
        28.1,
        26.9,
        23.2,
        20.9,
        19.7,
        20.7,
        23.6,
        27,
        28.8,
        29.6,
        29.8,
        28.5,
        27.7,
        28.7,
        28.9,
        27.5,
        27.5,
        22.7,
        20.7,
        20,
        20.7,
        22.4,
        25.3,
        27.8,
        29.3,
        29.2,
        28.4,
        27,
        28,
        28.4,
        26.8,
        26,
        22.4,
        19.9,
        20,
        20.7,
        21.5,
        24.2,
        26.3,
        27.2,
        27.4,
        27.5,
        26,
        26.8,
        27.7,
        26.8,
        25.2,
        22.4,
        19.9,
        20,
    ]
)

# Correct for thickness of measurement equipment
depth_measurements -= 1.5

# Convert depth to meters
depth_measurements *= 1e-3

depth_interpolator = RBFInterpolator(
    np.transpose(
        np.vstack(
            (
                x_measurements,
                y_measurements,
            )
        )
    ),
    depth_measurements,
)

# Evaluate depth function on 1cm x 1cm grid (2.8 m x 1.5 m) to determine depth map

# Fetch physical dimensions
width = 2.8
height = 1.5

# Determine number of voxels in each dimension - assume 2d image
Nx = 280
Ny = 150

dx = 0.01
dy = 0.01

# Define centers of pixels
x = np.linspace(0.5 * dx, width - 0.5 * dx, Nx)
y = np.flip(np.linspace(0.5 * dy, height - 0.5 * dy, Ny))
Y_coords, X_coords = np.meshgrid(y, x, indexing="ij")
coords_vector = np.transpose(np.vstack((np.ravel(X_coords), np.ravel(Y_coords))))

# Determine depth and shape to image format
depth_vector = depth_interpolator(coords_vector)
depth = depth_vector.reshape((Ny, Nx))

# Test the result
plt.figure("Interpolated depth measurements")
plt.imshow(depth)
plt.show()

# Store the result
np.save("depth.npy", depth)

# Build csv file with standard text on top.
header = f"Depth interoplated using RBF interpolation, based on updated measurements."

header += "\nEach entry identifies the phase distribution in a 1 cm by 1 cm cell, where"

header += "\n0 = pure water, 1 = water with dissolved CO2, 2 = gaseous CO2."

header += "\n2d array representing the laser grid of the benchmark description,"

header += "\nordered row-wise from top to bottom, and column-wise from left to right."

header += "\nThe first entry corresponds to the cell centered at coordinates (0.035 m, 1.525 m)"

header += "\nw.r.t. the spatial maps outlined in Section 3.1 of the description."

# Store array with preceding header
np.savetxt("./depth.csv", depth, fmt="%d", delimiter=",", header=header)
