import os

import darsia as da
import matplotlib.pyplot as plt
import numpy as np
from binary_mass_analysis import BinaryMassAnalysis
from create_pressure_array import pressure_array
from scipy import interpolate
from time_from_image_name import ImageTime
from total_injected_mass_large_FF import total_mass

# Read atmospheric pressure data (found at https://veret.gfi.uib.no/?action=period_query)
# NOTE: The atmospheric pressures are as of now not corrected for elevation (as Jan commented upon in the email)
atmospheric_pressures = pressure_array()

# Create time array corresponding to the found pressure data
time_pressure = np.linspace(
    -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
)

# Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
pressure = interpolate.interp1d(time_pressure, 0.001 * atmospheric_pressures)

# Provide path to segmentations
dir = "C:/Users/Erlend/src/darsia/images/c1_segmentation/coarse_npy_segmentation"
list_dir = os.listdir(dir)

base_segmentation = da.Image(
    np.load(os.path.join(dir, list_dir[0])), width=2.8, height=1.5, color_space="GRAY"
)

# Provide path to depths (It is provided for the large FF in the "depths-folder")
meas_dir = "depths/"
depth_measurements = (
    np.load(meas_dir + "x_measures.npy"),
    np.load(meas_dir + "y_measures.npy"),
    np.load(meas_dir + "depth_measures.npy"),
)

# Create mass analysis object (the porosity could also be provided
# as an np.ndarray with varying porosities depending on the sand layers.)
mass_analysis = BinaryMassAnalysis(
    base_segmentation,
    depth_measurements=depth_measurements,
    porosity=0.44,
)

# Create empty lists for plotting purposes
time_vec = []
mass_co2_vec = []
total_mass_vec = []
mass_dissolved_co2_vec = []
density_dissolved_co2 = []

# current relative time
t = 0

# Loop through directory of segmentations
for c, im in enumerate(list_dir):

    # Update relative time with image-title information
    if c != 0:
        t += ImageTime.dt(list_dir[c - 1], list_dir[c])
    time_vec.append(t)
    seg = np.load(os.path.join(dir, im))

    # Compute mass of free co2
    mass_co2 = mass_analysis.free_co2_mass(seg, pressure(t), 2)
    mass_co2_vec.append(mass_co2)

    # Compute mass of dissolved CO2 as the difference between total mass and mass of free CO2
    mass_dissolved_co2_vec.append(total_mass(t) - mass_co2)

    # Compute volume of dissolved CO2
    volume_dissolved = np.sum(mass_analysis.volume_map(seg, 1))

    # Compute densitylike entity for dissolved CO2.
    if volume_dissolved > 1e-9:
        density_dissolved_co2.append((total_mass(t) - mass_co2) / volume_dissolved)
    else:
        density_dissolved_co2.append(0)

    # Update total mass vector
    total_mass_vec.append(total_mass(t))


# On can now play around with the computed entities. Here we make some matplotlib plots.
fig = plt.figure("Mass analysis of C1")
plt.subplot(511)
plt.title("Free CO2 as a function of time")
plt.plot(time_vec, mass_co2_vec)
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")

plt.subplot(513)
plt.title("Dissolved CO2 as a function of time")
plt.plot(time_vec, mass_dissolved_co2_vec)
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")


plt.subplot(515)
plt.title("Total Injected CO2")
plt.plot(time_vec, total_mass_vec)
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")


plt.figure("Dissolved CO2/Volume")
plt.title("Dissolved CO2 divided by volume")
plt.plot(time_vec, density_dissolved_co2)
plt.xlabel("Time (min)")
plt.ylabel("Density (g/m^3)")

plt.show()
