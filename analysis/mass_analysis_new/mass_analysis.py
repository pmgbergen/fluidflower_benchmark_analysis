import os

import darsia as da
import matplotlib.pyplot as plt
import numpy as np
from benchmark.standardsetups.binary_mass_analysis import BinaryMassAnalysis
from benchmark.utils.time_from_image_name import ImageTime
# from create_pressure_array import pressure_array
from scipy import interpolate
from total_injected_mass_large_FF import total_mass
from numpy import genfromtxt
from datetime import datetime,timedelta
import pandas as pd



# Read atmospheric pressure data (found at https://veret.gfi.uib.no/?action=period_query)
# NOTE: The atmospheric pressures are as of now not corrected for elevation (as Jan commented upon in the email)
# atmospheric_pressures = pressure_array()

# # Create time array corresponding to the found pressure data
# time_pressure = np.linspace(
#     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
# )

Results_path = "E:/Git/fluidflower_benchmark_analysis/analysis/Results/fine_segmentation/"

seg_folders = [Results_path+i+"/" for i in os.listdir(Results_path)]
inj_start_times = [datetime(2021,11,24,8,31,0), # c1
                   datetime(2021,12,4,10,1,0), # c2
                   datetime(2021,12,14,11,20,0), # c3
                   datetime(2021,12,24,9,0,0), # c4
                   datetime(2022,1,4,11,0,0) # c5
                   ]
time_vec_all = []
mass_co2_vec_all = []
mass_co2_subregion_vec_all = []
total_mass_vec_all = []
mass_dissolved_co2_vec_all = []
mass_dissolved_co2_subregion_vec_all = []
density_dissolved_co2_all = []
for i,dir in enumerate(seg_folders):
    inj_start = inj_start_times[i]
    df = pd.read_excel("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/Florida_2021-11-24_2022-02-02_1669032742.xlsx")
    date = [datetime.strptime(df.Dato.loc[i]+" "+df.Tid.loc[i],'%Y-%m-%d %H:%M') for i in range(len(df))]
    
    df["Date"] = date
    df = df[(df.Date>=inj_start-timedelta(minutes=10)) & (df.Date<=inj_start+timedelta(days=5)+timedelta(minutes=10))]
    df["dt"] = ((df.Date-inj_start).dt.total_seconds())/60
    
    df["Lufttrykk"] = df.Lufttrykk+3.125 # adjust for the height difference
    
    # Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
    # pressure = interpolate.interp1d(time_pressure, 0.001 * atmospheric_pressures)
    pressure = interpolate.interp1d(df.dt.values, 0.001 * df.Lufttrykk.values)
    
    
    # Provide path to segmentations
    # dir = "E:/Git/FL/experiment/benchmarkdata/spatial_maps/run5"
    list_dir = [i for i in os.listdir(dir) if i.endswith(".npy")]
    
    base_segmentation = da.Image(
        np.load(os.path.join(dir, list_dir[0])), width=2.8, height=1.5, color_space="GRAY"
    )
    
    # Provide path to depths (It is provided for the large FF in the "depths-folder")
    meas_dir = "E:/Git/fluidflower_benchmark_analysis/analysis/depths/"
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
    mass_co2_subregion_vec = []
    total_mass_vec = []
    mass_dissolved_co2_vec = []
    mass_dissolved_co2_subregion_vec = []
    density_dissolved_co2 = []
    
    # Choose a subregion
    subregion = np.array([[1.1, 0.6], [2.8, 0.0]]) #boxA
    # subregion = np.array([[0, 1.2], [1.1, 0.6]]) #boxB
    
    # current relative time
    t = 0
    
    # Loop through directory of segmentations
    for c, im in enumerate(list_dir):
    
        # Update relative time with image-title information
        if c != 0:
            t += ImageTime.dt(list_dir[c - 1], list_dir[c])
        # t = int(im[13:-5])/60
        time_vec.append(t)
        seg = np.load(os.path.join(dir, im))
    
        # Compute mass of free co2
        mass_co2 = mass_analysis.free_co2_mass(seg, pressure(t), 2)
        mass_co2_vec.append(mass_co2)
    
        # Compute mass of dissolved CO2 as the difference between total mass and mass of free CO2
        mass_dissolved_co2_vec.append(total_mass(t) - mass_co2)
    
        # Compute volume of dissolved CO2 in entire rig
        volume_dissolved = np.sum(mass_analysis.volume_map(seg, 1))
    
        mass_co2_subregion_vec.append(mass_analysis.free_co2_mass(seg, pressure(t), roi = subregion))
    
        # Compute volume of dissolved CO2 in subregion (for box A now)
        volume_dissolved_subregion = np.sum(mass_analysis.volume_map(seg, 1, roi = subregion))
    
        # Compute dissolved co2 in subregion provided above
        if volume_dissolved > 1e-9:
            mass_dissolved_co2_subregion_vec.append((total_mass(t)-mass_co2)*volume_dissolved_subregion/volume_dissolved)
        else:
            mass_dissolved_co2_subregion_vec.append(0)
    
        # Compute densitylike entity for dissolved CO2.
        if volume_dissolved > 1e-9:
            density_dissolved_co2.append((total_mass(t) - mass_co2) / volume_dissolved)
        else:
            density_dissolved_co2.append(0)
    
        # Update total mass vector
        total_mass_vec.append(total_mass(t))
    time_vec_all.append(time_vec)
    mass_co2_vec_all.append(mass_co2_vec)
    mass_co2_subregion_vec_all.append(mass_co2_subregion_vec)
    total_mass_vec_all.append(total_mass_vec)
    mass_dissolved_co2_vec_all.append(mass_dissolved_co2_vec)
    mass_dissolved_co2_subregion_vec_all.append(mass_dissolved_co2_subregion_vec)
    density_dissolved_co2_all.append(density_dissolved_co2)
    
    df = pd.DataFrame()
    df["Time_[min]"] = time_vec
    df["Mobile_CO2_box_A_[g]"] = mass_co2_subregion_vec
    df["Dissolved_CO2_box_A_[g]"] = mass_dissolved_co2_subregion_vec
    df.to_excel(dir[-3:-1]+"_boxA.xlsx",index=None)
    


# On can now play around with the computed entities. Here we make some matplotlib plots.
# fig = plt.figure("Mass analysis of C1")
# plt.subplot(511)
# plt.title("Free CO2 as a function of time")
# plt.plot(time_vec, mass_co2_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")

# plt.subplot(513)
# plt.title("Dissolved CO2 as a function of time")
# plt.plot(time_vec, mass_dissolved_co2_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")


# plt.subplot(515)
# plt.title("Total Injected CO2")
# plt.plot(time_vec, total_mass_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")

# plt.figure(" BOX A ")
# plt.subplot(311)
# plt.title("CO2 in box A")
# plt.plot(time_vec, mass_co2_subregion_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")

# plt.subplot(313)
# plt.title("Dissolved CO2 in box A")
# plt.plot(time_vec, mass_dissolved_co2_subregion_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")

# plt.figure("Dissolved CO2/Volume")
# plt.title("Dissolved CO2 divided by volume")
# plt.plot(time_vec, density_dissolved_co2)
# plt.xlabel("Time (min)")
# plt.ylabel("Density (g/m^3)")

# plt.show()


