import os

import darsia as da
import matplotlib.pyplot as plt
import numpy as np
from benchmark.standardsetups.binary_mass_analysis import BinaryMassAnalysis
from scipy import interpolate
from numpy import genfromtxt
from benchmark.utils.time_from_image_name import ImageTime
from datetime import datetime,timedelta
import pandas as pd
from total_mass_function import total_mass_MFC

# Read atmospheric pressure data (found at https://veret.gfi.uib.no/?action=period_query)
# NOTE: The atmospheric pressures are as of now not corrected for elevation (as Jan commented upon in the email)

inj_start = datetime(2022,5,5,18,56, 0) # BC01 injection start time
df = pd.read_excel("C:/Users/Erlend/src/darsia/images/lluis_segmentation_comp/Florida_2022-05-02_2022-05-13_1669366856.xlsx")
date = [datetime.strptime(df.Dato.loc[i]+" "+df.Tid.loc[i],'%Y-%m-%d %H:%M') for i in range(len(df))]

df["Date"] = date
df = df[(df.Date>=inj_start-timedelta(minutes=10)) & (df.Date<=inj_start+timedelta(days=5)+timedelta(minutes=10))]
df["dt"] = ((df.Date-inj_start).dt.total_seconds())/60

# Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
pressure = interpolate.interp1d(df.dt.values, 0.001 * df.Lufttrykk.values)


# Read data from lluis's csv file
lluis_data = pd.read_csv("C:/Users/Erlend/src/darsia/images/lluis_segmentation_comp/mass_exp3_mesh5_modelcase3_D1e-09_I1m1_I2m1_repPhi_repSwcSgt_model1krgkrw_Cpc3.2_ESFpc2_ESFk0.33_Ck1_Ek1.2_Fk1.7.csv")
# Read time array from csv file
time = lluis_data["t_s"].values
# read total mass array from csv file
total_mass_array = lluis_data["tot_kg"].values
# read mobile mass array from csv file
mobile_mass_array = lluis_data["mob_kg"].values
# read immobile mass array from csv file
immobile_mass_array = lluis_data["immob_kg"].values
# read dissolved mass array from csv file
dissolved_mass_array = lluis_data["diss_kg"].values

# create total mass as a function of time
total_mass = interpolate.interp1d(time/60, 1000*total_mass_array)
# create mobile mass as a function of time from lluis
mobile_mass = interpolate.interp1d(time/60, 1000*mobile_mass_array)
# create dissolved mass as a function of time from lluis
dissolved_mass = interpolate.interp1d(time/60, 1000*dissolved_mass_array)


# Provide path to segmentations
dir = "C:/Users/Erlend/src/darsia/images/bc01_segmentation_malin"
list_dir = os.listdir(dir)

base_segmentation = da.Image(
    np.load(os.path.join(dir, list_dir[0])), width=0.92, height=0.55, color_space="GRAY"
)

# Create mass analysis object (the porosity could also be provided
# as an np.ndarray with varying porosities depending on the sand layers.)
mass_analysis = BinaryMassAnalysis(
    base_segmentation,
    depth_map=0.01*np.ones_like(base_segmentation.img),
    porosity=0.44,
)


# Create empty lists for plotting purposes
time_vec = []
mass_co2_vec = []
total_mass_vec = []
mass_dissolved_co2_vec = []

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

    # Update total mass vector
    total_mass_vec.append(total_mass(t))


# On can now play around with the computed entities. Here we make some matplotlib plots.
fig = plt.figure("Mass analysis of BC01")
plt.subplot(511)
plt.title("Free CO2 as a function of time")
plt.plot(time_vec, mass_co2_vec, label="Mass from image data")
plt.plot(time_vec, mobile_mass(time_vec), label="Mass from Lluis")
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")
plt.legend()

plt.subplot(513)
plt.title("Dissolved CO2 as a function of time")
plt.plot(time_vec, mass_dissolved_co2_vec, label="Mass from image data")
plt.plot(time_vec, dissolved_mass(time_vec), label="Mass from Lluis")
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")
plt.legend()


plt.subplot(515)
plt.title("Total Injected CO2")
plt.plot(time_vec, total_mass_vec)
plt.xlabel("Time (min)")
plt.ylabel("Mass (g)")

plt.show()
