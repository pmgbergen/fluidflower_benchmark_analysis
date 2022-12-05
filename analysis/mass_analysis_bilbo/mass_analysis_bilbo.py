import os

import darsia as da
import matplotlib.pyplot as plt
import numpy as np
from benchmark.standardsetups.binary_mass_analysis import BinaryMassAnalysis
from scipy import interpolate
from numpy import genfromtxt
from benchmark.utils.misc import read_time_from_path
from datetime import datetime,timedelta
import pandas as pd
from total_mass_function import total_mass_MFC

# Read atmospheric pressure data (found at https://veret.gfi.uib.no/?action=period_query)
# NOTE: The atmospheric pressures are as of now not corrected for elevation (as Jan commented upon in the email)
df = pd.read_excel("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/Florida_2021-09-15_2022-09-17_1669128204.xlsx")
date = [datetime.strptime(df.Dato.loc[i]+" "+df.Tid.loc[i],'%Y-%m-%d %H:%M') for i in range(len(df))]
df["Date"] = date

folders_path = "E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/"
folders = [folders_path+i+"/npy_segmentation/" for i in os.listdir(folders_path)]

inj_start_times = [datetime(2022,5,7,18,56, 47),# BC01 injection start time
             datetime(2022,5,23,15,6, 0),# BC02 injection start time
             datetime(2022,8,17,13,6, 0),# BC03 injection start time
             datetime(2022,8,31,10,36, 0),# BC04 injection start time
             ]

for c,i in enumerate(folders):
    inj_start = inj_start_times[c]
    df1 = df[(df.Date>=inj_start-timedelta(minutes=10)) & (df.Date<=inj_start+timedelta(days=5)+timedelta(minutes=10))]
    df1["dt"] = ((df1.Date-inj_start).dt.total_seconds())/60
    df1["Lufttrykk"] = df1.Lufttrykk+3.625 # adjust for the height difference
    
    # Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
    pressure = interpolate.interp1d(df1.dt.values, 0.001 * df1.Lufttrykk.values)
    
    
    
    # Provide path to segmentations
    list_dir = [i+img for img in os.listdir(i)]
    
    base_segmentation = da.Image(
        np.load(list_dir[0]), width=0.92, height=0.55, color_space="GRAY"
    )
    
    # Create mass analysis object (the porosity could also be provided
    # as an np.ndarray with varying porosities depending on the sand layers.)
    mass_analysis = BinaryMassAnalysis(
        base_segmentation,
        depth_map=0.0105*np.ones_like(base_segmentation.img),
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
    for im in list_dir:
        print(im)
        # Update relative time with image-title information
        t_img = read_time_from_path(im)
        t = (t_img-inj_start).total_seconds()/60
        time_vec.append(t)
        seg = np.load(im)
    
        # Compute mass of free co2
        mass_co2 = mass_analysis.free_co2_mass(seg, pressure(t), 2)
        mass_co2_vec.append(mass_co2)
    
        # Compute mass of dissolved CO2 as the difference between total mass and mass of free CO2
        mass_dissolved_co2_vec.append(total_mass_MFC(t) - mass_co2)
    
        # Update total mass vector
        total_mass_vec.append(total_mass_MFC(t))
    df2 = pd.DataFrame()
    df2["Time"] = time_vec
    df2["Mobile_co2_[g]"] = mass_co2_vec
    df2["dissolved_co2_[g]"] = mass_dissolved_co2_vec
    df2["total_co2_injected_[g]"] = total_mass_vec
    df2.to_excel(i[-22:-18]+".xlsx",index=None)
    


# On can now play around with the computed entities. Here we make some matplotlib plots.
# fig = plt.figure("Mass analysis of BC01")
# plt.subplot(511)
# plt.title("Free CO2 as a function of time")
# plt.plot(time_vec, mass_co2_vec, label="Mass from image data")
# plt.plot(time_vec, mobile_mass(time_vec), label="Mass from Lluis")
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")
# plt.legend()

# plt.subplot(513)
# plt.title("Dissolved CO2 as a function of time")
# plt.plot(time_vec, mass_dissolved_co2_vec, label="Mass from image data")
# plt.plot(time_vec, dissolved_mass(time_vec), label="Mass from Lluis")
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")
# plt.legend()


# plt.subplot(515)
# plt.title("Total Injected CO2")
# plt.plot(time_vec, total_mass_vec)
# plt.xlabel("Time (min)")
# plt.ylabel("Mass (g)")

# plt.show()
