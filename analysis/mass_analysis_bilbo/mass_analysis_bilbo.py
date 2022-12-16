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
from total_mass_function import total_mass

# Read atmospheric pressure data (found at https://veret.gfi.uib.no/?action=period_query)
# NOTE: The atmospheric pressures are as of now not corrected for elevation (as Jan commented upon in the email)
df = pd.read_excel("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/Florida_2021-09-15_2022-12-12_1669128204.xlsx")
date = [datetime.strptime(df.Dato.loc[i]+" "+df.Tid.loc[i],'%Y-%m-%d %H:%M') for i in range(len(df))]
df["Date"] = date

# folders_path = "E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/"
folders = ["E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/BC01/npy_segmentation/",
           "E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/BC02/npy_segmentation/",
           "E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/BC03/npy_segmentation/",
           "E:/Git/fluidflower_benchmark_analysis/analysis/bilbo_results/BC04/npy_segmentation/"]

inj_start_times = [datetime(2022,5,7,18,56,47),# BC01
                   datetime(2022,5,23,15,6,0),# BC02
                   datetime(2022,8,17,13,6,0),# BC03
                   datetime(2022,8,31,10,36,0)# BC04
             ]

# Depth map
scale1 = 0.01 # cm to m
scale2 = 0.001 # mm to m

x = [1,6,16,26,36,46,56,66,76,86,91]*6
y = [1]*11 + [6]*11 + [16]*11 + [26]*11 + [36]*11 + [46]*11
    
measured = [8.53,	8.52,	8.27,	8.38,	8.20,	8.27,	8.40,	8.34,	8.45,	8.64,	8.58,
                8.64,	8.52,	8.00,	7.70,	7.54,	7.50,	7.68,	7.74,	8.16,	8.51,	8.32,
                8.61,	8.42,	7.62,	6.92,	6.49,	6.46,	6.65,	7.26,	7.87,	8.24,	8.46,
                8.56,	8.34,	7.50,	6.75,	6.13,	6.12,	6.49,	7.05,	7.81,	8.60,	8.59,
                8.58,	8.62,	8.03,	7.45,	6.81,	6.66,	6.89,	7.01,	8.09,	8.79,	8.88,
                8.68,	8.97,	8.12,	8.12,	7.65,	7.48,	7.51,	8.03,	8.62,	9.03,	9.04]

def depth(y):
    return round((49.28 - (2 * (y-5))- (2*15.77)))
depth = [depth(i) for i in measured]
depth_measurements = (np.array(x)*scale1,np.array(y)*scale1,np.array(depth)*scale2)

for c,i in enumerate(folders):
    inj_start = inj_start_times[c]
    df1 = df[(df.Date>=inj_start-timedelta(minutes=100)) & (df.Date<=inj_start+timedelta(days=2)+timedelta(minutes=100))]
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
        depth_measurements=depth_measurements,
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
        mass_dissolved_co2_vec.append(total_mass(t) - mass_co2)
    
        # Update total mass vector
        total_mass_vec.append(total_mass(t))
    df2 = pd.DataFrame()
    df2["Time"] = time_vec
    df2["Mobile_co2_[g]"] = mass_co2_vec
    df2["dissolved_co2_[g]"] = mass_dissolved_co2_vec
    df2["total_co2_injected_[g]"] = total_mass_vec
    df2.to_excel(i[-22:-18]+".xlsx",index=None)
