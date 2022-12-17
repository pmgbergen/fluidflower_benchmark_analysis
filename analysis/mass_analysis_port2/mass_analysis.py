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
from skimage.measure import regionprops, label




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
labels = np.load("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_port1/labels_fine.npy")

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
    mass_co2_subregion1_vec = []
    mass_co2_subregion2_vec = []
    total_mass_vec = []
    mass_dissolved_co2_vec = []
    mass_dissolved_co2_subregion1_vec = []
    mass_dissolved_co2_subregion2_vec = []
    mass_dissolved_co2_esf_subregion1_vec = []
    mass_dissolved_co2_esf_subregion2_vec = []
    mass_dissolved_co2_esf_vec = []
    density_dissolved_co2 = []
    
    # Choose a subregion
    subregion1 = np.array([[1.1, 0.6], [2.8, 0.0]]) #boxA
    subregion2 = np.array([[0, 1.2], [1.1, 0.6]]) #boxB
    
    # current relative time
    t = 0
    
    # Loop through directory of segmentations
    for c, im in enumerate(list_dir[:-6]):
        print(im)
        # Update relative time with image-title information
        if c != 0:
            t += ImageTime.dt(list_dir[c - 1], list_dir[c])
        # t = int(im[13:-5])/60
        time_vec.append(t)
        seg = np.load(os.path.join(dir, im))
        
        # Remove co2 injected from port 2-----------------------
        # seg[:2737,2893:]=0
        
            # Get labels and regions using skimage functions
        seg_label = label(seg)
        regions = regionprops(seg_label)

            # Define the top right and rest of the image
        top_right = np.zeros_like(seg, dtype = bool)
        rest = np.zeros_like(seg, dtype = bool)
        top_right_bb = [[2670, 3450], [0, seg.shape[1]]]

            # Loop through regions and check if centroid is in top right
        for i in range(len(regions)):
            if regions[i].centroid[0] < top_right_bb[0][0] and regions[i].centroid[0] > top_right_bb[1][0] and regions[i].centroid[1] > top_right_bb[0][1] and regions[i].centroid[1] < top_right_bb[1][1]:
                top_right[seg_label == regions[i].label] = True
            else:
                rest[seg_label == regions[i].label] = True

            # Extract the top left and rest of the image as separate segmentations
        seg_port2 = np.zeros_like(seg)
        seg_port2[top_right] = seg[top_right]

        seg_port1 = np.zeros_like(seg)
        seg_port1[rest] = seg[rest]
        #-----------------------------------------------------------
        
        # Compute mass of free co2
        mass_co2 = mass_analysis.free_co2_mass(seg_port2, pressure(t), 2)
        mass_co2_vec.append(mass_co2)
    
        # Compute mass of dissolved CO2 as the difference between total mass and mass of free CO2
        mass_dissolved_co2_vec.append(total_mass(t) - mass_co2)
    
        # Compute volume of dissolved CO2 in entire rig
        volume_dissolved = np.sum(mass_analysis.volume_map(seg_port2, 1))
    
        mass_co2_subregion1_vec.append(mass_analysis.free_co2_mass(seg_port2, pressure(t), roi = subregion1))
        mass_co2_subregion2_vec.append(mass_analysis.free_co2_mass(seg_port2, pressure(t), roi = subregion2))
    
        # Compute volume of dissolved CO2 in subregions
        volume_dissolved_subregion1 = np.sum(mass_analysis.volume_map(seg_port2, 1, roi = subregion1))
        volume_dissolved_subregion2 = np.sum(mass_analysis.volume_map(seg_port2, 1, roi = subregion2))
        
        # Compute volume of dissolved CO2 in seal in subregions
        esf_label = 3
        seg_port2_esf = np.zeros_like(seg_port2)
        seg_port2_esf[labels == esf_label] = seg_port2[labels == esf_label]

        volume_dissolved_esf_subregion1 = np.sum(mass_analysis.volume_map(seg_port2_esf, 1, roi = subregion1))
        volume_dissolved_esf_subregion2 = np.sum(mass_analysis.volume_map(seg_port2_esf, 1, roi = subregion2))
        
        # Compute mass of dissolved CO2 in seal
        volume_dissolved_esf = np.sum(mass_analysis.volume_map(seg_port2_esf, 1))
        mass_dissolved_co2_esf_vec.append((total_mass(t)-mass_co2)*volume_dissolved_esf/volume_dissolved)
        
        # Compute dissolved co2 in seal and in subregions provided above
        if volume_dissolved > 1e-9:
            mass_dissolved_co2_subregion1_vec.append((total_mass(t)-mass_co2)*volume_dissolved_subregion1/volume_dissolved)
            mass_dissolved_co2_subregion2_vec.append((total_mass(t)-mass_co2)*volume_dissolved_subregion2/volume_dissolved)
            
            mass_dissolved_co2_esf_subregion1_vec.append((total_mass(t)-mass_co2)*volume_dissolved_esf_subregion1/volume_dissolved)
            mass_dissolved_co2_esf_subregion2_vec.append((total_mass(t)-mass_co2)*volume_dissolved_esf_subregion2/volume_dissolved)
        else:
            mass_dissolved_co2_subregion1_vec.append(0)
            mass_dissolved_co2_subregion2_vec.append(0)
            mass_dissolved_co2_esf_subregion1_vec.append(0)
            mass_dissolved_co2_esf_subregion2_vec.append(0)
    
        # Compute densitylike entity for dissolved CO2.
        if volume_dissolved > 1e-9:
            density_dissolved_co2.append((total_mass(t) - mass_co2) / volume_dissolved)
        else:
            density_dissolved_co2.append(0)
    
        # Update total mass vector
        total_mass_vec.append(total_mass(t))
    
    df = pd.DataFrame()
    df["Time_[min]"] = time_vec
    df["Mobile_CO2_[g]"] = mass_co2_vec
    df["Dissolved_CO2_[g]"] = mass_dissolved_co2_vec
    df["Mobile_CO2_boxA_[g]"] = mass_co2_subregion1_vec
    df["Dissolved_CO2_boxA_[g]"] = mass_dissolved_co2_subregion1_vec
    df["Dissolved_CO2_esf_boxA_[g]"] = mass_dissolved_co2_esf_subregion1_vec
    df["Mobile_CO2_boxB_[g]"] = mass_co2_subregion2_vec
    df["Dissolved_CO2_boxB_[g]"] = mass_dissolved_co2_subregion2_vec
    df["Dissolved_CO2_esf_boxB_[g]"] = mass_dissolved_co2_esf_subregion2_vec
    df["Dissolved_CO2_esf_[g]"] = mass_dissolved_co2_esf_vec
    df.to_excel(dir[-3:-1]+"_port2.xlsx",index=None)
    




