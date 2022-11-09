# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:08:05 2022

@author: bbe020
"""

import daria
from pathlib import Path
import numpy as np
from benchmark.utils.misc import read_time_from_path
import json
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt


box_A_loc = np.array([[1.1, 0.6], [2.8, 0.0]])


with open(Path("config.json"), "r") as openfile:
    config = json.load(openfile)
    
inj_start = datetime.strptime(config["injection_start"], "%y%m%d %H%M%S")
folder_path = Path(config["results_path"])
segmentations_path = list(sorted(folder_path.glob("*.npy")))



def max_free_co2(segmentations_path):
    df = pd.DataFrame(columns=["Time [h]","Normalized co2 area"])
    for seg_path in segmentations_path:
        seg = daria.Image(np.load(seg_path), width=2.8,height=1.5,color_space="GRAY")
        box_A = daria.extractROI(seg,box_A_loc)
        N_co2_area = (np.count_nonzero(box_A.img == 2))/(box_A.img.shape[0]*box_A.img.shape[1])
        timestamp = read_time_from_path(seg_path)
        sec = (timestamp-inj_start).total_seconds()
        df.loc[len(df.index)]=[sec/3600,N_co2_area]
        
    df.to_excel("max_free_gas_box_A.xlsx")
    plt.figure()
    plt.plot(df["Time [h]"],df["Normalized co2 area"])
    plt.xlabel("Time [h]")
    plt.ylabel("Normalized co2 area")
    print("Max free co2 (",round(df["Normalized co2 area"].max(),4)," %) at: ", df["Time [h]"][df["Normalized co2 area"]==df["Normalized co2 area"].max()].values[0]," Hours")
    return df

df = max_free_co2(segmentations_path)
    