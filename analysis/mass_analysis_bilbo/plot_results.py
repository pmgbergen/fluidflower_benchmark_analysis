# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:40:31 2022

@author: bbe020
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns
from scipy import interpolate
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from total_mass_function import total_mass


sns.set_style("whitegrid")

# sns.set_theme()
# sns.set_context("paper")

# palette = sns.color_palette()
# palette = sns.hls_palette(8, l=.3, s=.8)
# sns.palplot(palette)

colors = [[0,176,80],[112,48,160],[0,176,240],[0,112,192],[0,0,0]]

colors = np.array(colors[:])/255


def mobile_co2():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC01.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC02.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC03.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC04.xlsx"]
    
    
    # # Read data from lluis's csv file
    # lluis_data = pd.read_csv("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/mass_exp3_mesh5_modelcase3_D1e-09_I1m1_I2m1_repPhi_repSwcSgt_model1krgkrw_Cpc3.2_ESFpc2_ESFk0.33_Ck1_Ek1.2_Fk1.7.csv")
    # # Read time array from csv file
    # time_l = lluis_data["t_s"].values/60
    # # read total mass array from csv file
    # total_l= lluis_data["tot_kg"].values*1000
    # # read mobile mass array from csv file
    # mobile_l = lluis_data["mob_kg"].values*1000
    # # read immobile mass array from csv file
    # immobile_l = lluis_data["immob_kg"].values*1000
    # # read dissolved mass array from csv file
    # dissolved_l = lluis_data["diss_kg"].values*1000
    
    time = np.linspace(1,2580,2580)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    mobile = []
    for i in files:
        print(i)
        df = pd.read_excel(i,names = ["time","mobile","dissolved","injected"])
        interp = interpolate.interp1d(df.time.values, df.mobile.values)
        mobile.append(interp(time))
    
    avg = np.mean(mobile,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved","injected"])
        ax.plot(df.time,df.mobile,"o",color=colors[i],markersize = 3,fillstyle = "none",label = files[i][-9:-5])
        plt.title("Mobile $CO_{2}$")
        plt.xlabel("Time [min]")
        plt.ylabel("Mass (g)")
    plt.plot(time,avg,"--",color = colors[-1],label = "Average",linewidth=2)
    # plt.plot(time_l,mobile_l,color = palette[6],label = "Luis",linewidth=2)
    ax.legend()
    plt.savefig("Mobile_co2",dpi = 1000)
    
def dissolved_co2():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC01.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC02.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC03.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC04.xlsx"]
    
    
    # # Read data from lluis's csv file
    # lluis_data = pd.read_csv("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/mass_exp3_mesh5_modelcase3_D1e-09_I1m1_I2m1_repPhi_repSwcSgt_model1krgkrw_Cpc3.2_ESFpc2_ESFk0.33_Ck1_Ek1.2_Fk1.7.csv")
    # # Read time array from csv file
    # time_l = lluis_data["t_s"].values/60
    # # read total mass array from csv file
    # total_l= lluis_data["tot_kg"].values*1000
    # # read mobile mass array from csv file
    # mobile_l = lluis_data["mob_kg"].values*1000
    # # read immobile mass array from csv file
    # immobile_l = lluis_data["immob_kg"].values*1000
    # # read dissolved mass array from csv file
    # dissolved_l = lluis_data["diss_kg"].values*1000
    
    time = np.linspace(1,2580,2580)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    dissolved = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved","injected"])
        interp = interpolate.interp1d(df.time.values, df.dissolved.values)
        dissolved.append(interp(time))
    
    avg = np.mean(dissolved,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved","injected"])
        ax.plot(df.time,df.dissolved,"o",color=colors[i],markersize = 3,fillstyle = "none",label = files[i][-9:-5])
        plt.title("Dissolved $CO_{2}$")
        plt.xlabel("Time [min]")
        plt.ylabel("Mass (g)")
    plt.plot(time,avg,"--",color = colors[-1],label = "Average",linewidth=2)
    # plt.plot(time_l,dissolved_l,color = palette[6],label = "Luis",linewidth=2)
    ax.legend()
    plt.savefig("dissolved_co2",dpi = 1000)

def injected_co2():
    # Read data from lluis's csv file
    lluis_data = pd.read_csv("E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/mass_exp3_mesh5_modelcase3_D1e-09_I1m1_I2m1_repPhi_repSwcSgt_model1krgkrw_Cpc3.2_ESFpc2_ESFk0.33_Ck1_Ek1.2_Fk1.7.csv")
    time_l = lluis_data["t_s"].values/60
    total_l= lluis_data["tot_kg"].values*1000
    
    # Read data from BC01
    path = "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_bilbo/BC01.xlsx"
    df = pd.read_excel(path,names = ["time","mobile","dissolved","injected"])    # Read time array from csv file
    time = df.time
    # total = df.injected
    total = [total_mass(i) for i in time]
    
    fig, ax = plt.subplots()
    plt.plot(time,total,color = palette[5],label = "MFC",linewidth=1)
    plt.plot(time_l,total_l,color = palette[6],label = "Luis",linewidth=1)
    ax.legend()
    plt.title("Injected $CO_{2}$")
    plt.xlabel("Time [min]")
    plt.ylabel("Mass (g)")
    plt.savefig("injected_co2_new",dpi = 1000)


mobile_co2()
dissolved_co2()
# injected_co2()



