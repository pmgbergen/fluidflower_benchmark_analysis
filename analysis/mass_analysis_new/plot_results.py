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


sns.set_style("whitegrid")

# sns.set_theme()
# sns.set_context("paper")

palette = sns.color_palette()
# palette = sns.hls_palette(8, l=.3, s=.8)
# sns.palplot(palette)

def mobile_co2_box_A():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_boxA.xlsx"]
    
    
    
    time = np.linspace(0,432000,432000)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    mobile = []
    for i in files[1:-1]:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values*60, df.mobile.values)
        mobile.append(interp(time))
    
    avg = np.mean(mobile,axis=0)
    
    fig, ax = plt.subplots(figsize=(12,8))
    axins = zoomed_inset_axes(ax, 6, loc=6) # zoom = 6
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time*60,df.mobile,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-12:-10])
        axins.plot(df.time*60, df.mobile,color=palette[i],markersize = 3,fillstyle = "none")
        ax.set_title("Mobile $CO_{2}$ in box A")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Mass (g)")
        t_max = (df.time[df.mobile == max(df.mobile.values)].values[0])*60
        print("Time for Max mobile CO2 for ",files[i][-12:-10]," ",str(t_max)," seconds")
        print("Mobile CO2 after 72 Hours for",files[i][-12:-10],": ", round(df.mobile[df.time == 72*60].values[0],4)," grams")
    ax.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()
    ax.set_xscale("log")
    # axins.set_xscale("log")
    axins.set_xticks([15000,20000],["1.5x10\N{SUPERSCRIPT Four}","2x10\N{SUPERSCRIPT Four}"])
    
    # axins.plot(time,avg,color = palette[5],linewidth=2)
    axins.set_xlim(200*60, 350*60) # Limit the region for zoom
    axins.set_ylim(2.65, 2.85)
    
    axins.yaxis.tick_right()
    # plt.xticks(visible=False)  # Not present ticks
    # plt.yticks(visible=False)
    #
    ## draw a bbox of the region of the inset axes in the parent axes and
    ## connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    
    plt.draw()
    gs = fig.tight_layout()
    plt.savefig("Mobile_co2_box_A",dpi = 1000)
    
def dissolved_co2_box_A():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_boxA.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_boxA.xlsx"]

    time = np.linspace(0,7200,7200)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    dissolved = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values, df.dissolved.values)
        dissolved.append(interp(time))
    
    avg = np.mean(dissolved,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time,df.dissolved,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-12:-10])
        ax.set_title("Dissolved $CO_{2}$ in box A")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Mass (g)")
        print("Dissolved CO2 after 72 Hours: for",files[i][-12:-10],": ", round(df.dissolved[df.time == 72*60].values[0],4)," grams")
    ax.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()    
    plt.savefig("Dissolved_co2_box_A",dpi = 1000)

def mobile_co2_box_B():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_boxB.xlsx"]
    
    
    
    time = np.linspace(0,7200,7200)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    mobile = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values, df.mobile.values)
        mobile.append(interp(time))
    
    avg = np.mean(mobile,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time,df.mobile,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-12:-10])
        plt.title("Mobile $CO_{2}$ in box B")
        plt.xlabel("Time [min]")
        plt.ylabel("Mass (g)")
        t_max = (df.time[df.mobile == max(df.mobile.values)].values[0])*60
        print("Mobile CO2 after 72 Hours: for",files[i][-12:-10],": ", round(df.mobile[df.time == 72*60].values[0],4)," grams")
    plt.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()
    plt.savefig("Mobile_co2_box_B",dpi = 1000)
    
def dissolved_co2_box_B():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_boxB.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_boxB.xlsx"]
    
    time = np.linspace(0,7200,7200)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    dissolved = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values, df.dissolved.values)
        dissolved.append(interp(time))
    
    avg = np.mean(dissolved,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time,df.dissolved,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-12:-10])
        plt.title("Dissolved $CO_{2}$ in box B")
        plt.xlabel("Time [min]")
        plt.ylabel("Mass (g)")
        print("Mobile CO2 after 72 Hours: for",files[i][-12:-10],": ", round(df.dissolved[df.time == 72*60].values[0],4)," grams")
    plt.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()
    plt.savefig("Dissolved_co2_box_B",dpi = 1000)
    
def mobile_co2_whole_FL():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_whole_FL.xlsx"]
    
    
    
    time = np.linspace(0,7200,7200)
    #     -1, -1 + 10 * atmospheric_pressures.shape[0], atmospheric_pressures.shape[0]
    mobile = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values, df.mobile.values)
        mobile.append(interp(time))
    
    avg = np.mean(mobile,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time,df.mobile,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-16:-14])
        ax.set_title("Mobile $CO_{2}$ in Whole FL")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Mass (g)")
        t_max = (df.time[df.mobile == max(df.mobile.values)].values[0])*60
        print("Time for Max mobile CO2 for ",files[i][-16:-14]," ",str(t_max)," seconds")
        print("Mobile CO2 after 72 Hours for",files[i][-16:-14],": ", round(df.mobile[df.time == 72*60].values[0],4)," grams")
    ax.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()
    plt.savefig("Mobile_co2_wholw_FL",dpi = 1000)
    
def dissolved_co2_whole_FL():
    files = ["E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C1_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C2_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C3_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C4_whole_FL.xlsx",
             "E:/Git/fluidflower_benchmark_analysis/analysis/mass_analysis_new/C5_whole_FL.xlsx"]

    time = np.linspace(0,7200,7200)
    dissolved = []
    for i in files:
        df = pd.read_excel(i,names = ["time","mobile","dissolved"])
        interp = interpolate.interp1d(df.time.values, df.dissolved.values)
        dissolved.append(interp(time))
    
    avg = np.mean(dissolved,axis=0)
    
    fig, ax = plt.subplots()
    for i in range(len(files)):
        df = pd.read_excel(files[i],names = ["time","mobile","dissolved"])
        ax.plot(df.time,df.dissolved,"o",color=palette[i],markersize = 3,fillstyle = "none",label = files[i][-16:-14])
        ax.set_title("Dissolved $CO_{2}$ in Whole FL")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Mass (g)")
        print("Dissolved CO2 after 72 Hours for",files[i][-16:-14],": ", round(df.dissolved[df.time == 72*60].values[0],4)," grams")
    ax.plot(time,avg,color = palette[5],label = "Average",linewidth=2)
    ax.legend()    
    plt.savefig("Dissolved_co2_whole_FL",dpi = 1000)
    
mobile_co2_box_A()
# dissolved_co2_box_A()
# mobile_co2_box_B()
# dissolved_co2_box_B()
# mobile_co2_whole_FL()
# dissolved_co2_whole_FL()

