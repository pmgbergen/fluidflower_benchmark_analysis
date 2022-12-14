# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:09:49 2022

@author: bbe020
"""

import numpy as np

def total_mass(t):
    
    rate = [0,10,15,20,15,10,5,0,10,15,20,15,10,5]
    diff = [0,60,60,3793,60,60,60,77,60,60,4372,60,60,60]
    diff = np.array(diff)/60
    time = np.cumsum(diff)
    rate = np.array(rate)*0.1*1.82/1000
    
    if t < 0:
        return 0
    if t <= time[-1]:
        v = len(time[time<t])
        mass = np.sum(np.multiply(rate[:v],diff[:v])) + ((t-time[v-1])*rate[v])
        return mass
    else:
        return np.sum(np.multiply(rate,diff))


# def total_mass_MFC(t):
#     df = pd.read_excel("//klient.uib.no/FELLES/LAB-IT/IFT/Resfys/medium_FF_AB_data/Py_MFC_Script_Bilbo/Bilbo_MFC_Summary_relative_time.xlsx",sheet_name="BC02")
#     time = df["MFC time [min]"].values
#     time_dt = df["MFC time [min]"].diff().values[1:]
#     rate = df["MFC measured [ml/min]"].values[1:]*1.82
    
#     if t <= max(time):
#         time = time[time <= t]
#         time_dt = time_dt[:len(time)-1]
#         rate = rate[:len(time)-1]
#         return np.sum(time_dt*rate)*0.001
#     else:
#         return np.sum(time_dt*rate)*0.001
    
# def total_mass(t):
#     if t < 0:
#         return 0
#     elif t < 1:
#         return 1.82 * 1 * t / 1000
#     elif t < 2:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * (t - 1) / 1000
#         )
#     elif t < 3913 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (t - 2) / 1000
#         )
#     elif t < 3973 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * (t - (3913/60)) / 1000
#         )
#     elif t < 4033 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * (t - 3973) / 1000
#         )
#     elif t < 4093 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * (t - 4033) / 1000
#         )
#     elif t < 4170 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#         )
#     elif t < 4230 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * (t - (4170 / 60)) / 1000
#         )
#     elif t < 4290 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * (t - (4230 / 60)) / 1000
#         )
#     elif t < 8662 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (t - (4290 / 60)) / 1000
#         )
#     elif t < 8722 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (4372 / 60) / 1000
#             + 1.82 * 1.5 * (t - (8662 / 60)) / 1000
#         )
#     elif t < 8782 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (4372 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * (t - (8722 / 60)) / 1000
#         )
#     elif t < 8842 / 60:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (4372 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * (t - (8782 / 60)) / 1000
#         )
#     else:
#         return (
#             1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (3793 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 2 * (4372 / 60) / 1000
#             + 1.82 * 1.5 * 1 / 1000
#             + 1.82 * 1 * 1 / 1000
#             + 1.82 * 0.5 * 1 / 1000
#         )

# def total_mass_l(t):
#     if t < 0:
#         return 0
#     elif t < 1:
#         return 1.784 * 1 * t / 1000
#     elif t < 2:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * (t - 1) / 1000
#         )
#     elif t < 3913 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (t - 2) / 1000
#         )
#     elif t < 3973 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * (t - (3913/60)) / 1000
#         )
#     elif t < 4033 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * (t - (3973 / 60)) / 1000
#         )
#     elif t < 4093 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * (t - (4033 / 60)) / 1000
#         )
#     elif t < 4170 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#         )
#     elif t < 4230 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * (t - (4170 / 60)) / 1000
#         )
#     elif t < 4290 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * (t - (4230 / 60)) / 1000
#         )
#     elif t < 8662 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784* 1.5 * 1 / 1000
#             + 1.784 * 2 * (t - (4290 / 60)) / 1000
#         )
#     elif t < 8722 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (4372 / 60) / 1000
#             + 1.784 * 1.5 * (t - (8662 / 60)) / 1000
#         )
#     elif t < 8782 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (4372 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * (t - (8722 / 60)) / 1000
#         )
#     elif t < 8842 / 60:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (4372 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * (t - (8782 / 60)) / 1000
#         )
#     else:
#         return (
#             1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (3793 / 60) / 1000
#             + 1.7842 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 2 * (4372 / 60) / 1000
#             + 1.784 * 1.5 * 1 / 1000
#             + 1.784 * 1 * 1 / 1000
#             + 1.784 * 0.5 * 1 / 1000
#         )
