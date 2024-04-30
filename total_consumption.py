# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:58:17 2024

@author: matheo
"""


"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
from scipy.stats import shapiro
import seaborn as sb
"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(1)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, False)

#%%
# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

# Assuming df is your DataFrame
# Replace zeros with NaN values
df_nan = Loads.replace(0, np.nan)

# Calculate the mean over columns
mean_values = df_nan.mean()


plt.bar(range(len(mean_values)), 4*mean_values, color=sb.color_palette("hls", 13)[8])
# Set the x-axis ticks to the list of names
plt.xticks(range(len(mean_values)), Loads.columns.tolist())
#plt.yscale("log")
plt.ylabel("load [$kW_{el}/m^2$]")
plt.xlabel("Consumers' IDs")
plt.title("Overall average load per meter squared")
plt.grid(axis="y")

#%% grading for comparison matrix - overload score 

x = Loads.columns
y = 4*mean_values

def get_score(typology_names, parameters):
    min_ = min(parameters)
    max_= max(parameters)
    
    grades = {}
    classes = {}
    
    for i, value in enumerate(parameters):
        grades[typology_names[i]] = 100*(parameters[i]-min_)/(max_-min_)
    
        if grades[typology_names[i]] <= 20 : 
            classes[typology_names[i]] = 1
        elif grades[typology_names[i]] <= 40 :
            classes[typology_names[i]] = 2
        elif grades[typology_names[i]] <= 60 :
            classes[typology_names[i]] = 3
        elif grades[typology_names[i]] <= 80 :
            classes[typology_names[i]] = 4
        elif grades[typology_names[i]] <= 100 :
            classes[typology_names[i]] = 5
    
    thresholds = [v/100*(max_-min_) + min_ for v in [0, 20, 40, 60, 80 , 100]]


    
    return grades, classes, thresholds 

grades, classes, thresholds = get_score(x, y)

plt.figure()
for i, (k, v) in enumerate(classes.items()):
    print(type(v))
    if v == 1:
        plt.bar(i,v, color="green")
    elif v == 2:
        plt.bar(i,v, color="yellow")
    elif v == 3:
        plt.bar(i,v, color="orange")
    elif v == 4:
        plt.bar(i,v, color="red" )
    elif v == 5:
        plt.bar(i,v, color="purple")
plt.grid(axis='y')
plt.xticks(range(len(x)), x)
plt.show()


#%%

#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#color list 
#color = ["darkblue", "royalblue", "green", "yellow", "orange", "red", "purple"]
color = sb.color_palette("husl", 29, 1)
colori = 0
#initiating figure
plt.figure()
for i, typo in enumerate(Typo_list) : 
    
    loads = Typo_all_loads[typo]
    # Obtain a typical year
    t_year = f.typical_period(loads,  "year")
    #obtain typical day 
    t_day = f.typical_period(t_year, "day")
    

    

    for j, col in enumerate(t_day.columns):
        t_day[col].plot(color=color[colori], label=col)
        colori += 1

plt.yscale("log")

# Manually set x-axis ticks to display only the hour component
#hours = mdates.HourLocator(interval=365)
#hours_fmt = mdates.DateFormatter("%H:%M")
#plt.gca().xaxis.set_major_locator(hours)
#plt.gca().xaxis.set_major_formatter(hours_fmt)
tick_labels = ["0"+str(i)+":00" if i < 10 else str(i)+":00" for i in range(0, 24, 2)]
tick_positions = [i*8 for i in range(12)]
plt.xticks(tick_positions, tick_labels, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Consumers")
plt.tight_layout(rect=[0, 0, 1, 2.3])
plt.xlabel("Hours")
plt.ylabel("Mean daily consumption [$kWh_{el}$]")
plt.title("Mean daily consumption")
plt.grid()
plt.show()

