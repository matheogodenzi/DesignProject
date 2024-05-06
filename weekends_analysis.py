# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:48:04 2024

@author: matheo
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
from scipy.stats import shapiro
import seaborn as sb
from sklearn.linear_model import LinearRegression
"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, False)
#%%

def get_mean_load_kW(df):
    """
    Delineate annual tendencies over days, weeks, and months and returns mean load in kW

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    result : DataFrame
        DataFrame containing the mean of the 6 smallest values of each column.
    """

    num_rows = df.shape[0]
    averages = []

    # Iterate over the DataFrame in chunks of 96 rows
    #if you want it per week you multiply the chunk_size by seven otherwise you keep 96 values per day
    chunk_size = 96
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get the current chunk of 96 rows
        chunk_kW = 4*chunk
        # Calculate the mean of the smallest values for each column
        average_of_smallest = chunk_kW.mean()
        
        averages.append(average_of_smallest)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    return result


#%% creating a benchmark over available years

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

print(type(Loads.index[0]))
# Obtain a typical year averaged
typical_year = f.typical_period(Loads,  "year")


Loads_2022 = Typo_loads_2022[Typology]
Loads_2023 = Typo_loads_2023[Typology]

#%%
# Replace zeros with NaN values
"""If you want to have both years instead of their average, change typical_year by Loads"""
df_nan = Loads.replace(0, np.nan)
df_nan.index = pd.to_datetime(df_nan.index, format='%d.%m.%Y %H:%M:%S')
week_ends_df = df_nan[(df_nan.index.weekday == 5) | (df_nan.index.weekday == 6)]  # 5 and 6 represent Saturday and Sunday respectively

Weekend_day_average_load = get_mean_load_kW(week_ends_df)

my_colors = sb.color_palette("hls", Weekend_day_average_load.shape[1])


plt.figure()
for i in range(Weekend_day_average_load.shape[1]):
    plt.plot(Weekend_day_average_load.iloc[:, i], c= my_colors[i], label=Weekend_day_average_load.columns[i])
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Etablissements")
plt.title("")
plt.show()

df = Weekend_day_average_load.copy()
# Plot boxplot
plt.figure()
boxplot = df.boxplot()
#plt.scatter(range(1, len(df.columns) + 1), means, color='red', label='Mean', zorder=3, s=10)
plt.xticks(ticks=range(1, len(df.columns) + 1), labels=df.columns, rotation=45)
plt.xlabel("Identifiants des consommateurs")
plt.ylabel("Charge [$kW_{el}$]")
plt.title("Distribution annuelle de la charge journalière - Ecoles")
plt.grid(axis="x")
#%%

# Replace zeros with NaN values
"""If you want to have both years instead of their average, change typical_year by Loads"""
df_nan = Loads.replace(0, np.nan)
df_nan.index = pd.to_datetime(df_nan.index, format='%d.%m.%Y %H:%M:%S')
week_ends_df = df_nan[(df_nan.index.weekday == 0) |(df_nan.index.weekday == 1) |(df_nan.index.weekday == 2) |(df_nan.index.weekday == 3) |(df_nan.index.weekday == 4)]  # 0-4 represent week days respectively

Weekend_day_average_load = get_mean_load_kW(week_ends_df)

my_colors = sb.color_palette("hls", Weekend_day_average_load.shape[1])


plt.figure()
for i in range(Weekend_day_average_load.shape[1]):
    plt.plot(Weekend_day_average_load.iloc[:, i], c= my_colors[i], label=Weekend_day_average_load.columns[i])
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Etablissements")
plt.title("")
plt.show()

df = Weekend_day_average_load.copy()
# Plot boxplot
plt.figure()
boxplot = df.boxplot()
#plt.scatter(range(1, len(df.columns) + 1), means, color='red', label='Mean', zorder=3, s=10)
plt.xticks(ticks=range(1, len(df.columns) + 1), labels=df.columns, rotation=45)
plt.xlabel("Identifiants des consommateurs")
plt.ylabel("Charge [$kW_{el}$]")
plt.title("Distribution annuelle de la charge journalière - Ecoles")
plt.grid(axis="x")

