# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:21:04 2024

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
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, True)

#%% creating a benchmark over available years

# parameters to change
Typology = "Apems"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

print(type(Loads.index[0]))
# Obtain a typical year averaged
typical_year = f.typical_period(Loads,  "year")


Loads_2022 = Typo_loads_2022[Typology]
Loads_2023 = Typo_loads_2023[Typology]

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

#%%

HP_mean_annual_cons = 12.5 #kWh/m2 from swedish study 

HP_vect = np.zeros(365*24*4)

"distinction between momments of the year"
HP_time_on = (31+28+31+30+31+31+30+31)*24*4
HP_vect[:(31+28+31)*24*4] = HP_mean_annual_cons/HP_time_on #kWh/m2
HP_vect[(31+28+31+30+31)*24*4:(31+28+31+30+31+30+31+31+30)*24*4]=HP_mean_annual_cons/HP_time_on #kWh/quarter of hour/m2
HP_vect[(31+28+31+30+31+30+31+31+30+31+30)*24*4:(31+28+31+30+31+30+31+31+30+31+30+31)*24*4]=HP_mean_annual_cons/HP_time_on #kWh/m2

"if no distinction is made throughout the year"
#HP_time_on = (31+28+31+30+31+30+31+31+30+31+30+31)*24*4 #full year
#HP_vect[:]=HP_mean_annual_cons/HP_time_on


HP_vect_power = 4*HP_vect #kW/m2

df_HP = pd.DataFrame(HP_vect)

"trying to remobe the heat pump only during week days"
#df_HP.index = pd.to_datetime(Loads_2023.index, format='%d.%m.%Y %H:%M:%S')
#df_HP[(df_HP.index.weekday == 5) | (df_HP.index.weekday == 6)] = 0
    
mean_HP_vect_power = get_mean_load_kW(df_HP)

plt.plot(HP_vect_power)
plt.show()

#%%
building = 3
mean_load = get_mean_load_kW(Loads_2023)

diff = mean_load.iloc[:,building].to_numpy()-mean_HP_vect_power.to_numpy()[:,0]
print(mean_load.iloc[:,building].to_numpy().shape)
print(mean_HP_vect_power.to_numpy().shape)
plt.figure()

plt.plot(mean_load.iloc[:,building].to_numpy(), color="blue")
plt.plot(diff, color="red")

plt.show()
