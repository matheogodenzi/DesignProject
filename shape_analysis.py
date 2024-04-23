# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:09:03 2024

@author: matheo
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import sklearn as skl
import pandas as pd
import os
import seaborn as sb
from datetime import datetime
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


"""data acquisition"""

#%%

# DEFINING PATHS
## Generic path of the folder in your local terminal 
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)


## Creating specificpath for each commune
renens = parent_directory + "\\Renens"
ecublens = parent_directory + "\\Ecublens"
crissier = parent_directory + "\\Crissier"
chavannes = parent_directory + "\\Chavannes"

Commune_paths = [renens, ecublens, crissier, chavannes]


## reading excel files 
load_data_2023 = []
load_data_2022 = []
building_data_2023 = []
pv_2022 = []


for i, commune in enumerate(Commune_paths):
    
    # extracting load curves 
    load_2023 = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2)
    load_2023.set_index("Date", inplace=True)
    load_2022 = pd.read_excel(commune+"\\"+ f.get_variable_name(commune, globals()) +"_cch_podvert_2022.xlsx", sheet_name=2)
    load_2022.set_index("Date", inplace=True)
    
    given_file ="\\" + f.get_variable_name(commune, globals()) + "_cch_plus_20MWh_complement"
    pv_commune = []
    for root, dirs, files in os.walk(commune):
        if given_file in files: 
            file_path = os.path.join(root, given_file)
            try:
                # Read the Excel file using pandas
                pv_prod_2022 = pd.read_excel(file_path)
                pv_prod_2022.set_index("Date", inplace=True)
                # Perform actions with the DataFrame 'df'
                print(f"Successfully read {given_file} in {root}.")
                # Add more code to work with the DataFrame if needed
                pv_2022.append(pv_prod_2022)
                pv_commune.append(f.get_variable_name(commune, globals()))
            except Exception as e:
                # Handle any exceptions raised during reading or processing
                print(f"An error occurred while reading {given_file} in {root}: {e}")
        else:
            print(f"{given_file} not found in {root}.")
            # Add code to handle this case or simply pass
    
        
    # extracting buildings
    buildings = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=0)
    
    # storing data 
    load_data_2023.append(load_2023)
    load_data_2022.append(load_2022)
    
    building_data_2023.append(buildings)


LoadCurve_2023_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2023[i] for i in range(len(Commune_paths))}
LoadCurve_2022_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2022[i] for i in range(len(Commune_paths))}
Building_dict_2023 = {f.get_variable_name(Commune_paths[i], globals()): building_data_2023[i] for i in range(len(Commune_paths))}
pv_2022_dict = {pv_commune[i]: pv_2022[i] for i in range(len(pv_commune))}

print(pv_2022_dict)

#%% get all typologies sorted for all provided year 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]

#getting typologies from 2022
Typo_loads_2022 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)


#%%

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

test =np.log(df.iloc[:, 0].values)

# Plot histogram
plt.hist(test, bins=30, density=True)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

#%%


# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

data = df.iloc[:,0]
plt.plot(data[1000:1344])
plt.show()

# Calculate the 25th, 50th (median), and 75th percentiles
p5 = np.percentile(data, 5)
p50 = np.percentile(data, 50)  # Equivalent to np.median(data)
p75 = np.percentile(data, 75)
p98 = np.percentile(data, 98)

# Plot histogram
plt.hist(data, bins=30, density=True)
# Plot vertical lines
plt.axvline(x=p5, color='r', linestyle='--')  # Vertical line at x=2
plt.axvline(x=p98, color='g', linestyle=':')   # Vertical line at x=4

plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


#%%

# Specify the month you want to extract (e.g., January)
desired_month = 12

Overall_means = []
for desired_month in range(1, 13):
    df.index = pd.to_datetime(df.index)
    
    # Extract all instances of the desired month
    month_df = df[df.index.month == desired_month]
    
    # Extract weekdays
    weekdays_df = month_df[month_df.index.weekday < 5]
    
    #discarding the 2024 value
    if desired_month == 1 :
        weekdays_df = weekdays_df[:-1]
    else : 
        weekdays_df = weekdays_df[:]
    
    Daily_data = weekdays_df.to_numpy()
    
    # Define the number of rows in each slice
    rows_per_slice = 96
    
    # Calculate the number of slices
    num_slices = Daily_data.shape[0] // rows_per_slice
    
    # Slice the array and reshape to create a 3D array
    sliced_3d_array = Daily_data.reshape(num_slices, rows_per_slice, -1)
    
    #calculate median, 5 and 95 percentile
    
    # Calculate median throughout the depth dimension
    median_depth = np.median(sliced_3d_array, axis=0)
    # Calculate 5th and 95th percentiles throughout the depth dimension
    percentile_5 = np.percentile(sliced_3d_array, 5, axis=0)
    percentile_95 = np.percentile(sliced_3d_array, 95, axis=0)
    
    
    daily_mean = sliced_3d_array.mean(axis=0)
    
    daily_mean = daily_mean.mean(axis=1)
    
    Overall_means.append(daily_mean)

daily_mean = np.mean(np.array(Overall_means), axis=0)

plt.figure()
plt.plot(daily_mean/np.max(daily_mean), label="Mean daily profile", c="royalblue")
tick_labels = ["0"+str(i)+":00" if i < 10 else str(i)+":00" for i in range(1, 25, 2)]
tick_positions = [3 +i*8 for i in range(12)]
plt.xticks(tick_positions, tick_labels, rotation=45)
plt.xlabel("Hours of the day")
plt.ylabel("Relative load [-]")
plt.title("Generic Day")
plt.legend()
plt.grid()
plt.show()


#%% Generic week plot 

# Specify the month you want to extract (e.g., January)

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)


df.index = pd.to_datetime(df.index)

# Extract all instances of the desired month
array = df.to_numpy()
array = array[:7*96*52, :]

# Define the number of rows in each sliceshape ana
rows_per_slice = 7*96

# Calculate the number of slices
num_slices = array.shape[0] // rows_per_slice

# Slice the array and reshape to create a 3D array
sliced_3d_array = array.reshape(num_slices, rows_per_slice, -1)

#calculate median, 5 and 95 percentile


weekly_mean = sliced_3d_array.mean(axis=0)

weekly_mean = weekly_mean.mean(axis=1)

plt.figure()
plt.plot(weekly_mean/np.max(weekly_mean), label="Mean weekly profile", c="orange")
tick_labels = ['Saturday', 'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
tick_positions = [i*96 +96/2 for i in range(7)]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.xlabel("Days of the week")
plt.ylabel("Relative load [-]")
plt.legend()
plt.title("Generic week")
plt.grid()
plt.show()

#%% Generic year plot 

# parameters to change
Typology = "Ecole"
Period = "week"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)


df.index = pd.to_datetime(df.index)

# Obtain a typical year
typical_year = f.typical_period(df,  "year")

# Annual weekly smoothing of the loag curve
tendency = f.period_tendencies(typical_year, Period)

#f.plot_mean_load(None, tendency, period=Period, Typology=Typology)
Annual_weekly_mean = tendency.mean(axis=1)


plt.figure()
plt.plot(Annual_weekly_mean/np.max(Annual_weekly_mean), label="Mean annual profile", c= "salmon")
plt.xlabel("Weeks of the year")
plt.ylabel("Relative baseload [-]")
plt.title("Generic year")
plt.legend()
plt.grid()
plt.show()

