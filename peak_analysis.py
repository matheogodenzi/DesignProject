# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:26:11 2024

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
from scipy.stats import shapiro

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
desired_month = 11

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

Daily_data[Daily_data == 0] = np.nan

# Define the number of rows in each slice
rows_per_slice = 96

# Calculate the number of slices
num_slices = Daily_data.shape[0] // rows_per_slice

# Slice the array and reshape to create a 3D array
sliced_3d_array = Daily_data.reshape(num_slices, rows_per_slice, -1)

#calculate median, 5 and 95 percentile

# Calculate median throughout the depth dimension
median_depth = np.nanmedian(sliced_3d_array, axis=0)
# Calculate 5th and 95th percentiles throughout the depth dimension
percentile_5 = np.nanpercentile(sliced_3d_array, 5, axis=0)
percentile_95 = np.nanpercentile(sliced_3d_array, 95, axis=0)

#%% testing the normality of the data

# Create an empty array to store the results
normality_results_array = np.zeros((sliced_3d_array.shape[1], sliced_3d_array.shape[2]), dtype=int)

# Iterate over the second dimension
for j in range(sliced_3d_array.shape[1]):
    # Iterate over the third dimension
    for k in range(sliced_3d_array.shape[2]):
        # Extract the slice
        slice_data = sliced_3d_array[:, j, k]
        # Perform Shapiro-Wilk test
        stat, p = shapiro(slice_data)
        # Assign 1 if normally distributed, 0 otherwise
        normality_results_array[j, k] = int(p > 0.05)

print("Result array:")
print(normality_results_array)

#%% Extracting and classifying anomalies

# threshold for classifying anomalies into significative of mild
threshold = percentile_95 + ((1/3) * (percentile_95 - median_depth))

# Create peak_anomaly_test array
peak_anomaly_test = np.zeros_like(sliced_3d_array, dtype=int)

# Compare sliced_3d_array with percentile_95 and assign values based on the condition
peak_anomaly_test[sliced_3d_array > threshold] = 2
peak_anomaly_test[(sliced_3d_array > percentile_95) & (sliced_3d_array <= threshold)] = 1

#%% creating plotting curves
anomalies_mild = sliced_3d_array.copy()
anomalies_signi = sliced_3d_array.copy()

# Mask the values based on the conditions specified by peak_anomaly_test
anomalies_mild[peak_anomaly_test != 1] = np.nan
anomalies_signi[peak_anomaly_test != 2] = np.nan

#%%

x = np.array([i for i in range(96)])

for j in range(sliced_3d_array.shape[2]):
    
    plt.figure()
    for i in range(sliced_3d_array.shape[0]):
        plt.scatter(x, sliced_3d_array[i, :, j], c="royalblue", alpha=0.3, s=15)
        

    plt.plot(percentile_5[:,j], c="orange", label="5% percentile")
    plt.plot(median_depth[:, j], c="red", label="median")
    plt.plot(percentile_95[:,j], c="purple", label="95% percentile")
    plt.xlabel("quarter of hours (to be changed)")
    plt.ylabel("load [$kWh_{el}/m^2$]")
    plt.grid()
    plt.legend()
    
    plt.show()

#%% additional pplots of interest 
plt.plot(month_df.head(2*96).values)
plt.show()

plt.plot(weekdays_df.iloc[19*96:20*96].values)
plt.show()

daily_mean = sliced_3d_array.mean(axis=0)

daily_mean = daily_mean.mean(axis=1)

plt.plot(daily_mean)
plt.show()


#%% plotting with anomalies highlighted

x = np.arange(96)  # Using arange instead of list comprehension

# Initialize the labels for anomalies
anomalies_mild_label = None
anomalies_signi_label = None

for j in range(sliced_3d_array.shape[2]):
    plt.figure()
    
    # Plotting the data points
    for i in range(sliced_3d_array.shape[0]):
        plt.scatter(x, sliced_3d_array[i, :, j], c="royalblue", alpha=0.3, s=10)
        plt.scatter(x, anomalies_mild[i, :, j], c="orange", alpha=0.7, s=10)
        plt.scatter(x, anomalies_signi[i, :, j], c="red", alpha=0.7, s=10)
        
        # Set labels for anomalies outside the loop
        if i == 0:
            anomalies_mild_label = plt.scatter([], [], c="orange", label="Anomalies Mild", alpha=0.7)
            anomalies_signi_label = plt.scatter([], [], c="red", label="Anomalies Significant", alpha=0.7)
    
    # Plotting percentiles and median
    plt.plot(percentile_5[:, j], c="orange", label="5% percentile")
    plt.plot(median_depth[:, j], c="red", label="median")
    plt.plot(percentile_95[:, j], c="purple", label="95% percentile")
    
    # Setting labels, legend, and grid
    plt.xlabel("Quarter of hours (to be changed)")
    plt.ylabel("Load [$kWh_{el}/m^2$]")
    plt.grid()
    plt.legend(handles=[anomalies_mild_label, anomalies_signi_label])
    
    plt.show()

#%%

plt.plot(Daily_data[:,9])


