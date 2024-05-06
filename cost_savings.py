# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:02:20 2024

@author: mimag
"""

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
import baseload_analysis as b


#%% 
"""
Three ways of computing savings:
    1. reducing the baseload by some % analyze data to find 
    2. reducing maxima by some % of the monthly maxima
    3. reducing average consumption by some %

"""

#%%
"""data acquisition"""

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
Typo_loads_2022, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list, True)

#getting typologies from 2023
Typo_loads_2023, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list, True)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)

#%%

def get_baseload_2(df):
    """
    Delineate annual tendencies over days, weeks, and months

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
    chunk_size = 96
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get the current chunk of 96 rows
        
        # Calculate the 6 smallest values of each column
        smallest_values = chunk.iloc[:16, :] #if using 96 we get the daily average and if using nlargest(n) we get the n largest data points of the day
        #print(smallest_values)
        # Calculate the mean of the smallest values for each column
        average_of_smallest = smallest_values.mean()
        
        averages.append(average_of_smallest)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    return result



#%% 
# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = 4* Typo_all_loads[Typology]

Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]

df = Loads_last_year.astype(np.longdouble)

#%%Baseload savings
#print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload_2(df_no_duplicates)

baseloads.replace(0, np.nan, inplace=True)

baseload_min= np.min(baseloads, axis=0)

baseload_min_min = np.min(baseload_min)

baseload_avg = np.mean(baseload_min)

baseload_savings = baseload_min.copy()

# output in kW/m2 of energy savings
for i in range(baseloads.shape[1]):
    if baseload_min[i] > baseload_avg:
        # dragging the big consumers towards average value
        baseload_savings.iloc[i,:] = baseload_min[i] - baseload_avg
    else:
        # dragging lower consumers to best in class
        baseload_savings.iloc[i,:] = baseload_min[i] - baseload_min_min
    
"""
Now we have the baseloads, we want to compute how much savings by
a reduction of that baseload by a given percentage

"""

plt.bar(Loads.columns, baseload_savings)

#%% Peak shaving savings


# Initialize lists to store maximum values and corresponding indexes for each month
max_values = []
max_indices = []

# Loop through each month of the last year
for month in range(1, 13):
    # Slice the DataFrame for the current month
    df_month = Loads_last_year[Loads_last_year.index.month == month]

    # Find the maximum value and its index for each column (client)
    max_values_month = df_month.max()
    max_indices_month = df_month.idxmax()

    # Append maximum value and its index to the lists
    max_values.append(max_values_month)
    max_indices.append(max_indices_month)

# Convert lists to DataFrames
max_values_df = pd.DataFrame(max_values, index=range(1, 13))
max_indices_df = pd.DataFrame(max_indices, index=range(1, 13))

# Set the columns of the new DataFrames to be the same as the columns of Loads_last_year
#max_values_df.columns = Loads_last_year.columns
#max_indices_df.columns = Loads_last_year.columns

# Set the columns of the new DataFrames to be numerical
max_values_df.columns = range(1, len(Loads_last_year.columns) + 1)
max_indices_df.columns = range(1, len(Loads_last_year.columns) + 1)


max_values_df[max_values_df == 0] = np.nan

max_values_dfkW = max_values_df

# peak shaving new values
df_shaved = df.copy()


for i in range(max_values_dfkW.shape[0]):
    
    for j in range(df_shaved[df_shaved.index.month == i + 1].shape[1]):
        condition = (df_shaved.index.month == i + 1) & (df_shaved.iloc[:, j] > 0.9* max_values_dfkW.iloc[i, j])
        df_shaved.loc[condition, df_shaved.columns[j]] = 0.9 * max_values_dfkW.iloc[i, j]

power_economies= df-df_shaved
energy_economies = power_economies.mean(0)*365*24 #kWh/an/m2 (si normalisation préalable)
plt.bar(Loads.columns, energy_economies)
plt.xticks(rotation=45)
plt.show()
#%% plotting

# Initialize an empty list to store the energy economies for each factor
energy_economies_list = []

# Iterate over the factors from 0.9 to 0.4 with 10 intervals
for factor in np.linspace(0.9, 0.4, 10):
    # Apply peak shaving with the current factor
    df_shaved = df.copy()  # Reset df_shaved to the original DataFrame
    
    for i in range(max_values_dfkW.shape[0]):
        for j in range(df_shaved[df_shaved.index.month == i + 1].shape[1]):
            condition = (df_shaved.index.month == i + 1) & (df_shaved.iloc[:, j] > factor * max_values_dfkW.iloc[i, j])
            df_shaved.loc[condition, df_shaved.columns[j]] = factor * max_values_dfkW.iloc[i, j]

    # Calculate power economies
    power_economies = df - df_shaved
    
    # Calculate energy economies and append to the list
    energy_economies = power_economies.mean() * 365 * 24  # kWh/year/m2 (assuming prior normalization)
    energy_economies_list.append(energy_economies)

#%%
# Plot the energy economies for each factor
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0.9, 0.4, 10),energy_economies_list)
#plt.yscale("log")
plt.title('Energy Economies for Varying Shaving Factors')
plt.xlabel('Loads')
plt.ylabel('Energy Economies (kWh/year/m2)')
#plt.xticks(rotation=45)
plt.legend(Loads.columns)
plt.grid(axis='y')
plt.show()





