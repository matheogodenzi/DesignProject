# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:30:43 2024

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

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Commune2", "Buvette", "Parking"]

#getting typologies from 2022
Typo_loads_2022 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)


#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Apems"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 0].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)


#%% creating a benchmark over available years

# Obtain a typical year averaged
typical_year = f.typical_period(Loads,  "year")


Loads_2022 = Typo_loads_2022[Typology]
Loads_2023 = Typo_loads_2023[Typology]
#%%


# Convert index to datetime type if it's not already
Loads_2022.index = pd.to_datetime(Loads_2022.index)

baseload_df = pd.DataFrame(columns=Loads_2022.columns)

i = 0
for (year, month, day), group in Loads_2022.groupby([Loads_2022.index.year, Loads_2022.index.month, Loads_2022.index.day]):
    #data_by_day.append(group)
    
    # Calculate the average of the smallest values of each column within each group
    average_of_smallest_values = group.apply(lambda x: x.nsmallest(6).mean())
    
    
    print(day, month)
    # Display the result
    print(average_of_smallest_values)
    print()
    i+=1
    

"""
for column_name in Loads_2022.columns : 
    
    # Calculate the 10th percentile value for the chosen column
    percentile_10 = Loads[column_name].quantile(0.1)
    
    # Extract the values from the chosen column that are less than or equal to the 10th percentile
    lowest_10_percentile = Loads[Loads[column_name] <= percentile_10][column_name]
    
    plt.plot(lowest_10_percentile)
    plt.show()
    exit()
"""
#%%


# Create an empty DataFrame with columns for 'Day' and 'Month'
results_df = pd.DataFrame(columns=Loads_2022.columns)

df = Loads_2022.astype(np.longdouble)

for (year, month, day), group in df.groupby([df.index.year, df.index.month, df.index.day]):
    # Calculate the average of the smallest values of each column within each group
    
    average_of_smallest_values = group.apply(lambda x: x.nsmallest(3).mean())
    print(group.shape)
    # Create a DataFrame for the current iteration
    current_df = pd.DataFrame(average_of_smallest_values).T  # Transpose to make it a row
    #current_df['Day'] = day
    #current_df['Month'] = month
    
    # Set datetime index for the current iteration
    current_df.index = pd.to_datetime([f'{year}-{month}-{day}'])
    
    # Append the DataFrame to the results DataFrame
    results_df = pd.concat([results_df, current_df])
    
    
# Display the filled DataFrame
print(results_df)

#%%
plt.figure()
plt.plot(results_df)
plt.legend([i for i in range(results_df.shape[0])])
plt.show

plt.figure()
plt.plot(results_df)
plt.show()

#%% test 

Period = "day"

tendency_day = f.period_tendencies_new(Loads_2022, Period)


plt.plot(results_df.iloc[:,3].values[:] - tendency_day.iloc[:,3].values[:])
plt.show()

#%%

import pandas as pd

def get_baseload(df):
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
        smallest_values = chunk.apply(lambda x: x.nsmallest(6)) #if using 96 we get the daily average and if using nlargest(n) we get the n largest data points of the day

        # Calculate the mean of the smallest values for each column
        average_of_smallest = smallest_values.mean()
        
        averages.append(average_of_smallest)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    return result





#%%

df = Loads.astype(np.longdouble)

print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

#%%
baseloads = get_baseload(df_no_duplicates)

plt.figure()
plt.plot(baseloads)
plt.show()

#%% Linear regressions 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload(df_no_duplicates)


# smoothing calculations
Loads = Typo_all_loads[Typology]


df  = baseloads

# Define your color palette
palette = sns.color_palette("bright", df.shape[0])
# Create an iterator to cycle through the colors
color_iterator = cycle(palette)
# Create subplots
fig, ax = plt.subplots(figsize=(25, 10))

# Perform linear regression and plot for each column
for i, column in enumerate(df.columns):
    #if i != 6 and i != 7 : 
        # Get the next color from the iterator
        color = next(color_iterator)
        X = np.array(df.index).reshape(-1, 1)   # Independent variable
        y = df[column].values.reshape(-1, 1)              # Dependent variable
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Plot data points
        ax.scatter(X, y, color=color, alpha=0.3)
        
        # Plot regression line
        ax.plot(X, model.predict(X), color=color, label=column, linewidth=3, alpha = 1)
        
        # Set labels and title
        ax.set_title(f'{column}')
        ax.set_xlabel('Independent Variable')
        ax.set_ylabel('Dependent Variable')
        ax.legend()

plt.tight_layout()
plt.show()

