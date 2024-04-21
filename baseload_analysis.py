# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:30:43 2024

@author: matheo
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import seaborn as sb
import seaborn as sns
from sklearn.linear_model import LinearRegression

"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


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
Typology = "Ecole"
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


#%% Linear regressions 

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

#print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload_2(df_no_duplicates)


# smoothing calculations
Loads = Typo_all_loads[Typology]


df  = baseloads

# Define your color palette
palette = sns.color_palette("hls", df.shape[1])
# Create an iterator to cycle through the colors
#color_iterator = cycle(palette)
# Create subplots
fig, ax = plt.subplots(figsize=(8, 5))

coef_df =  pd.DataFrame({'slope': [], 'y-intercept': []})

# Perform linear regression and plot for each column
for i, column in enumerate(df.columns):
    #if i in [4, 5, 8, 9, 11, 13]: 
            #plt.ylim(6e-13, 3e-12)
    #if i in [6, 7, 10, 12]: 
    #if i in [0, 1, 2, 3]:
            #plt.ylim(0, 7e-15)
            
            # Replace 0 values with NaN
            infra = df[column].copy()
            # Convert zeros to NaN
            infra.replace(0, np.nan, inplace=True)
            
            # Drop NaN values
            infra.dropna(inplace=True)
    
            X = np.array(infra.index).reshape(-1, 1)   # Independent variable
            print(X)
            y = infra.values.reshape(-1, 1)              # Dependent variable
            
            # Plot data points
            ax.scatter(X, y, color=palette[i], alpha=0.3, s=10)
            
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Extract regression coefficients
            coefficients = model.coef_
            intercept = model.intercept_
            
            specific_values = {'slope': coefficients[0] , 'y-intercept': intercept}

            coef_df = coef_df.append(specific_values, ignore_index=True)
                        
            # Plot regression line
            ax.plot(X, model.predict(X), c=palette[i], label=column, linewidth=2, alpha = 1)
            
            # Set labels and title
            ax.set_title(f'{column}')
            ax.set_xlabel('Days')
            ax.set_ylabel('Baseload [$kWh_{el}/m^2$]')
            ax.legend()

#plt.ylim(0, 3e-12)
plt.yscale("log")
plt.title("All Schools")
# Place legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(which='both')
plt.tight_layout()
plt.show()



#%% average weekly baseloads



# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload_2(df_no_duplicates)


# smoothing calculations
Loads = Typo_all_loads[Typology]

num_rows = baseloads.shape[0]
chunk_size = 7
df = baseloads 

averages = []
# Iterate over the DataFrame in chunks of 96 rows
for i in range(0, num_rows, chunk_size):
    chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
    chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
    averages.append(chunk_avg)  # Append the averages to the list
    print(i)
# Concatenate the averages into a single DataFrame
result = pd.concat(averages, axis=1).T


# Define your color palette
my_colors = sb.color_palette("hls", result.shape[1])
#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#palette = sns.color_palette("YlGnBu_r", result.shape[1])

plt.figure
plt.plot(result.head(53))
plt.plot(result.tail(53))
plt.grid()
plt.xlabel("weeks of the year")
plt.show()

plt.figure()

for i, column in enumerate(result.columns):
    plt.plot((result[column].head(53).values + result[column].tail(53).values) / 2, color=my_colors[i])

#plt.yscale('log')

plt.grid(which="both", alpha=0.5)
plt.xlabel("weeks of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.title("Annual baseload variation - Schools")
plt.legend(result.columns, loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


#%% average monthly baseload 

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = f.get_baseload_2(df_no_duplicates)


# smoothing calculations
Loads = Typo_all_loads[Typology]

num_rows = baseloads.shape[0]
chunk_size = 30
df = baseloads 

averages = []
# Iterate over the DataFrame in chunks of 96 rows
for i in range(0, num_rows, chunk_size):
    chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
    chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
    averages.append(chunk_avg)  # Append the averages to the list
    print(i)
# Concatenate the averages into a single DataFrame
result = pd.concat(averages, axis=1).T


# Define your color palette
palette = sns.color_palette("bright", 2*result.shape[0])


plt.figure
plt.plot(result.head(13))
plt.plot(result.tail(13))
plt.grid()
plt.xlabel("weeks of the year")
plt.show()

plt.figure()
plt.semilogy((result.head(13).values+result.tail(13).values)/2)
plt.grid()
plt.xlabel("weeks of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.show()

#%% average daily baeload 


# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

#print(df[df.index.duplicated()])

# Remove duplicate indices
#df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = f.get_baseload_2(df)


# smoothing calculation


#selecting a subset 
#result = result.iloc[:, :]

# Define your color palette
my_colors = sb.color_palette("hls", result.shape[1])
#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#palette = sns.color_palette("YlGnBu_r", result.shape[1])

plt.figure
plt.plot(baseloads.head(365))
plt.plot(baseloads.tail(365))
plt.grid()
plt.xlabel("weeks of the year")
plt.show()

plt.figure()

for i, column in enumerate(result.columns):
    #if i in [4, 5, 8, 9, 11, 13]: 
    #if i in [6, 7, 10, 12]: 
    #if i in [0, 1, 2, 3]:
    
            if column == "S202" or column == "S301":
                plt.plot((baseloads[column].tail(365).values), color=my_colors[i], label=column)
            else : 
                plt.plot((baseloads[column].head(365).values + baseloads[column].tail(365).values) / 2, color=my_colors[i], label=column)

#plt.yscale('log')

plt.grid(which="both", alpha=0.5)
plt.xlabel("Days of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.title("Lower medium-level consumers - Schools").set_position([0.55, 1])
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.legend()
#plt.subplots_adjust(top=2)
plt.show()


