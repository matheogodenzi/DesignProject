# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:53:58 2024

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
import matplotlib as mpl

"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


"""data acquisition"""

#%% data acquisition
#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict,True)

#%% consumption dictionnary sorting
"""
typo_loads2, ID_mapping = p.discriminate_typologies2(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

Cons_list = ["bas", "moyen", "haut", "fort"]

Cons_loads_2022 = p.discriminate_conslevels(Building_dict_2023, LoadCurve_2022_dict, Cons_list, ID_mapping)

Cons_loads_2023 = p.discriminate_conslevels(Building_dict_2023, LoadCurve_2023_dict, Cons_list, ID_mapping)

Cons_all_loads = {}

for cons in Cons_list:
    Cons_all_loads[cons] = pd.concat([Cons_loads_2022[cons], Cons_loads_2022[cons]], axis=0)

"""

#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology] *4 *1000 #[W/m2]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 4].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)

#%% creating a benchmark over available years

# Obtain a typical year
typical_year = f.typical_period(Loads,  "year")

#%% Comparing typologies to distinguish benchmarks
typical_day = f.typical_period(typical_year, Period)

#color list 
color = ["darkblue", "royalblue", "green", "yellow", "orange", "red", "purple", "pink"]
Typo_list = ["Culture", "Apems", "Commune", "Admin", "Buvette", "Parking", "Sport"]
#initiating figure
plt.figure()
for i, typo in enumerate(Typo_list) : 
    
    loads = Typo_all_loads[typo] *4 *1000 #[W/m2]
    # Obtain a typical year
    t_year = f.typical_period(loads,  "year")
    #obtain typical day 
    t_day = f.typical_period(t_year, "day")
    
    word_to_find = " Renens"
    
    if typo == "Ecole":
        # Get the column names containing the word
        matching_columns = [col for col in t_day.columns if word_to_find.lower() in col.lower()]
        
        # Set colors for matching and non-matching columns
        colors = ['pink' if col in matching_columns else 'darkblue' for col in t_day.columns]
        for i, col in enumerate(t_day.columns):
            t_day[col].plot(color=colors[i], label=typo)

    else:
        plt.plot(t_day, color=color[i], label=typo)
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Typologies")
plt.tight_layout(rect=[0, 0, 1, 2.3])
plt.xlabel("Heures")
plt.ylabel("Mean daily consumption [$W_{el}/m^2$]")
plt.title("Mean day consumption - typology distinction")
plt.yscale("log")
plt.grid()
plt.show()


#%%  benchmarks
typical_day = f.typical_period(typical_year, Period)

#color list 
color = ["darkblue", "royalblue", "green", "yellow", "orange", "red", "purple", ]

#initiating figure
plt.figure()
for i, typo in enumerate(Typo_list) : 
    
    loads = Typo_all_loads[typo]
    # Obtain a typical year
    t_year = f.typical_period(loads,  "year")
    #obtain typical day 
    t_day = f.typical_period(t_year, "day")
    

    

    for j, col in enumerate(t_day.columns):
        t_day[col].plot(color=color[i], label=col)


plt.yscale("log") 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Typologies")
plt.tight_layout(rect=[0, 0, 1, 2.3])
plt.xlabel("Heures")
plt.ylabel("Mean daily consumption [$kWh_{el}/m^2$]")
plt.title("Mean day consumption - typology distinction")
plt.grid()
plt.show()

#%%  benchmarks

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
# Generate a color palette using 'tab20' and another colormap
def get_contrasting_colors(n_colors):
    # Use multiple colormaps to get enough colors
    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Paired]
    colors = []
    for cmap in colormaps:
        colors.extend([cmap(i) for i in range(cmap.N)])
        if len(colors) >= n_colors:
            break
    return colors[:n_colors]

# Number of colors needed
n_colors = 29

# Get the colors
colors = get_contrasting_colors(n_colors)
#%%

# Get the colors
colors = get_contrasting_colors(n_colors)

#color = sb.color_palette("husl", 29, 1)
#color = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
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
        t_day[col].plot(label=col, color=colors[colori])
        colori += 1

#plt.yscale("log")

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
plt.ylabel("Mean daily consumption [$kWh_{el}/m^2$]")
plt.title("Mean daily consumption")
plt.grid()
plt.show()

#%% plotting one specific load over given benchmark 

Load1 = typical_day.iloc[:, 7].to_frame()
f.plot_mean_load(Load1, typical_day, Period, Typology)



#%% test of plotting benchark averages alone 

# Annual weekly smoothing of the loag curve
tendency = f.period_tendencies(typical_year, Period)

f.plot_mean_load(None, tendency, period=Period, Typology=Typology)


#%% All benchmark displayed  for a day


#data = Typo_loads["Apems"]
Period = "day"

tendency_day = f.period_tendencies_new(Loads, Period)
data_day = f.typical_period(Loads, Period)

# Convert timestamp index to string index
data_day.index = data_day.index.strftime('%Y-%m-%d %H:%M:%S')

# Typical day for all infrastructures 
f.plot_typical_day(data_day, Typology)


#daily smoothing along the year for all insfrastrctures 
f.plot_tendency(tendency_day.head(365), title="Load curve weekly average for "+Typology+"s", period=Period, show_legend=True)


#%% All Benchmark plotted for a week 

#data = Typo_loads["Apems"]
Period = "day"

# Annual weekly smoothing of the loag curve
tendency_week = f.period_tendencies(Loads, Period)
data_week = f.typical_period(Loads, Period)


#typical week for all infrastructures 
#f.plot_typical_week(data_week, Typology)

#weekly smoothing along the year for all insfrastrctures 
f.plot_tendency(tendency_week, title="Load curve weekly average for "+Typology+"s", period=Period, show_legend=True)


#%% extraction of a given week 


Typology = "Ecole"
Period = "week"

# smoothing calculations
Loads = Typo_all_loads[Typology]

typical_week = f.typical_period(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 6].to_frame()
#print(single_load)
interest_period = aa.extract_period(single_load, pd.to_datetime("14.01.2023 00:15:00"), pd.to_datetime("21.01.2023 00:00:00"))

# plotting 
updated_tendency = f.plot_mean_load(interest_period, typical_week, Period, Typology)

#%% Extracting all instances of the same exact time

#time_of_interest = aa.extract_time(Load, pd.Timestamp('00:00:00'))


#%% align years



for column_name in Loads.columns : 
    
    # Calculate the 10th percentile value for the chosen column
    percentile_10 = Loads[column_name].quantile(0.2)
    
    # Extract the values from the chosen column that are less than or equal to the 10th percentile
    lowest_10_percentile = Loads[Loads[column_name] <= percentile_10][column_name]
    
    plt.plot(lowest_10_percentile)
    plt.show()
    exit()



#%%

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
        smallest_values = chunk.apply(lambda x: x.nsmallest(24)) #if using 96 we get the daily average and if using nlargest(n) we get the n largest data points of the day

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
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

print(df[df.index.duplicated()])

# Remove duplicate indices
df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload(df_no_duplicates)


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

#selecting a subset 
#result = result.iloc[:, 1:4]

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
print(f"my colors : {my_colors}")


plt.figure()

for i, column in enumerate(result.columns):
    plt.plot((result[column].head(53).values + result[column].tail(53).values) / 2, color=my_colors[i])

plt.yscale('log')
plt.grid(which="both", alpha=0.5)
plt.xlabel("weeks of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.title("Annual baseload variation - Schools")
plt.legend(result.columns, loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

#%% Two consecutive years evolution
plt.figure
plt.plot(result.head(53))
plt.plot(result.tail(53))
plt.grid()
plt.xlabel("weeks of the year")
plt.show()



