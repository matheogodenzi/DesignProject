# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:26:11 2024

@author: matheo & mika
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import pandas as pd
import os
from scipy.stats import shapiro
import seaborn as sb
"""functions imports"""

import functions as f
import process_data as p


#%% data acquisition
#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, True)

#%%   
#print(Typo_loads)


Typology = "Ecole"
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

#%%

# Specify the month you want to extract (e.g., January)
desired_month = 5

df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

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

# take instead only last 22 slices, so that only last year is counted
sliced_3d_array = sliced_3d_array[-22:, :, :]

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



#%% Simple plot with percentiles

x = np.array([i for i in range(96)])

for j in range(sliced_3d_array.shape[2]):
    
    plt.figure()
    for i in range(sliced_3d_array.shape[0]):
        plt.scatter(x, sliced_3d_array[i, :, j], c="royalblue", alpha=0.3, s=15)
        

    plt.plot(percentile_5[:,j], c="orange", label="$5^{ème}$ percentile")
    plt.plot(median_depth[:, j], c="red", label="Mediane")
    plt.plot(percentile_95[:,j], c="purple", label="$95^{ème}$ percentile")
    plt.xlabel("quarter of hours (to be changed)")
    plt.ylabel("load [$kWh_{el}/m^2$]")
    plt.grid()
    plt.legend()
    
    plt.show()

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



#%% Plotting with anomalies highlighted

x = np.arange(96)  # Using arange instead of list comprehension

hour_ticks = np.arange(0, 96, 8)


# Create hour labels (0, 1, 2, ..., 23)
hour_labels = [str(i) for i in range(0, 24, 2)]



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
            anomalies_mild_label = plt.scatter([], [], c="orange", label="Anomalies bénignes", alpha=0.7)
            anomalies_signi_label = plt.scatter([], [], c="red", label="Anomalies significatives", alpha=0.7)
            percentile_5_label = plt.plot([], [], c="orange", label="$5^{ème}$ percentile")
    # Plotting percentiles and median
    plt.plot(percentile_5[:,j], c="orange", label="$5^{ème}$ percentile")
    plt.plot(median_depth[:, j], c="red", label="Mediane")
    plt.plot(percentile_95[:,j], c="purple", label="$95^{ème}$ percentile")
    


    # Setting labels, legend, and grid
    plt.xticks(hour_ticks)
    plt.gca().set_xticklabels(hour_labels)
    plt.xlabel("Heures de la journée")
    plt.ylabel("Charge [$kWh_{el}/m^2$]")
    plt.grid()
    plt.legend(handles=[anomalies_mild_label, anomalies_signi_label])
    plt.show()

#%% Counting the anomalies per category

def count_occurrences(dataframe):
    # Sum along the first two dimensions to count occurrences of 1s and 2s
    count_ones = np.sum(dataframe == 1, axis=(0, 1))
    count_twos = np.sum(dataframe == 2, axis=(0, 1))

    return count_ones, count_twos

# Assuming your dataframe is called 'peak_anomaly_test' with shape (44, 96, 13)
# Call the function to count occurrences
count_ones, count_twos = count_occurrences(peak_anomaly_test)


#%%Counting occurencies for all months
Typology = "Ecole"
Loads = Typo_all_loads[Typology]

df = Loads.astype(np.longdouble)

# Initialize empty data frames to store results
count_ones_df = pd.DataFrame(columns=range(1, 13))  # Months as columns
count_twos_df = pd.DataFrame(columns=range(1, 13))  # Months as columns

# Loop through desired months
for desired_month in range(1, 13):
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

    # Extract all instances of the desired month
    month_df = df[df.index.month == desired_month]

    # Extract weekdays
    weekdays_df = month_df[month_df.index.weekday < 5]

    # Discard the 2024 value
    if desired_month == 1:
        weekdays_df = weekdays_df[:-1]
    else: 
        weekdays_df = weekdays_df[:]

    Daily_data = weekdays_df.to_numpy()

    Daily_data[Daily_data == 0] = np.nan

    # Define the number of rows in each slice
    rows_per_slice = 96

    # Calculate the number of slices
    num_slices = Daily_data.shape[0] // rows_per_slice

    # Slice the array and reshape to create a 3D array
    sliced_3d_array = Daily_data.reshape(num_slices, rows_per_slice, -1)
    
    # Only take the last year by slicing the last 22 slices
    sliced_3d_array = sliced_3d_array[-22:, :, :]

    
    # Calculate median, 5th and 95th percentile
    median_depth = np.nanmedian(sliced_3d_array, axis=0)
    percentile_5 = np.nanpercentile(sliced_3d_array, 5, axis=0)
    percentile_95 = np.nanpercentile(sliced_3d_array, 95, axis=0)

    # Threshold for classifying anomalies into significant or mild
    threshold = percentile_95 + ((1/3) * (percentile_95 - median_depth))

    # Create peak_anomaly_test array
    peak_anomaly_test = np.zeros_like(sliced_3d_array, dtype=int)

    # Compare sliced_3d_array with percentile_95 and assign values based on the condition
    peak_anomaly_test[sliced_3d_array > threshold] = 2
    peak_anomaly_test[(sliced_3d_array > percentile_95) & (sliced_3d_array <= threshold)] = 1

    # Call the function to count occurrences
    count_ones, count_twos = count_occurrences(peak_anomaly_test)

    # Append counts to data frames
    count_ones_df[desired_month] = count_ones
    count_twos_df[desired_month] = count_twos
    
count_ones_dfT = count_ones_df.T
count_twos_dfT = count_twos_df.T

count_ones_dfT[count_ones_dfT == 0] = np.nan    
count_twos_dfT[count_twos_dfT == 0] = np.nan

"""
#%% sum of mild and significant anomalies for each customer
total_mild_occurences = count_ones_dfT.sum(axis=0)
total_sign_occurences = count_twos_dfT.sum(axis=0)

# Generate HLS color palette with 13 colors
hls_palette = sb.color_palette("hls", 13)
#categories = list(total_mild_occurences.keys())
#values = list(total_mild_occurences.values())

# Plotting the bar plot
plt.bar(Loads.columns, total_mild_occurences, color=hls_palette[8])

# Adding labels and title
plt.xlabel('Consumers')
plt.ylabel('Occurences')
plt.title('Occurences of mild anomalies')

# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Displaying the plot
plt.show()
"""
#%% sum of mild and significant anomalies for each customer
total_mild_occurences = count_ones_dfT.sum(axis=0)
total_sign_occurences = count_twos_dfT.sum(axis=0)

#categories = list(total_mild_occurences.keys())
#values = list(total_mild_occurences.values())

# Plotting the bar plot
plt.bar(Loads.columns, total_sign_occurences, color='royalblue')

# Adding labels and title
plt.xlabel('Consommateurs')
plt.ylabel('Occurences')
plt.title('Occurences annuelles des anomalies significatives')

# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Displaying the plot
plt.show()

#%% Computing the averages
avg_ones = np.nanmean(count_ones_dfT, axis=1)
avg_twos = np.nanmean(count_twos_dfT, axis=1)
avg_twos = pd.Series(avg_twos)

# Ensure avg_twos has the correct index
avg_twos.index = range(1, len(avg_twos) + 1)
print(avg_twos.index)
#%% plotting the occurences
my_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']
# Generate HLS color palette with 13 colors
hls_palette = sb.color_palette("hls", Loads.shape[1])
"""
# Plot count_ones_df
plt.figure(figsize=(10, 6))
for client in count_ones_dfT.columns:
    plt.plot(count_ones_dfT.index, count_ones_dfT[client], label=Loads.columns[client-1], color=hls_palette[client-1])
plt.plot(avg_ones.index, avg_ones, label="Average", color="blue", linewidth=5, alpha=0.5)
plt.title('Occurrences of Mild Anomalies by Month')
plt.xlabel('Month')
plt.ylabel('Occurrences')
plt.xticks(range(1, 13))
plt.legend(title='Consumers', loc='center left', bbox_to_anchor=(1, 0.4))
plt.grid(True)
plt.show()"""

# Plot count_twos_df
plt.figure(figsize=(6, 5))
for i, client in enumerate(count_twos_dfT.columns):
    plt.plot(count_twos_dfT.index, count_twos_dfT[client], label=Loads.columns[client-1], color=my_palette[client-1])
plt.plot(avg_twos.index, avg_twos, label="Moyenne", color="blue", linewidth=5, alpha=0.5)
plt.title('Occurrences mensuelles des anomalies significatives')
plt.xlabel('Mois')
plt.ylabel('Occurrences')
plt.xticks(range(1, 13))
plt.legend(title='Consommateurs', loc='center left', bbox_to_anchor=(1, 0.4))
plt.grid(True)
plt.show()

#%% Overload peaks extraction

# Initialize lists to store maximum values and their indices for each client
max_values = []
max_indices = []

# Loop through desired months
for desired_month in range(1, 13):
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

    # Extract all instances of the desired month
    month_df = df[df.index.month == desired_month]

    # Calculate the maximum value and its index for each client (column)
    max_values_month = month_df.max()  # Maximum values for the current month
    max_indices_month = month_df.idxmax()  # Indices of the maximum values for the current month

    # Append the maximum values and their indices to the respective lists
    max_values.append(max_values_month)
    max_indices.append(max_indices_month)

# Print the maximum values and their indices for each client
for month, max_values_month, max_indices_month in zip(range(1, 13), max_values, max_indices):
    print(f"Month {month}:")
    print("Maximum values:")
    print(max_values_month)
    print("Indices of maximum values:")
    print(max_indices_month)
    print()


#%% Homemade try

# Convert the index of the DataFrame to a DatetimeIndex
Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]



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


# Print the DataFrames
print("Maximum Values for Each Month:")
print(max_values_df)
print("\nCorresponding Indices for Each Month:")
print(max_indices_df)



#%%
max_values_df[max_values_df == 0] = np.nan

max_values_dfkW = max_values_df * 4
avg_maxvalues = np.nanmean(max_values_dfkW, axis=1)
# Generate HLS color palette with 13 colors
hls_palette = sb.color_palette("hls", Loads.shape[1])

# Plot max_values_df
plt.figure(figsize=(10, 6))
for client in max_values_df.columns:
    plt.plot(max_values_dfkW.index, max_values_dfkW[client], label=Loads_last_year.columns[client-1], color=hls_palette[client-1])
plt.plot(max_values_dfkW.index, avg_maxvalues, label="Average", color="blue", linewidth=5, alpha=0.5)
plt.title('Maximum Values for Each Month by Client')
plt.xlabel('Month')
plt.ylabel('Maximum Load [$kW_{el}/m^2$]')
plt.xticks(range(1, 13))
plt.legend(title="Consumers", loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

#%%  sum of max_load for each customer
total_max_values = max_values_dfkW.sum(axis=0)


#categories = list(total_mild_occurences.keys())
#values = list(total_mild_occurences.values())

# Plotting the bar plot
plt.bar(Loads.columns, total_max_values, color=hls_palette[8])

# Adding labels and title
plt.xlabel('Consumers')
plt.ylabel('Sum of maximal loads [$kW_{el}/m^2$]')
plt.title('Total power of monthly maximal loads')

# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Displaying the plot
plt.show()

#%%

"""
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
print(type(Building_dict_2023), type(LoadCurve_2022_dict), type(Typo_list))
#getting typologies from 2022
Typo_loads_2022 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)

"""

"""#%%

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

"""
