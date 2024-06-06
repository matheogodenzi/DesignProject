# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:22:56 2024

@author: mimag
"""

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
from total_consumption import get_score
"""functions imports"""

import functions as f
import process_data as p


#%% data acquisition
#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, False)

#%%   
#print(Typo_loads)


Typology = "Culture"

Loads = Typo_all_loads[Typology]
Loads_buv = Typo_all_loads["Buvette"]
Loads_sport = Typo_all_loads["Sport"]
Loads_parking = Typo_all_loads["Parking"]
Loads_unique = pd.concat([Loads_buv, Loads_sport, Loads_parking], axis=1)

Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]

df = Loads_last_year.astype(np.longdouble)

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
    plt.figure(figsize=(6,5), dpi=300)
    
    # Plotting the data points
    for i in range(sliced_3d_array.shape[0]):
        plt.scatter(x, sliced_3d_array[i, :, j], c="royalblue", alpha=0.3, s=10)
        plt.scatter(x, anomalies_mild[i, :, j], c="orange", alpha=0.7, s=10)
        plt.scatter(x, anomalies_signi[i, :, j], c="red", alpha=0.7, s=10)
        
        # Set labels for anomalies outside the loop
        if i == 0:
            anomalies_mild_label = plt.scatter([], [], c="orange", label="Anomalies bénignes", alpha=0.7)
            anomalies_signi_label = plt.scatter([], [], c="red", label="Anomalies significatives", alpha=0.7)
            
    # Plotting percentiles and median
    plt.plot(percentile_5[:,j], c="orange", label="$5^{ème}$ percentile")
    plt.plot(median_depth[:, j], c="red", label="Médiane")
    plt.plot(percentile_95[:,j], c="purple", label="$95^{ème}$ percentile")
    percentile_5_line, = plt.plot(percentile_5[:,j], c="orange", label="$5^{ème}$ percentile")
    median_line, = plt.plot(median_depth[:, j], c="red", label="Médiane")
    percentile_95_line, = plt.plot(percentile_95[:,j], c="purple", label="$95^{ème}$ percentile")

    

    # Setting labels, legend, and grid
    plt.xticks(hour_ticks)
    plt.gca().set_xticklabels(hour_labels)
    plt.xlabel("Heures de la journée")
    plt.ylabel("Charge [$kWh_{el}/m^2$]")
    plt.grid()
    plt.legend(handles=[anomalies_mild_label, anomalies_signi_label, percentile_5_line, median_line, percentile_95_line],
               loc='center left', bbox_to_anchor=(1, 0.84))
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

def calculate_max_values(df):
    # Initialize lists to store maximum values and corresponding indexes for each month
    max_values = []
    max_indices = []

    # Loop through each month of the last year
    for month in range(1, 13):
        # Slice the DataFrame for the current month
        df_month = df[df.index.month == month]

        # Find the maximum value and its index for each column (client)
        max_values_month = df_month.max()
        max_indices_month = df_month.idxmax()

        # Append maximum value and its index to the lists
        max_values.append(max_values_month)
        max_indices.append(max_indices_month)

    # Convert lists to DataFrames
    max_values_df = pd.DataFrame(max_values, index=range(1, 13))
    max_indices_df = pd.DataFrame(max_indices, index=range(1, 13))

    # Set the columns of the new DataFrames to be numerical
    max_values_df.columns = range(1, len(df.columns) + 1)
    max_indices_df.columns = range(1, len(df.columns) + 1)

    # Handle zero values, if needed
    max_values_df[max_values_df == 0] = np.nan

    return max_values_df

def calculate_peak_economies(df, max_values_df, factor_df):
    """
    Calculate peak economies based on a factor and maximum values DataFrame.

    Parameters:
    df (pd.DataFrame): Original DataFrame.
    max_values_df (pd.DataFrame): DataFrame containing maximum values.
    factor_df (pd.DataFrame): Factor for peak shaving.

    Returns:
    pd.DataFrame: DataFrame containing peak economies.
    """
    df_shaved = df.copy()

    for i in range(1, 13):  # Loop over months (1 to 12)
        month_condition = (df_shaved.index.month == i)
        
        for j in range(df_shaved.shape[1]):  # Loop over columns
            factor_value = factor_df.iloc[i-1, j]
            max_value = max_values_df.iloc[i-1, j]
            column_condition = df_shaved.iloc[:, j] > factor_value * max_value
            condition = month_condition & column_condition
            
            df_shaved.loc[condition, df_shaved.columns[j]] = factor_value * max_value

    peak_economies = df - df_shaved
    
    return peak_economies, df_shaved

def calculate_load_shifting(max_values_df, min_factor=0.9):
    load_shifting_df = pd.DataFrame()
    
    # Iterate over factor values from 1 to 0.9 in 10 intervals
    for factor in np.linspace(1, min_factor, 20):
        # Calculate savings
        save_factor = 1 - factor
        max_value_savings = max_values_df * save_factor
        tarif = 6.19  # [CHF/kW] TOP A pic mensuel
        peak_cost_saving = max_value_savings * tarif
        annual_cost_saving = np.sum(peak_cost_saving, axis=0)
        
        # Append annual cost saving as a new row to the dataframe
        load_shifting_df = load_shifting_df.append(annual_cost_saving, ignore_index=True)
        
    return load_shifting_df

def calculate_financial_savings(peak_economies, HP_cost=8.44, HC_cost=2.6):
    """
    Calculate financial savings based on peak economies and energy costs.
    
    Parameters:
    peak_economies (pd.DataFrame): DataFrame containing peak economies.
    HP_cost (float): Cost for Monday to Friday in cents/kWh for Heures Pleines.
    HC_cost (float): Cost for Saturday and Sunday in cents/kWh for Heures Creuses.
    
    Returns:
    float: Annual financial savings.
    """
    # Aggregate the energy values to hourly by taking the sum
    hourly_energy = peak_economies.resample('H').sum()

    # Define a function to calculate the cost based on the time
    def calculate_cost(hour):
        if 6 <= hour < 22 and 0 <= pd.to_datetime(hour).weekday() <= 4:
            return HP_cost
        else:
            return HC_cost

    # Calculate the cost for each hour
    hourly_energy['Cost'] = hourly_energy.index.hour.map(calculate_cost)

    # Create a new dataframe for cost
    cost_df = pd.DataFrame(hourly_energy['Cost'], columns=['Cost'])

    # Remove the 'Cost' column from hourly_energy
    hourly_energy.drop(columns=['Cost'], inplace=True)

    # Multiply each row of hourly_energy by the corresponding cost
    financial_savings_df = hourly_energy.mul(cost_df['Cost'], axis=0) / 100  # Convert cost from cents to CHF

    # Calculate annual financial savings
    annual_financial_savings = np.sum(financial_savings_df, axis=0)

    return annual_financial_savings

def calculate_energy_economies(peak_economies):
    
    energy_economies_df = np.sum(peak_economies, axis=0)
    return energy_economies_df
#%% Statistical difference
Loads_buv = Typo_all_loads["Buvette"]
Loads_sport = Typo_all_loads["Sport"]
Loads_parking = Typo_all_loads["Parking"]
Loads_unique = pd.concat([Loads_buv, Loads_sport, Loads_parking], axis=1)
Loads = Typo_all_loads["Ecole"]
#Loads = Loads_unique
Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]

df = Loads_last_year.astype(np.longdouble)

#df = Loads.astype(np.longdouble)
average_monthly_diff_all_clients= []
for column in df:
    # Initialize a dictionary to store average monthly differences
    average_monthly_differences = {}
    
    # Loop through each month
    for desired_month in range(1, 13):
        # Ensure the datetime index is properly formatted
        df[column].index = pd.to_datetime(df[column].index, format='%d.%m.%Y %H:%M:%S')
    
        # Extract all instances of the desired month
        month_df = df[column][df[column].index.month == desired_month]
    
        # Extract weekdays
        weekdays_df = month_df[month_df.index.weekday < 5]
    
        # Discard the last value for January if necessary
        if desired_month == 1:
            weekdays_df = weekdays_df[:-1]
    
        # Convert to numpy array
        Daily_data = weekdays_df.to_numpy()
    
        # Replace zeros with NaN
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
    
        # Identify significant anomalies
        significant_anomalies_mask = sliced_3d_array > threshold
    
        # Initialize an array to store the differences
        differences = np.zeros_like(sliced_3d_array)
    
        # Compute the differences only for significant anomalies
        differences[significant_anomalies_mask] = sliced_3d_array[significant_anomalies_mask] - np.broadcast_to(percentile_95, sliced_3d_array.shape)[significant_anomalies_mask]
    
        # Flatten the differences to calculate the mean
        flattened_differences = differences[significant_anomalies_mask]
    
        # Remove NaN values
        valid_differences = flattened_differences[~np.isnan(flattened_differences)]
        
    
        # Calculate the average difference for the current month
        average_difference = np.mean(valid_differences) if len(valid_differences) > 0 else np.nan
        
        # Store the result in the dictionary
        average_monthly_differences[desired_month] = average_difference
    
    average_monthly_diff_all_clients.append(average_monthly_differences)
    # The result is stored in average_monthly_differences
#print(average_monthly_diff_all_clients)
avg_monthly_diff_df = pd.DataFrame(average_monthly_diff_all_clients).T


# finalization
max_values_df = calculate_max_values(df)
peak_economies, df_shaved = calculate_peak_economies(df, max_values_df, avg_monthly_diff_df)
stat_annual_savings = calculate_financial_savings(peak_economies)
stat_energy_savings = calculate_energy_economies(peak_economies)
