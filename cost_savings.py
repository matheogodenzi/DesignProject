# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:02:20 2024

@author: mimag
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, False)

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
Typology = "Culture"
Period = "day"

# smoothing calculations
Loads = 1* Typo_all_loads[Typology] # conversion to [kW] if factor = 4, else kWh/quart d'heure

Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]

df = Loads_last_year.astype(np.longdouble)

dfkW = 4 * df



#%% Peak shaving savings


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

max_values_df = calculate_max_values(df)

def calculate_peak_economies(df, max_values_df, factor=0.8):
    """
    Calculate peak economies based on a factor and maximum values DataFrame.
    
    Parameters:
    df (pd.DataFrame): Original DataFrame.
    max_values_df (pd.DataFrame): DataFrame containing maximum values.
    factor (float): Factor for peak shaving.
    
    Returns:
    pd.DataFrame: DataFrame containing peak economies.
    """
    df_shaved = df.copy()

    for i in range(max_values_df.shape[0]):
        for j in range(df_shaved[df_shaved.index.month == i + 1].shape[1]):
            condition = (df_shaved.index.month == i + 1) & (df_shaved.iloc[:, j] > factor * max_values_df.iloc[i, j])
            df_shaved.loc[condition, df_shaved.columns[j]] = factor * max_values_df.iloc[i, j]

    peak_economies = df - df_shaved
    
    return peak_economies, df_shaved

# Usage:
peak_economies, df_shaved = calculate_peak_economies(df, max_values_df, factor =0.8)
energy_economies = peak_economies.mean(0)*365*24 #kWh/an/m2 (si normalisation préalable)
#plt.bar(Loads.columns, energy_economies)
#plt.xticks(rotation=45)
#plt.show()

#%% Computing cost savings related to the energy savings
# savings from reducing the monthly maximum load
# Initialize an empty dataframe to store the results

def calculate_load_shifting(max_values_df, min_factor=0.9):
    load_shifting_df = pd.DataFrame()
    
    # Iterate over factor values from 1 to 0.9 in 10 intervals
    for factor in np.linspace(1, min_factor, 20):
        # Calculate savings
        save_factor = 1 - factor
        max_value_savings = max_values_df * save_factor
        tarif = 12.39  # [CHF/kW] TOP B pic mensuel
        peak_cost_saving = max_value_savings * tarif
        annual_cost_saving = np.sum(peak_cost_saving, axis=0)
        
        # Append annual cost saving as a new row to the dataframe
        load_shifting_df = load_shifting_df.append(annual_cost_saving, ignore_index=True)
        
    return load_shifting_df



#%% calcul de réduction des coûts selon tarif au kWh
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

#%%
# parameters to change
Typology = "Commune"
Period = "day"

Loads_buv = Typo_all_loads["Buvette"]
Loads_sport = Typo_all_loads["Sport"]
Loads_parking = Typo_all_loads["Parking"]
Loads_unique = pd.concat([Loads_buv, Loads_sport, Loads_parking], axis=1)
Loads = Loads_unique
df = Loads_unique.astype(np.longdouble)

# smoothing calculations
#Loads = 1* Typo_all_loads[Typology] # conversion to [kW] if factor = 4, else kWh/quart d'heure

Loads_copy = Loads.copy()

Loads_copy.index = pd.to_datetime(Loads_copy.index, format='%d.%m.%Y %H:%M:%S')

# Get the end date of the last year
end_date_last_year = Loads_copy.index[-1] - pd.DateOffset(years=1)

# Slice the DataFrame to get data from only the last year
Loads_last_year = Loads_copy[end_date_last_year:]

df = Loads_last_year.astype(np.longdouble)

dfkW = 4 * df

### financial savings for varying factor of saving
load_shifting_df = calculate_load_shifting(calculate_max_values(dfkW), min_factor=0.8) # has to be in kW

annual_financial_savings = calculate_financial_savings(peak_economies)
financial_savings_list = []

for factor in np.linspace(1,0.8,20):
    peak_economies, df_shaved = calculate_peak_economies(df, max_values_df, factor)
    financial_savings_list.append(calculate_financial_savings(peak_economies))


financial_savings_df = pd.concat(financial_savings_list, axis=1).T
my_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']
load_shifting_df.columns = financial_savings_df.columns

total_financial_savings_df = financial_savings_df.add(load_shifting_df)
"""
# PUISSANCE
plt.figure(figsize=(6, 5), dpi=300)
for i, column in enumerate(financial_savings_df.columns): # adapt to which type of cost to plot
    plt.plot(np.linspace(0, 20, 20),load_shifting_df[column], color=my_colors[i])
#plt.yscale("log")
plt.title('Economies financières par écrêtement des pointes (Puissance)')
plt.xlabel('Facteur de réduction des maxima mensuels [%]')
plt.ylabel('Economies financières (CHF/année)')
plt.locator_params(axis='y', nbins=10)
#plt.xticks(rotation=45)
plt.legend(Loads.columns)
plt.grid(axis='y')
plt.show()

# ENERGIE
plt.figure(figsize=(6, 5), dpi=300)
for i, column in enumerate(financial_savings_df.columns): # adapt to which type of cost to plot
    plt.plot(np.linspace(0, 20, 20),financial_savings_df[column], color=my_colors[i])
#plt.yscale("log")
plt.title('Economies financières par écrêtement des pointes (Energie)')
plt.xlabel('Facteur de réduction des maxima mensuels [%]')
plt.ylabel('Economies financières (CHF/année)')
plt.locator_params(axis='y', nbins=10)
#plt.xticks(rotation=45)
plt.legend(Loads.columns)
plt.grid(axis='y')
plt.show()
"""

# TOTAL
plt.figure(figsize=(6, 5), dpi=300)
for i, column in enumerate(financial_savings_df.columns): # adapt to which type of cost to plot
    plt.plot(np.linspace(0, 20, 20),total_financial_savings_df[column], color=my_colors[i])
#plt.yscale("log")
plt.title('Economies financières par écrêtement des pointes (Total)')
plt.xlabel('Facteur de réduction des maxima mensuels [%]')
plt.ylabel('Economies financières (CHF/année)')
plt.locator_params(axis='y', nbins=10)
#plt.xticks(rotation=45)
plt.legend(Loads.columns)
plt.grid(axis='y')
plt.show()


###peak shaving with load reduction

### ATTENTION garder Loads en kWh/quart d'heure
def calculate_energy_economies(df, max_values_df):
# Initialize an empty list to store the energy economies for each factor
    energy_economies_list = []
    
    # Iterate over the factors from 0.9 to 0.4 with 10 intervals
    for factor in np.linspace(1, 0.8, 20):
        # Apply peak shaving with the current factor
        
        peak_economies, df_shaved = calculate_peak_economies(df, max_values_df, factor)
        # Calculate energy economies and append to the list
        energy_economies = peak_economies.sum() # kWh/year/m2 (assuming prior normalization)
        energy_economies_list.append(energy_economies)
        
    energy_economies_df = pd.DataFrame(energy_economies_list)
    return energy_economies_df


energy_economies_df = calculate_energy_economies(df, calculate_max_values(df))

my_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']

# Plot the energy economies for each factor
plt.figure(figsize=(6, 5), dpi=300)
for i, column in enumerate(energy_economies_df.columns):
    plt.plot(np.linspace(0, 20, 20),energy_economies_df[column], color=my_colors[i])
#plt.yscale("log")
plt.title('Economies énergétiques par écrêtement des pointes')
plt.xlabel('Facteur de réduction des maxima mensuels [%]')
plt.ylabel('Economie électrique ($kWh_{el}$/année)') # verifier si normalisation activée
#plt.xticks(rotation=45)
plt.legend(Loads.columns)
plt.grid(axis='y')
plt.show()

#%% illustrative example of peak shaving 
palette = sb.color_palette("hls", 13)

peak_economies, df_shaved = calculate_peak_economies(dfkW, calculate_max_values(dfkW), factor=0.8)
slct_cons = 'S280'

plt.figure(dpi=300)
plt.plot(dfkW[slct_cons], color=palette[0])
plt.plot(df_shaved[slct_cons], color='royalblue')
plt.legend(["Ecrêtement 20%","Charge restante"], loc='center left', bbox_to_anchor=(1, 0.84))
# Format x-axis ticks to display only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%W'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.tick_params(axis='both', which='major', labelsize=9)
plt.xlabel("Semaine")
plt.ylabel("Charge - [$kW_{el}/m^2$]")
#plt.title("Illustration de l'écrêtement des pointes")
plt.show()

