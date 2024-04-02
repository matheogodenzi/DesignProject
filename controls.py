# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:19:09 2024

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

"""functions imports"""

import functions as f


#%%

# check for anomaly by comparing mean consumption over the whole dataset for a given granularity

def total_cons_ctrl(testLoad, df, granulo):
 
    # get the mean for given period of time
    df_tendency = f.period_tendencies(df, granulo)
    tested_tendency = f.period_tendencies(testLoad, granulo)
    
    # calculation of metrics
    df_tendency["Mean"]= df_tendency.apply(f.filter_and_calculate_mean, axis=1).copy()
    df_tendency["STD"] = df_tendency.apply(f.filter_and_calculate_std, axis=1).copy()
    
    # comparative vectors
    mean = df_tendency["Mean"].values
    std1 = df_tendency["Mean"].values+df_tendency["STD"].values
    std2 = df_tendency["Mean"].values+ 2 *df_tendency["STD"].values
    
    # test result vector: 0 if normal, 1 if >std1, 2 if >std2
    total_cons_test = np.zeros_like(tested_tendency)
    
    # running statistical test
    for i in range(len(tested_tendency)):
        if tested_tendency.iloc[i,0] > std1[i]:
            total_cons_test[i] = 1
        elif tested_tendency.iloc[i,0] > std2[i]:
            total_cons_test[i] = 2
    
    return total_cons_test


#%%

def typical_period_control(testLoad, df):
      
    # computing metrics
    df["Mean"] = df.apply(f.filter_and_calculate_mean, axis=1).copy()
    mean = df["Mean"].values
    
    # test result vector: 0 if normal, 1 if > mean, 2 if > 2means
    typical_period_test = np.zeros_like(testLoad)
     # running test
    for i in range(len(testLoad)):
        if testLoad.iloc[i,0] > mean[i]:
            typical_period_test[i] = 1
        elif testLoad.iloc[i,0] > (2 * mean[i]):
            typical_period_test[i] = 2
    
    print(f"Value: {testLoad.iloc[i,0]}, Mean: {mean[i]}, Result: {typical_period_test[i]}")
    return typical_period_test

#%%
# implementation of control to general plot
def plot_mean_load_control(Load, Tendency, granulo="Specify granulotmetry", Typology="Specify Typologie", xaxis="specify label"):
    """
    

    Parameters
    ----------
    Tendency : TYPE
        DESCRIPTION.
    granulo : TYPE, optional
        DESCRIPTION. The default is "Specify period".
    Typology : TYPE, optional
        DESCRIPTION. The default is "Specify Typologie".

    Returns
    -------
    Tendency : TYPE
        DESCRIPTION.

    """
    #row_mean = Tendency.mean(axis=1)
    #row_std = Tendency.std(axis=1)
    #Tendency["Mean"] = row_mean
    #Tendency["STD"] = row_std
    
    # x axis label 
    x = Tendency.index

    
    # Calculate the interval for the DayLocator
    if granulo == "day":
        num_ticks = 12
        # Replace existing indices with hours and minutes
        #Tendency.index = Tendency.index.strftime('%H:%M')
       # datetime_list =  Tendency.index.tolist()
        #time_list = [dt.strftime('%H:%M') for dt in datetime_list]
        
    elif granulo == "week":
        num_ticks = 7
       # datetime_list = Tendency.index.tolist()
       # time_list = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in datetime_list]
    
    elif granulo=="month":
        num_ticks = 12

    
    # TODO : Add months 
    
    
    # calculation of metrics
    Tendency["Mean"]= Tendency.apply(f.filter_and_calculate_mean, axis=1).copy()
    Tendency["STD"] = Tendency.apply(f.filter_and_calculate_std, axis=1).copy()
    
    #plotlines 
    mean = Tendency["Mean"].values
    std1 = Tendency["Mean"].values+Tendency["STD"].values
    std2 = Tendency["Mean"].values+ 2 *Tendency["STD"].values
    
    # control implementation
    total_cons_test = np.zeros_like(Load)
    
    # running statistical test
    for i in range(len(Load)):
        if Load.iloc[i,0] > std1[i]:
            total_cons_test[i] = 1
        elif Load.iloc[i,0] > std2[i]:
            total_cons_test[i] = 2    

    # converting Load_control to curve
    Load_controlstd1 = Load.copy()
   
    for i in range(len(Load_controlstd1)):
        if total_cons_test[i] == 0:
            Load_controlstd1.iloc[i,0] = np.nan

    
    fig, ax = plt.subplots()
    # plotting stats
    ax.plot(x, std2, color="b", alpha=0.2, linestyle=':', linewidth=1, label="anomaly threshold")
    ax.plot(x, std1, color="mediumblue", alpha=0.2, linestyle='--', linewidth=1, label="surveillance treshold")
    ax.plot(x, mean, color="darkblue", alpha=0.4, linewidth=1, label="Typology mean")
    if Load is not None:
        ax.plot(x, Load, color="black", alpha=1, linestyle='solid', linewidth=2, label= Load.columns[0])
        ax.plot(x, Load_controlstd1, color="red", alpha=1, linestyle='solid', linewidth=2, label="surveillance")
    #plt.plot(Tendency["Mean"].values-Tendency["STD"].values, color="blue", alpha=0.3)
    
    
    #fillings 
    # Shade the area between the lines
    plt.fill_between(x, mean, std1, color='darkblue', alpha=0.4)
    plt.fill_between(x, std1, std2, color='mediumblue', alpha=0.3)
    
        
    interval = len(x) // num_ticks
    locator = mdates.DayLocator(interval=interval)
    plt.gca().xaxis.set_major_locator(locator)
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)
    
    #pimping the plot 
    if granulo == "day":
        plt.title("Mean load profile for " + Typology + " on a daily basis")  
    else :
        plt.title("Mean load profile for " + Typology + " on a " +granulo+"ly basis")
        # Customize the x-axis tick labels
        #ax.xaxis.set_major_locator(mdates.DayLocator(interval))  # Set tick locator to daily intervals
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # Format tick labels to display weekday 
   
   
   
    plt.xlabel(xaxis)
    plt.ylabel(r"$kWh_{el}/m^2$")
    plt.legend(fontsize=8)
    plt.grid()
    plt.show()
    
    return Tendency

#%%
# implementation of control to general plot
def plot_typical_week_control(testLoad, data_week, typology):
    # control on data
    # computing metrics
    data_week["Mean"] = data_week.apply(f.filter_and_calculate_mean, axis=1).copy()
    mean = data_week["Mean"].values
    
    # test result vector: 0 if normal, 1 if > mean, 2 if > 2means
    typical_period_test = np.zeros_like(testLoad)
    
    # running tests
    for i in range(len(testLoad)):
        if testLoad.iloc[i,0] > mean[i]:
            typical_period_test[i] = 1
        elif testLoad.iloc[i,0] > (2 * mean[i]):
            typical_period_test[i] = 2
    
    
    # converting typical_period_test to curves
    control_curve1 = testLoad.copy()
    control_curve2 = testLoad.copy()
    
    for i in range(len(typical_period_test)):
        if typical_period_test[i] != 1:
            control_curve1.iloc[i,0] = np.nan
        if typical_period_test[i] != 2:
            control_curve2.iloc[i,0] = np.nan
    
    # managing datetime
    indices_list = data_week.index.tolist()
    
    datetime_list = [datetime.strptime(index, '%d.%m.%Y %H:%M:%S') for index in indices_list]
    #time_list = [dt.strftime('%d.%m.%Y %H:%M:%S') for dt in datetime_list]
    #print(time_list)
    
    #plotting
    fig, ax = plt.subplots()
    ax.plot(datetime_list, data_week, linewidth=0.5)
    ax.plot(datetime_list, testLoad, color="black", alpha=1, linestyle='solid', linewidth=2, label="test")
    ax.plot(datetime_list, control_curve1, color="orange", alpha=1, linestyle='solid', linewidth=2, label="surveillance")
    ax.plot(datetime_list, control_curve2, color="red", alpha=1, linestyle='solid', linewidth=2, label="anomaly")
    # Customize the x-axis tick labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Set tick locator to daily intervals
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # Format tick labels to display weekday 
    
    # axis
    plt.xticks(rotation=45)
    plt.xlabel("Week days", fontsize=12)
    plt.ylabel("electric consumption" + r" $kWh_{el}/{m^2} $", fontsize=12)
    ax.grid()
    
    # legend 
    plt.legend([i for i in range(data_week.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    #title 
    plt.title("Annual typical week - " + typology, fontsize=15, fontdict={'fontweight': 'bold'})
    
    plt.show()
    
    return

#%%

# implementation of control to general plot
def plot_typical_week_control_clean(testLoad, data_week, typology):
    # control on data
    # computing metrics
    data_week["Mean"] = data_week.apply(f.filter_and_calculate_mean, axis=1).copy()
    mean = data_week["Mean"].values
    
    # test result vector: 0 if normal, 1 if > mean, 2 if > 2means
    typical_period_test = np.zeros_like(testLoad)
    
    # running test
    for i in range(len(testLoad)):
        if testLoad.iloc[i,0] > mean[i]:
            typical_period_test[i] = 1
        elif testLoad.iloc[i,0] > (2 * mean[i]):
            typical_period_test[i] = 2
    
    
    # converting typical_period_test to curve
    control_curve1 = testLoad.copy()
    control_curve2 = testLoad.copy()
    
    for i in range(len(typical_period_test)):
        if typical_period_test[i] != 1:
            control_curve1.iloc[i,0] = np.nan
        if typical_period_test[i] != 2 or typical_period_test[i] == 1:
            control_curve2.iloc[i,0] = np.nan
    
    # managing datetime
    indices_list = data_week.index.tolist()
    
    datetime_list = [datetime.strptime(index, '%d.%m.%Y %H:%M:%S') for index in indices_list]
    #time_list = [dt.strftime('%d.%m.%Y %H:%M:%S') for dt in datetime_list]
    #print(time_list)
    
    #plotting
    fig, ax = plt.subplots()
    ax.plot(datetime_list, data_week["Mean"], linewidth=0.5)
    ax.plot(datetime_list, testLoad, color="black", alpha=1, linestyle='solid', linewidth=2, label="test")
    ax.plot(datetime_list, control_curve1, color="orange", alpha=1, linestyle='solid', linewidth=2, label="surveillance")
    ax.plot(datetime_list, control_curve2, color="red", alpha=1, linestyle='solid', linewidth=2, label="anomaly")
   
    # Customize the x-axis tick labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Set tick locator to daily intervals
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # Format tick labels to display weekday 
    
    # axis
    plt.xticks(rotation=45)
    plt.xlabel("Week days", fontsize=12)
    plt.ylabel("electric consumption" + r" $kWh_{el}/{m^2} $", fontsize=12)
    ax.grid()
    
    # legend 
    plt.legend([i for i in range(data_week.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    #title 
    plt.title("Annual typical week - " + typology, fontsize=15, fontdict={'fontweight': 'bold'})
    
    plt.show()
    
    return


#%%

def yearly_consumption(Load, df):
    # computing metrics
    df_yearly_sum = df.sum().to_frame().T
    df_yearly_sum["Mean"]= df_yearly_sum.apply(f.filter_and_calculate_mean, axis=1).copy()
    df_yearly_sum["STD"] = df_yearly_sum.apply(f.filter_and_calculate_std, axis=1).copy()
    
    # comparing the Load
    load_yearly_sum = Load.sum().to_frame().T
    load_yearly_sum["ratio_to_mean"] = load_yearly_sum.iloc[0,0]/df_yearly_sum["Mean"]
    load_yearly_sum["ratio_to_std"] = (load_yearly_sum.iloc[0,0]- df_yearly_sum["Mean"])/df_yearly_sum["STD"]
    
    return load_yearly_sum


#%%
def quarterly_consumption(Load, df):
    # List to store quarterly results
    quarterly_results = []

    # Number of entries per quarter (assuming each month has 30 days)
    days_per_quarter = 90
    entries_per_day = 24 * 4  # Assuming 15-minute intervals

    # Calculate number of entries per quarter
    total_entries_per_quarter = days_per_quarter * entries_per_day

    # Determine the total number of days in the dataset
    total_days = len(df) // entries_per_day

    # Iterate over the DataFrame in chunks of 90 days (or the nearest approximation)
    for i in range(0, total_days, days_per_quarter):
        # Calculate the end index for the current quarter
        end_index = min((i + days_per_quarter) * entries_per_day, len(df))

        # Check if the end index exceeds the length of the DataFrame
        if end_index >= len(df):
            break  # Break out of the loop if we've reached the end of the data

        # Slice the DataFrame for the current quarter
        quarterly_df = df.iloc[i * entries_per_day:end_index]

        # Computing metrics for the quarterly DataFrame
        quarterly_sum = quarterly_df.sum().to_frame().T
        quarterly_sum["Mean"] = quarterly_sum.apply(f.filter_and_calculate_mean, axis=1).copy()
        quarterly_sum["STD"] = quarterly_sum.apply(f.filter_and_calculate_std, axis=1).copy()

        # Comparing the Load for the quarterly DataFrame
        quarterly_load_sum = Load.iloc[i * entries_per_day:end_index].sum().to_frame().T
        quarterly_load_sum["ratio_to_mean"] = quarterly_load_sum.iloc[0, 0] / quarterly_sum["Mean"]
        quarterly_load_sum["ratio_to_std"] = (quarterly_load_sum.iloc[0, 0] - quarterly_sum["Mean"]) / quarterly_sum["STD"]

        # Append the results for this quarter to the list
        quarterly_results.append(quarterly_load_sum)

    # Concatenate the results for all quarters into a single DataFrame
    quarterly_consumption_result = pd.concat(quarterly_results)

    return quarterly_consumption_result


#%%
# Example usage or tests
if __name__ == "__main__":
    
    print("testing functions !")

