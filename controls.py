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






# Example usage or tests
if __name__ == "__main__":
    
    print("testing functions !")

