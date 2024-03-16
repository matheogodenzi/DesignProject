# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:56:39 2024

@author: matheo

"""

""" import libraries"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def get_variable_name(var, namespace):
    """returns the name of the variable in a string format"""
    
    for name, obj in namespace.items():
        if obj is var:
            return name
    return None



def average_24h(df):
    
    chunk_size = 96 #96 quarters of hour
    num_rows = len(df)
    averages = []

    # Iterate over the DataFrame in chunks of 96 rows
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
        chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
        averages.append(chunk_avg)  # Append the averages to the list

    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T

    # Print the result
    print(result)
    return result



def av_1_week(df):
    
    chunk_size = 7*24*4 #96 quarters of hour
    num_rows = len(df)
    averages = []
    
    # Iterate over the DataFrame in chunks of 96 rows
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
        chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
        averages.append(chunk_avg)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    # Print the result
    print(result)
    
    return result


def av_1_month(df):
    
    chunk_size = 28*24*4 #96 quarters of hour
    num_rows = len(df)
    averages = []
    
    # Iterate over the DataFrame in chunks of 96 rows
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
        chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
        averages.append(chunk_avg)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    # Print the result
    print(result)
    
    return result

def period_tendencies(df, period="week"):
    
    if period == "day":
        chunk_size = 96 #96 quarters of hour in a day
    elif period == "week":
        chunk_size = 7*96 #96 quarters of hour in a day
    elif period == "month":
        chunk_size = 30*96 #TO BE MODIFIED TO HAVE THE MONTHS IN DETAIL 
    
    num_rows = len(df)
    averages = []
    
    # Iterate over the DataFrame in chunks of 96 rows
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
        chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
        averages.append(chunk_avg)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    # Print the result
    print(result)
    
    return result

def plot_tendency(tendency,specific_load=None, title="Electric consumptions ***insert Typology***", period="week", show_legend=False):
    """
    
    """
    plt.plot(tendency, linewidth=1)
    plt.title(title)
    plt.xlabel(period)
    plt.ylabel('kWh_{el}')
    #plt.legend().set_visible(False)

    if show_legend == True:
        # Place legend outside the plot area attributing numbers to the plotlines
        plt.legend([i for i in range(tendency.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')
        # Show the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of legend
    
    plt.grid()
    plt.show()
    
    return 




def plot_mean_load(Tendency, period="Specify period", Typology="Specify Typologie"):
    row_mean = Tendency.mean(axis=1)
    row_std = Tendency.std(axis=1)
    Tendency["Mean"] = row_mean
    Tendency["STD"] = row_std

    plt.figure()
    
    # plotting stats
    plt.plot(Tendency["Mean"].values+Tendency["STD"].values, color="blue", alpha=0.3)
    plt.plot(Tendency["Mean"].values, color="blue")
    plt.plot(Tendency["Mean"].values-Tendency["STD"].values, color="blue", alpha=0.3)
    
    #pimping the plot 
    if period == "day":
      plt.title("Mean load profile for " + Typology + " on a daily basis")  
    else :
        plt.title("Mean load profile for " + Typology + " on a " +period+"ly basis")
   
    plt.xlabel(period)
    plt.ylabel("kWh_el")
    plt.legend(["mean + std", "mean", "mean-std"])
    plt.grid()
    plt.show()
    
    return Tendency


def typical_period(df, period):
    
    day_nbr = 365
    week_days = 7
    month_days = 30
    week_nbr = day_nbr//week_days
    month_nbr = day_nbr//month_days
    
    if period == "day":
        intervals = day_nbr
        
        for i in range(intervals):
            if i == 0: 
                period_df = df.iloc[:96, :]
            else: 
                period_df += df.iloc[(i-1)*96:i*96, :].values
        
    
    
    if period == "week":
        intervals = week_nbr
        
        for i in range(intervals):
            if i == 0: 
                period_df = df.iloc[:96*week_days, :]
            else: 
                period_df += df.iloc[(i-1)*96*week_days:i*96*week_days, :].values
        
        
    if period == "month":
        intervals = month_nbr
    
        for i in range(intervals):
            if i == 0: 
                period_df = df.iloc[:96*month_days, :]
            else:
                period_df += df.iloc[(i-1)*96*month_days:i*96*month_days, :].values
    
    period_df /= intervals
    
    return period_df