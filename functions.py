# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:56:39 2024

@author: matheo

"""

""" import libraries"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import sklearn as skl
import pandas as pd
import os
import seaborn as sb
from datetime import datetime
import matplotlib.dates as mdates


def get_variable_name(var, namespace):
    """
    

    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    namespace : TYPE
        DESCRIPTION.

    Returns
    -------
    name : TYPE
        DESCRIPTION.

    """
    """returns the name of the variable in a string format"""
    
    for name, obj in namespace.items():
        if obj is var:
            return name
    return None


def period_tendencies(df, period="week"):
    """
    Delineate annual tendencies over days, weeks, and months

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    period : TYPE, optional
        DESCRIPTION. The default is "week".

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    
    if period == "day":
        chunk_size = 96 #96 quarters of hour in a day
    elif period == "week":
        chunk_size = 7*96 #96 quarters of hour in a day
    elif period == "month":
        chunk_size = 30*96 #TO BE MODIFIED TO HAVE THE MONTHS IN DETAIL 
    
    num_rows = df.shape[0]
    averages = []
    
    df = df.astype(np.longdouble)

    # Iterate over the DataFrame in chunks of 96 rows
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
        chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
        averages.append(chunk_avg)  # Append the averages to the list
        print(i)
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    # Print the result
    #print(result)
    
    return result

def period_tendencies_new(df, period="week"):
    """
    Delineate annual tendencies over days, weeks, and months

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    period : TYPE, optional
        DESCRIPTION. The default is "week".

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    
    df.index = pd.to_datetime(df.index)
    results_df = pd.DataFrame(columns=df.columns)
    
    if period == "day":
        for (year, month, day), group in df.groupby([df.index.year, df.index.month, df.index.day]):
            # Calculate the average of the values of each column within each group
            
            average_of_smallest_values = group.apply(lambda x: x.mean())
            #print(group.shape)
            # Create a DataFrame for the current iteration
            current_df = pd.DataFrame(average_of_smallest_values).T  # Transpose to make it a row
            #current_df['Day'] = day
            #current_df['Month'] = month
            
            # Set datetime index for the current iteration
            current_df.index = pd.to_datetime([f'{year}-{month}-{day}'])
            
            # Append the DataFrame to the results DataFrame
            results_df = pd.concat([results_df, current_df])
    
    if period == "week":
        # Group the DataFrame by weeks and include month and year information in the index
        groups = df.groupby([pd.Grouper(freq='W'), df.index.year, df.index.month])
        
        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=df.columns)
        
        # Iterate over the groups
        for (week_start, year, month), group in groups:
            # Calculate the average of the values of each column within each group
            average_of_smallest_values = group.apply(lambda x: x.mean())
            
            # Create a DataFrame for the current iteration
            current_df = pd.DataFrame(average_of_smallest_values).T  # Transpose to make it a row
            
            # Set datetime index for the current iteration (include week start date, year, and month)
            current_df.index = pd.MultiIndex.from_tuples([(week_start, year, month)], names=['Week_Start', 'Year', 'Month'])
            
            # Append the DataFrame to the results DataFrame
            results_df = pd.concat([results_df, current_df])
            
    if period == "month":
        
        # Group the DataFrame by month and include year information in the index
        groups = df.groupby([df.index.year, df.index.month])
        
        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=df.columns)
        
        # Iterate over the groups
        for (year, month), group in groups:
            # Calculate the average of the values of each column within each group
            average_values = group.mean()
            
            # Create a DataFrame for the current iteration
            current_df = pd.DataFrame(average_values).T  # Transpose to make it a row
            
            # Set datetime index for the current iteration (include year and month)
            current_df.index = pd.MultiIndex.from_tuples([(year, month)], names=['Year', 'Month'])
            
            # Append the DataFrame to the results DataFrame
            results_df = pd.concat([results_df, current_df])

    
    return results_df

def plot_tendency(tendency,specific_load=None, title="Electric consumptions ***insert Typology***", period="week", show_legend=False):
    """
    

    Parameters
    ----------
    tendency : TYPE
        DESCRIPTION.
    specific_load : TYPE, optional
        DESCRIPTION. The default is None.
    title : TYPE, optional
        DESCRIPTION. The default is "Electric consumptions ***insert Typology***".
    period : TYPE, optional
        DESCRIPTION. The default is "week".
    show_legend : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    plt.plot(tendency, linewidth=1)
    plt.title(title)
    plt.xlabel(period)
    plt.ylabel('kWh_{el}/m2')
    #plt.legend().set_visible(False)

    if show_legend == True:
        # Place legend outside the plot area attributing numbers to the plotlines
        # plt.legend([i for i in range(tendency.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')
        # Show the plot
        plt.legend(tendency.columns,loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title="Adresses")
        # Restraining legend box to a certain width
        #plt.legend.get_frame().set_linewidth(0.5)  # Adjust the border width of the legend box
        #plt.legend.get_frame().set_width(0.3)      # Adjust the width of the legend box

        #plt.tight_layout()  # Adjust layout to prevent clipping of legend
    
    plt.grid()
    plt.show()
    
    return 

def filter_and_calculate_mean(row):
    mean = row.mean()
    std_dev = row.std()
    threshold = 2  # Number of standard deviations beyond which a value is considered an outlier
    outliers_mask = (row - mean).abs() > threshold * std_dev
    filtered_row = row[~outliers_mask]
    return filtered_row.mean()

def filter_and_calculate_std(row):
    mean = row.mean()
    std_dev = row.std()
    threshold = 2 # Number of standard deviations beyond which a value is considered an outlier
    outliers_mask = (row - mean).abs() > threshold * std_dev
    filtered_row = row[~outliers_mask]
    return filtered_row.std()


def plot_mean_load(Load, Tendency, period="Specify period", Typology="Specify Typologie", xaxis="specify label"):
    """
    

    Parameters
    ----------
    Tendency : TYPE
        DESCRIPTION.
    period : TYPE, optional
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
    if period == "day":
        num_ticks = 12
        # Replace existing indices with hours and minutes
        #Tendency.index = Tendency.index.strftime('%H:%M')
       # datetime_list =  Tendency.index.tolist()
        #time_list = [dt.strftime('%H:%M') for dt in datetime_list]
        
    elif period == "week":
        num_ticks = 7
       # datetime_list = Tendency.index.tolist()
       # time_list = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in datetime_list]
    
    elif period=="month":
        num_ticks = 12
        
    elif period=="year":
        num_ticks = 12
    
    
    # calculation of metrics
    Tendency["Mean"]= Tendency.apply(filter_and_calculate_mean, axis=1).copy()
    Tendency["STD"] = Tendency.apply(filter_and_calculate_std, axis=1).copy()
    
    #plotlines 
    mean = Tendency["Mean"].values
    std1 = Tendency["Mean"].values+Tendency["STD"].values
    std3 = Tendency["Mean"].values+ 3 *Tendency["STD"].values
    

    #
    fig, ax = plt.subplots()
    
    # plotting stats
    ax.plot(x, std3, color="b", alpha=0.2, linestyle=':', linewidth=1, label="anomaly threshold")
    ax.plot(x, std1, color="mediumblue", alpha=0.2, linestyle='--', linewidth=1, label="surveillance treshold")
    ax.plot(x, mean, color="darkblue", alpha=0.4, linewidth=1, label="Typology mean")
    if Load is not None:
        ax.plot(x, Load, color="black", alpha=1, linestyle='solid', linewidth=2, label= Load.columns[0])
    #plt.plot(Tendency["Mean"].values-Tendency["STD"].values, color="blue", alpha=0.3)
    
    
    #fillings 
    # Shade the area between the lines
    plt.fill_between(x, mean, std1, color='darkblue', alpha=0.4)
    plt.fill_between(x, std1, std3, color='mediumblue', alpha=0.3)
    
    """
    if period == "day":
        plt.gca().xaxis.set_major_formatter(lambda x, pos: pd.to_datetime(x).strftime('%H:%M'))
        
    elif period == "week":
       plt.gca().xaxis.set_major_formatter(lambda x, pos: pd.to_datetime(x).strftime('%A'))

    elif period=="month":
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
        
    elif period=="year":
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    """
    
    
    interval = len(x) // num_ticks
    locator = mdates.DayLocator(interval=interval)
    plt.gca().xaxis.set_major_locator(locator)
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)
    
    #pimping the plot 
    if period == "day":
        plt.title("Mean load profile for " + Typology + " on a daily basis")  
    else :
        plt.title("Mean load profile for " + Typology + " on a " +period+"ly basis")
        # Customize the x-axis tick labels
        #ax.xaxis.set_major_locator(mdates.DayLocator(interval))  # Set tick locator to daily intervals
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # Format tick labels to display weekday 
   
   
   
    plt.xlabel(xaxis)
    plt.ylabel(r"$kWh_{el}/m^2$")
    plt.legend(fontsize=8)
    plt.grid()
    plt.show()
    
    return Tendency


def typical_period(df, period):
    
    day_nbr = int(df.shape[0]/4/24)
    week_days = 7
    month_days = 30
    year_days = 365
    week_nbr = day_nbr // week_days
    month_nbr = day_nbr // month_days
    year_nbr = day_nbr // year_days
    
    if period == "day":
        intervals = day_nbr
        period_length = 96  # Number of data points in a day
    elif period == "week":
        intervals = week_nbr
        period_length = 96 * week_days  # Number of data points in a week
    elif period == "month":
        intervals = month_nbr
        period_length = 96 * month_days  # Number of data points in a month
    elif period == "year":
        intervals = year_nbr
        period_length = 96 * year_days  # Number of data points in a year
    
    # Initialize period_df with zeros to accumulate data
    period_df = df.iloc[:period_length, :].copy()
    
    for i in range(1,intervals):
        try:
            period_df += df.iloc[i * period_length : (i + 1) * period_length, :].values
            #print(period_df)
        except ValueError:
            print("ValueError occurred during iteration. Skipping this interval.")
    period_df /= intervals
    
    return period_df





def plot_typical_day(data_day, typology):
    indices_list = data_day.index.tolist()
    
    datetime_list = [datetime.strptime(index, '%Y-%m-%d %H:%M:%S') for index in indices_list]
    time_list = [dt.time().strftime("%H:%M") for dt in datetime_list]
    
    plt.plot(time_list, data_day, linewidth=0.5)
    plt.legend([i for i in range(data_day.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')
    # Set the x-axis ticks to display only every n-th value
    n = 7 # Display every 10th value
    plt.xticks(np.arange(0, data_day.shape[0], n), fontsize=7, rotation=45)
    plt.xlabel("hours of the day", fontsize=12)
    plt.ylabel("electric consumption" + r" $kWh_{el}/{m^2} $ ", fontsize=12)
    plt.title("Typical Day - " + typology, fontsize=15, fontdict={'fontweight': 'bold'})
    plt.grid()
    
    plt.show()
    
    return





def plot_typical_week(data_week, typology):
    indices_list = data_week.index.tolist()
    
    datetime_list = [datetime.strptime(index, '%d.%m.%Y %H:%M:%S') for index in indices_list]
    #time_list = [dt.strftime('%d.%m.%Y %H:%M:%S') for dt in datetime_list]
    #print(time_list)
    
    #plotting
    fig, ax = plt.subplots()
    ax.plot(datetime_list, data_week, linewidth=0.5)
    
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





# Example usage or tests
if __name__ == "__main__":
    
    #what if 
    print("hello")