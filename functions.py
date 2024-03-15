# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:56:39 2024

@author: matheo

"""

""" import libraries"""

import numpy as np
import pandas as pd 

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






















 
 
