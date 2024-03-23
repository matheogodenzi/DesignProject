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

# check for anomaly by comparing total consumption over time period

def total_cons_ctrl(tested, df, period):
 
    # get the mean for given period of time
    df_tendency = f.period_tendencies(df, period)
    tested_tendency = f.period_tendencies(tested, period)
    
    # calculation of metrics
    df_tendency["Mean"]= df_tendency.apply(f.filter_and_calculate_mean, axis=1).copy()
    df_tendency["STD"] = df_tendency.apply(f.filter_and_calculate_std, axis=1).copy()
    
    # comparative vectors
    mean = df_tendency["Mean"].values
    std1 = df_tendency["Mean"].values+df_tendency["STD"].values
    std3 = df_tendency["Mean"].values+ 2 *df_tendency["STD"].values
    
    # test result vector: 0 if normal, 1 if >std1, 2 if std3
    total_cons_test = np.zeros_like(tested_tendency)
    
    # running statistical test
    for i in range(len(tested_tendency)):
   
        if tested_tendency.iloc[i,0] > (mean[i] + std1[i]):
            total_cons_test[i] = 1
        elif tested_tendency.iloc[i,0] > (mean[i] + std3[i]):
            total_cons_test[i] = 2
    
    return total_cons_test


#%%






# Example usage or tests
if __name__ == "__main__":
    
    print("testing functions !")

