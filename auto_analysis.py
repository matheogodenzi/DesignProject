# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:21:41 2024

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
import controls as c


"""functions"""

def extract_period(df, start_date, end_date):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    start_date : TYPE
        DESCRIPTION.
    end_date : TYPE
        DESCRIPTION.

    Returns
    -------
    period : TYPE
        DESCRIPTION.

    """
    
    df.index = pd.to_datetime(df.index)
    period = df.loc[start_date:end_date]
    print(f'period : \n {period}')
    
    return period


def extract_time(df, datetime):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    datetime : TYPE
        DESCRIPTION.

    Returns
    -------
    midnight_instances : TYPE
        DESCRIPTION.

    """
    df.index = pd.to_datetime(df.index)
    # Extract instances at midnight
    selected_time_instances = df[df.index.time == datetime.time()]
    print(f'time : \n {selected_time_instances}')
    
    return selected_time_instances



