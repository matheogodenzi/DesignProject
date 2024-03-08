# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:56:39 2024

@author: matheo

"""

""" import libraries"""
import os 
import numpy as np
import pandas as pd 

def get_variable_name(var, namespace):
    """returns the name of the variable in a string format"""
    
    for name, obj in namespace.items():
        if obj is var:
            return name
    return None

 
