# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:53:58 2024

@author: matheo
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import sklearn as skl
import pandas as pd
import os


"""functions imports"""
import functions as f

"""data acquisition"""


# DEFINING PATHS
## Generic path of the folder in your local terminal 
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)

## Creating specificpath for each commune
renens = parent_directory + "\Renens"
ecublens = parent_directory + "\Ecublens"
crissier = parent_directory + "\Crissier"
chavannes = parent_directory + "\Chavannes"

Commune_paths = [renens, ecublens, crissier, chavannes]

## reading excel files 
load_data = []
building_data = []
for i, commune in enumerate(Commune_paths):
    
    # extracting load curves 
    df = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2)
    df.set_index("Date", inplace=True)
    
    # extracting buildings
    buildings = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=0)
    
    # storing data 
    load_data.append(df)
    building_data.append(buildings)


LoadCurve_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data[i] for i in range(len(Commune_paths))}
Building_dict = {f.get_variable_name(Commune_paths[i], globals()): building_data[i] for i in range(len(Commune_paths))}

#%%

## Extracting Typologies 

print(LoadCurve_dict["crissier"])


#%%

