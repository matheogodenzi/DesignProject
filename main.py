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
Commune_data = []
for i, commune in enumerate(Commune_paths):
    
    df = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2)
    df.set_index("Date", inplace=True)
    Commune_data.append(df)


Commune_dict = {f.get_variable_name(Commune_paths[i], globals()): Commune_data[i] for i in range(len(Commune_paths))}

#%%



## Extracting Typologies 

print(Commune_dict["crissier"])

