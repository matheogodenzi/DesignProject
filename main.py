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
import seaborn as sb


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
load_data_2023 = []
load_data_2022 = []
building_data_2023 = []
for i, commune in enumerate(Commune_paths):
    
    # extracting load curves 
    load_2023 = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2)
    load_2023.set_index("Date", inplace=True)
    load_2022 = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_cch_podvert_2022.xlsx", sheet_name=2)
    load_2022.set_index("Date", inplace=True)
    
    
    # extracting buildings
    buildings = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=0)
    
    # storing data 
    load_data_2023.append(load_2023)
    load_data_2022.append(load_2022)
    
    building_data_2023.append(buildings)


LoadCurve_2023_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2023[i] for i in range(len(Commune_paths))}
LoadCurve_2022_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2022[i] for i in range(len(Commune_paths))}
Building_dict_2023 = {f.get_variable_name(Commune_paths[i], globals()): building_data_2023[i] for i in range(len(Commune_paths))}

#%%

# get all typologies sorted 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

for i, (k, v) in enumerate(Building_dict_2023.items()):
    
    Commune =  Building_dict_2023[k]
    
    Typo_loads = {}
    Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]

    for typo in Typo_list: 
        
        Building_ID = Commune[Commune["Typo"]== typo]
        ID_list = Building_ID["Référence"].tolist()
        Complete_IDs = ["Livraison active."+elem+".kWh" for elem in ID_list]
        load_selected = LoadCurve_2023_dict[k][Complete_IDs]
        
        print(typo)
        if i== 0:
            Typo_loads[typo] = load_selected.copy()
            break 
        else : 
            Typo_loads[typo][Complete_IDs] = load_selected.loc[:,Complete_IDs]
    

print(Typo_loads["Ecole"])
#%%
    
    
    Commune_culture = Commune[Commune["Typo"]== "Culture"]
    Culture_list.extend(Commune_culture["Référence"].tolist())
    
    Commune_apems = Commune[Commune["Typo"]== "Apems"]
    Apems_ID.extend(Commune_apems["Référence"].tolist())
    
    Commune_institutions = Commune[Commune["Typo"]== "Commune"]
    Institutions_ID.extend(Commune_institutions["Référence"].tolist())
    
    Commune_bar = Commune[Commune["Typo"]== "Buvette"]
    Bar_ID.extend(Commune_bar["Référence"].tolist())
    
    Commune_parking = Commune[Commune["Typo"]== "Parking"]
    Parkinglot_ID.extend(Commune_parking["Référence"].tolist())
    
    



## Extracting Typologies 

#%% Plotting typologies 


custom_palette = sb.set_palette("deep")

# plot of the 
sb.lineplot(data=School_loads, linewidth=0.5, palette=custom_palette)
plt.title('Electric consumptions')
plt.xlabel('dates')
plt.ylabel('kWh_{el}')
plt.legend().set_visible(False)
plt.show()

    


#%%

