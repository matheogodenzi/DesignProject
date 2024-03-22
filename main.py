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
from datetime import datetime
import matplotlib.dates as mdates

"""functions imports"""

import functions as f
import controls as c


"""data acquisition"""

#%%

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

#%% get all typologies sorted 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

Typo_loads = {}
Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]

for i, (k, v) in enumerate(Building_dict_2023.items()):
    
    Commune =  Building_dict_2023[k] #v

    for typo in Typo_list: 
        
        Building_ID = Commune[Commune["Typo"]== typo]
        ID_list = Building_ID["Référence"].tolist()
        surface_list = Building_ID["Surface"].tolist()
        address_list = Building_ID["Emplacement"].tolist()
        Complete_IDs = ["Livraison active."+elem+".kWh" for elem in ID_list]
        load_selected = LoadCurve_2023_dict[k][Complete_IDs]
        
        #linking surface to ID
        surf_id_dict = {k: v for k, v in zip(Complete_IDs, surface_list)}
        address_id_dict = {k: v for k, v in zip(Complete_IDs, address_list)}
        
        for col_name in load_selected.columns:
            load_selected /= surf_id_dict[col_name]
        
        if i== 0:
            Typo_loads[typo] = load_selected.copy()
        else : 
            df = Typo_loads[typo].copy() 
            df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
            Typo_loads[typo] = df
        
        #renaming columns with adresses 
        Typo_loads[typo].rename(columns=address_id_dict, inplace=True)
            
    
#print(Typo_loads)


#%% Plotting typologies 

"""
custom_palette = sb.set_palette("deep")

# plot of the 
sb.lineplot(data=Typo_loads["Commune"].head(900), linewidth=0.5, palette=custom_palette)
plt.title('Electric consumptions')
plt.xlabel('dates')
plt.ylabel('kWh_{el}')
plt.legend().set_visible(False)
plt.show()
"""

#%% Averaging over a a given period (smoothing of the curve)

# parameters to change
Typology = "Apems"
Period = "week"

Loads = Typo_loads[Typology]
plt.plot(Loads.head[:900,:])
plt.show()

#%%


# smoothing calculation
Loads = Typo_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)

# plotting 
f.plot_tendency(Tendency, title= Typology+" "+ Period, period=Period)



#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Ecole"
Period = "week"

# smoothing calculations
Loads = Typo_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)

# plotting 
updated_tendency = f.plot_mean_load(Tendency, Period, Typology)

#verifying that the mean and standard deviation have their own columns in the output
print(updated_tendency)
#%% creating a typical day 

#data = Typo_loads["Apems"]
#period = "week"


data_day = f.typical_period(Loads, Period)

f.plot_typical_week(data_day, Typology)

typical_day_schools = f.typical_period(Loads, Period)

f.plot_mean_load(typical_day_schools, Period, Typology)

#%% creating a typical week

#data = Typo_loads["Apems"]
#period = "week"


data_day = f.typical_period(Loads, Period)

f.plot_typical_week(data_day, Typology)

#%% test 
tendency = f.period_tendencies(Loads, Period)



#%%

f.plot_tendency(tendency, title="Load curve weekly average for "+Typology+"s", period=Period, show_legend=True)
f.plot_mean_load(tendency, period=Period, Typology=Typology)


#%%








