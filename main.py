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


#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Apems"
Period = "month"

# smoothing calculations
Loads = Typo_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_loads[Typology].iloc[:, 0].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)

#%% creating a typical period 
typical_day_schools = f.typical_period(Loads, Period)

Load1 = typical_day_schools.iloc[:, 3].to_frame()

f.plot_mean_load(Load1, typical_day_schools, Period, Typology)



#%% test of plotting benchark averages alone 

# Annual weekly smoothing of the loag curve
tendency = f.period_tendencies(Loads, Period)

f.plot_mean_load(None, tendency, period=Period, Typology=Typology)


#%% All benchmark displayed  for a day


#data = Typo_loads["Apems"]
Period = "day"

tendency_day = f.period_tendencies(Loads, Period)
data_day = f.typical_period(Loads, Period)

# Typical day for all infrastructures 
f.plot_typical_day(data_day, Typology)


#daily smoothing along the year for all insfrastrctures 
f.plot_tendency(tendency_day, title="Load curve weekly average for "+Typology+"s", period=Period, show_legend=True)


#%% All Benchmark plotted for a week 

#data = Typo_loads["Apems"]
Period = "week"

# Annual weekly smoothing of the loag curve
tendency_week = f.period_tendencies(Loads, Period)
data_week = f.typical_period(Loads, Period)


#typical week for all infrastructures 
f.plot_typical_week(data_week, Typology)

#weekly smoothing along the year for all insfrastrctures 
f.plot_tendency(tendency_week, title="Load curve weekly average for "+Typology+"s", period=Period, show_legend=True)












