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
import auto_analysis as aa
import process_data as p


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

#%% get all typologies sorted for all provided year 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]

#getting typologies from 2022
Typo_loads_2022 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)


#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 4].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)

#%% creating a benchmark over available years

# Obtain a typical year
typical_year = f.typical_period(Loads,  "year")

#%% Obtain a typical day from benchmarked typical year
typical_day = f.typical_period(typical_year, Period)

#%% plotting one specific load over given benchmark 

Load1 = typical_day.iloc[:, 4].to_frame()
f.plot_mean_load(Load1, typical_day, Period, Typology)



#%% test of plotting benchark averages alone 

# Annual weekly smoothing of the loag curve
tendency = f.period_tendencies(typical_year, Period)

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


#%% extraction of a given week 


Typology = "Ecole"
Period = "week"

# smoothing calculations
Loads = Typo_all_loads[Typology]

typical_week = f.typical_period(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 6].to_frame()
#print(single_load)
interest_period = aa.extract_period(single_load, pd.to_datetime("15.01.2023 00:15:00"), pd.to_datetime("22.01.2023 00:00:00"))

# plotting 
updated_tendency = f.plot_mean_load(interest_period, typical_week, Period, Typology)

#%% Extracting all instances of the same exact time

time_of_interest = aa.extract_time(Load1, pd.Timestamp('00:00:00'))


#%% align years








