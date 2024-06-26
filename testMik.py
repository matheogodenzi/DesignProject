# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:06:23 2024

@author: mimag

For main developments
"""

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

#%% get all typologies sorted 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Commune2", "Buvette", "Parking"]

#getting typologies from 2022
Typo_loads_2022 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)


#%% get consumptions sorted

Cons_list = ["bas", "moyen", "haut", "fort"]

Cons_loads_2022 = p.discriminate_conslevels(Building_dict_2023, LoadCurve_2022_dict, Cons_list)

Cons_loads_2023 = p.discriminate_conslevels(Building_dict_2023, LoadCurve_2023_dict, Cons_list)

Cons_all_loads = {}

for cons in Cons_list:
    Cons_all_loads[cons] = pd.concat([Cons_loads_2022[cons], Cons_loads_2022[cons]], axis=0)


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
Typology = "culture"
Period = "week"
#%%



#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Commune2"
Period = "week"

# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)

# plotting 
updated_tendency = f.plot_mean_load(Tendency, Period, Typology)

#verifying that the mean and standard deviation have their own columns in the output
print(updated_tendency)
#%% creating a typical day 

#data = Typo_loads["Apems"]
#period = "week"


data_day = f.typical_period(Loads, Period)

f.plot_typical_day(data_day, Typology)

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


#%% total consumption control

tested_curve = Loads.iloc[:,1].to_frame()

granulo = "day"

total_cons_tested = c.total_cons_ctrl(tested_curve, Loads, granulo)
print(total_cons_tested)

#%% calculating mean and standard deviation for a typical day the year 


# parameters to change
Typology = "Commune2"
Period = "week"

# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Loads.iloc[:,2].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
#updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)
updated_tendency = c.plot_mean_load_control(smoothed_load, Tendency, Period, Typology)




#%% yearly_consumption computation


single_load = Loads.iloc[:,2].to_frame()

print(c.quarterly_consumption(single_load, Loads))


#%% typical week control

# parameters to change
Typology = "Ecole"
Period = "week"
Loads = Typo_all_loads[Typology]

typical_period_load = f.typical_period(Loads, Period)
single_load = typical_period_load.iloc[:,10].to_frame()

#typical_period_test = c.typical_period_control(single_load, typical_period_load)
#print(typical_period_test)

# for comparison
data_week = f.typical_period(Loads, Period)
c.plot_typical_week_control(single_load, data_week, Typology)

c.plot_typical_week_control_clean(single_load, data_week, Typology)


#%% typical day control

Typology = "Ecole"
Period = "day"
Loads = Typo_all_loads[Typology]

data_day = f.typical_period(Loads, Period)
c.plot_typical_day_control_clean(single_load, data_day, Typology)