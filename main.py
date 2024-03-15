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
    
    Commune =  Building_dict_2023[k]

    for typo in Typo_list: 
        
        Building_ID = Commune[Commune["Typo"]== typo]
        ID_list = Building_ID["Référence"].tolist()
        Complete_IDs = ["Livraison active."+elem+".kWh" for elem in ID_list]
        load_selected = LoadCurve_2023_dict[k][Complete_IDs]
        
        if i== 0:
            Typo_loads[typo] = load_selected.copy()
        else : 
            df = Typo_loads[typo].copy() 
            df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
            Typo_loads[typo] = df
    
#print(Typo_loads)


#%% Plotting typologies 


custom_palette = sb.set_palette("deep")

# plot of the 
sb.lineplot(data=Typo_loads["Commune"].head(900), linewidth=0.5, palette=custom_palette)
plt.title('Electric consumptions')
plt.xlabel('dates')
plt.ylabel('kWh_{el}')
plt.legend().set_visible(False)
plt.show()

#%% Averaging over a day (24h)

result = f.average_24h(Typo_loads["Ecole"])

#%% seaborn graphic average 24h

custom_palette = sb.set_palette("deep")

# plot of the 
sb.lineplot(data=result, linewidth=1, palette=custom_palette, linestyle="solid")
plt.title('Electric consumptions')
plt.xlabel('days')
plt.ylabel('kWh_{el}')
#plt.legend().set_visible(False)
plt.legend(title='Custom Legend', loc='upper left')
plt.show()

#%% matplotlib graphic average 24h

plt.plot(result, linewidth=0.5)
plt.title('Electric consumptions ***insert Typology***')
plt.xlabel('days')
plt.ylabel('kWh_{el}')
#plt.legend().set_visible(False)

# labels 

# Place legend outside the plot area
plt.legend([i for i in range(result.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.grid()
plt.show()


#%%Averaging over a week 

res = f.av_1_week(Typo_loads["Ecole"])

#%% matplotlib graphic average 1 week

plt.plot(res, linewidth=1)
plt.title('Electric consumptions ***insert Typology***')
plt.xlabel('weeks')
plt.ylabel('kWh_{el}')
#plt.legend().set_visible(False)

# labels 

# Place legend outside the plot area
plt.legend([i for i in range(result.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.grid()
plt.show()

#%%Averaging over a week 

res = f.av_1_month(Typo_loads["Ecole"])

#%% matplotlib graphic average 1 month


plt.plot(res, linewidth=1)
plt.title('Electric consumptions ***insert Typology***')
plt.xlabel('month')
plt.ylabel('kWh_{el}')
#plt.legend().set_visible(False)

# labels 

# Place legend outside the plot area
plt.legend([i for i in range(result.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of legend
plt.grid()
plt.show()

#%% calculating mean and standard deviation for a typical day the year 

data_frame = f.av_1_month(Typo_loads["Ecole"])
row_mean = data_frame.mean(axis=1)
row_std = data_frame.std(axis=1)
data_frame["Mean"] = row_mean
data_frame["STD"] = row_std

#%%
plt.figure()

plt.plot(data_frame["Mean"].values+data_frame["STD"].values, color="blue", alpha=0.3)
plt.plot(data_frame["Mean"].values, color="blue")
plt.plot(data_frame["Mean"].values-data_frame["STD"].values, color="blue", alpha=0.3)
plt.show()


#%% creating a typical day 

data_day = Typo_loads["Buvette"]

days = 365
for i in range(365):
    
    if i == 0: 
        data_day = Typo_loads["Apems"].iloc[:96, :]
    else: 
        data_day += Typo_loads["Apems"].iloc[(i-1)*96:i*96, :].values
    
data_day /= days


plt.plot(data_day)
plt.legend([i for i in range(result.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


#%% creating typical week 

Typologie = "Ecole"
weeks = 52
for i in range(weeks):
    
    if i == 0: 
        data_week = Typo_loads[Typologie].iloc[:96*7, :]
    else: 
        data_week += Typo_loads[Typologie].iloc[(i-1)*96*7:i*96*7, :].values
    
data_week /= days


plt.plot(data_week)
plt.legend([i for i in range(result.shape[1])], bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()















