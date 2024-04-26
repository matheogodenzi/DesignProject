# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:58:17 2024

@author: matheo
"""


"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
from scipy.stats import shapiro
import seaborn as sb
"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


"""data acquisition"""

#%%
def discriminate_typologies_absolute(Building_dict, LoadCurve_dict, Typo_list):
    """

    Parameters
    ----------
    Building_dict : TYPE
        DESCRIPTION.
    LoadCurve_dict : TYPE
        DESCRIPTION.
    Typo_list : TYPE
        DESCRIPTION.

    Returns
    -------
    Typo_loads : TYPE
        DESCRIPTION.

    """
    Typo_loads = {}
    
    for i, (k, v) in enumerate(Building_dict.items()):
        
        Commune =  Building_dict[k] #v
    
        for j, typo in enumerate(Typo_list): 
            
            Building_ID = Commune[Commune["Typo"]==typo]
            ID_list = Building_ID["Référence"].tolist()
            surface_list = Building_ID["Surface"].tolist()
            address_list = Building_ID["Emplacement"].tolist()
            #address_list = [address + " " + period for address in adress_list]
            Complete_IDs = ["Livraison active."+elem+".kWh" for elem in ID_list]
            load_selected = LoadCurve_dict[k][Complete_IDs]
            
            # translating from french to english
            if typo == "Ecole":
                simple_IDs = ["S" + str(i) + str(j) + str(k) for k in range(len(address_list))]
            elif typo == "Commune" or typo == "Commune2":
                simple_IDs = ["A" + str(i) + str(j) + str(k) for k in range(len(address_list))]
            elif typo == "Culture":
                simple_IDs = ["C" + str(i) + str(j) + str(k) for k in range(len(address_list))]
            elif typo == "Apems":
                simple_IDs = ["D" + str(i) + str(j) + str(k) for k in range(len(address_list))]
            else : 
                simple_IDs = ["O" + str(i) + str(j) + str(k) for k in range(len(address_list))]
                
                
                
                
            #linking surface to ID
            surf_id_dict = {k: v for k, v in zip(Complete_IDs, surface_list)}
            address_id_dict = {k: v for k, v in zip(Complete_IDs, address_list)}
            simple_id_dict = {k:v for k, v in zip(Complete_IDs,simple_IDs)}
            
            for col_name in load_selected.columns:
                load_selected[col_name]/=surf_id_dict[col_name]
            
            if i== 0:
                Typo_loads[typo] = load_selected.copy()
            else : 
                df = Typo_loads[typo].copy() 
                df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
                Typo_loads[typo] = df
            
            #renaming columns with adresses 
            #Typo_loads[typo].rename(columns=address_id_dict, inplace=True)
            
            #renaiming columns with simple IDs to conserve anonimity 
            Typo_loads[typo].rename(columns=simple_id_dict, inplace=True)
      
    return Typo_loads



#%%

# DEFINING PATHS
## Generic path of the folder in your local terminal 
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)


## Creating specificpath for each commune
renens = parent_directory + "\\Renens"
ecublens = parent_directory + "\\Ecublens"
crissier = parent_directory + "\\Crissier"
chavannes = parent_directory + "\\Chavannes"

Commune_paths = [renens, ecublens, crissier, chavannes]


## reading excel files 
load_data_2023 = []
load_data_2022 = []
building_data_2023 = []
pv_2022 = []


for i, commune in enumerate(Commune_paths):
    
    # extracting load curves 
    load_2023 = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2)
    load_2023.set_index("Date", inplace=True)
    load_2022 = pd.read_excel(commune+"\\"+ f.get_variable_name(commune, globals()) +"_cch_podvert_2022.xlsx", sheet_name=2)
    load_2022.set_index("Date", inplace=True)
    
    given_file ="\\" + f.get_variable_name(commune, globals()) + "_cch_plus_20MWh_complement"
    pv_commune = []
    for root, dirs, files in os.walk(commune):
        if given_file in files: 
            file_path = os.path.join(root, given_file)
            try:
                # Read the Excel file using pandas
                pv_prod_2022 = pd.read_excel(file_path)
                pv_prod_2022.set_index("Date", inplace=True)
                # Perform actions with the DataFrame 'df'
                print(f"Successfully read {given_file} in {root}.")
                # Add more code to work with the DataFrame if needed
                pv_2022.append(pv_prod_2022)
                pv_commune.append(f.get_variable_name(commune, globals()))
            except Exception as e:
                # Handle any exceptions raised during reading or processing
                print(f"An error occurred while reading {given_file} in {root}: {e}")
        else:
            print(f"{given_file} not found in {root}.")
            # Add code to handle this case or simply pass
    
        
    # extracting buildings
    buildings = pd.read_excel(commune + "\\" + f.get_variable_name(commune, globals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=0)
    
    # storing data 
    load_data_2023.append(load_2023)
    load_data_2022.append(load_2022)
    
    building_data_2023.append(buildings)


LoadCurve_2023_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2023[i] for i in range(len(Commune_paths))}
LoadCurve_2022_dict = {f.get_variable_name(Commune_paths[i], globals()): load_data_2022[i] for i in range(len(Commune_paths))}
Building_dict_2023 = {f.get_variable_name(Commune_paths[i], globals()): building_data_2023[i] for i in range(len(Commune_paths))}
pv_2022_dict = {pv_commune[i]: pv_2022[i] for i in range(len(pv_commune))}

print(pv_2022_dict)

#%% get all typologies sorted for all provided year 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]
print(type(Building_dict_2023), type(LoadCurve_2022_dict), type(Typo_list))
#getting typologies from 2022
Typo_loads_2022 = discriminate_typologies_absolute(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023 = discriminate_typologies_absolute(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)

#%%
# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

# Assuming df is your DataFrame
# Replace zeros with NaN values
df_nan = Loads.replace(0, np.nan)

# Calculate the mean over columns
mean_values = df_nan.mean()


plt.scatter(range(len(mean_values)), 4*mean_values)
# Set the x-axis ticks to the list of names
plt.xticks(range(len(mean_values)), Loads.columns.tolist())
plt.yscale("log")