# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:27:03 2024

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


#%%

def get_load_curves(total_cons=0):
    # DEFINING PATHS
    ## Generic path of the folder in your local terminal 
    current_script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_script_path)
    
    
    ## Creating specificpath for each commune
    renens = os.path.join(parent_directory, "Renens")
    ecublens = os.path.join(parent_directory, "Ecublens")
    crissier = os.path.join(parent_directory, "Crissier")
    chavannes = os.path.join(parent_directory, "Chavannes")
    
    Commune_paths = [renens, ecublens, crissier, chavannes]
    
    
    ## reading excel files 
    load_data_2023 = []
    load_data_2022 = []
    building_data_2023 = []
    pv_2022 = []
    pv_commune = []
    locals_dict = locals()
    
    
    for i, commune in enumerate(Commune_paths):
        
        # extracting load curves 
        load_2023 = pd.read_excel(os.path.join(commune, f.get_variable_name(commune, locals_dict) +"_courbes_de_charge_podvert_2023.xlsx"), sheet_name=2)
        load_2023.set_index("Date", inplace=True)
        load_2022 = pd.read_excel(os.path.join(commune, f.get_variable_name(commune, locals_dict) +"_cch_podvert_2022.xlsx"), sheet_name=2)
        load_2022.set_index("Date", inplace=True)
        
        if total_cons ==1:
            given_file = f.get_variable_name(commune, locals_dict) + "_cch_plus_20MWh_complement.xlsx"
            for root, dirs, files in os.walk(commune):
                if given_file in files: 
                    file_path = os.path.join(root, given_file)
                    print(file_path)
                    try:
                        # Read the Excel file using pandas
                        pv_prod_2022 = pd.read_excel(file_path, sheet_name=1)
                        print(pv_prod_2022)
                        pv_prod_2022.set_index("Date", inplace=True)
                        # Perform actions with the DataFrame 'df'
                        print(f"Successfully read {given_file} in {root}.")
                        # Add more code to work with the DataFrame if needed
                        pv_2022.append(pv_prod_2022)
                        pv_commune.append(f.get_variable_name(commune, locals_dict))
                    except Exception as e:
                        # Handle any exceptions raised during reading or processing
                        print(f"An error occurred while reading {given_file} in {root}: {e}")
                else:
                    print(f"{given_file} not found in {root}.")
                    # Add code to handle this case or simply pass
        
            
        # extracting buildings
        buildings = pd.read_excel(os.path.join(commune, f.get_variable_name(commune, locals_dict) +"_courbes_de_charge_podvert_2023.xlsx"), sheet_name=0)
        
        # storing data 
        load_data_2023.append(load_2023)
        load_data_2022.append(load_2022)
        
        building_data_2023.append(buildings)
        
        #adding to the dict of ocal variables to reach  them all
        locals_dict.update(locals())
    
    
    LoadCurve_2023_dict = {f.get_variable_name(Commune_paths[i], locals_dict): load_data_2023[i] for i in range(len(Commune_paths))}
    LoadCurve_2022_dict = {f.get_variable_name(Commune_paths[i], locals_dict): load_data_2022[i] for i in range(len(Commune_paths))}
    Building_dict_2023 = {f.get_variable_name(Commune_paths[i], locals_dict): building_data_2023[i] for i in range(len(Commune_paths))}
    
    pv_2022_dict = {pv_commune[i]: pv_2022[i] for i in range(len(pv_commune))}

    return LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict


#%% get all typologies sorted for all provided year 

#School_loads =[]
#Culture_loads = []
#Apems_loads = []
#Institutions_loads = []
#Bar_loads =[]
#Parkinglot_loads =[]

#1 > total load, 0 normalized load 
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(1)

#%%
Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]
print(type(Building_dict_2023), type(LoadCurve_2022_dict), type(Typo_list))
#getting typologies from 2022
Typo_loads_2022, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list, 0)

#getting typologies from 2023
Typo_loads_2023, Correspondance = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list, 0)

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


plt.bar(range(len(mean_values)), 4*mean_values, color=sb.color_palette("hls", 13)[8])
# Set the x-axis ticks to the list of names
plt.xticks(range(len(mean_values)), Loads.columns.tolist())
#plt.yscale("log")
plt.ylabel("load [$kW_{el}/m^2$]")
plt.xlabel("Consumers' IDs")
plt.title("Overall average load per meter squared")
plt.grid(axis="y")
#%%
print(type(locals()))