# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:23:53 2024

@author: matheo
"""

"""libraries imports """

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
import sklearn as skl
import pandas as pd
import os
import seaborn as sb
from datetime import datetime
import matplotlib.dates as mdates


"""modules imports"""

import functions as f
import controls as c
import auto_analysis as aa


"""functions """





def discriminate_conslevels(Building_dict, LoadCurve_dict, Cons_list, ID_mapping):

    Cons_loads = {}
    
    for i, (k, v) in enumerate(Building_dict.items()):
        
        Commune =  Building_dict[k] #v
    
        for cons in Cons_list: 
            
            Building_ID = Commune[Commune["Cons"]==cons]
            ID_list = Building_ID["Référence"].tolist()
            surface_list = Building_ID["Surface"].tolist()
            address_list = Building_ID["Emplacement"].tolist()
            #address_list = [address + " " + period for address in adress_list]
            Complete_IDs = ["Livraison active."+elem+".kWh" for elem in ID_list]
            load_selected = LoadCurve_dict[k][Complete_IDs]
            
            
            #linking surface to ID
            surf_id_dict = {k: v for k, v in zip(Complete_IDs, surface_list)}
            address_id_dict = {k: v for k, v in zip(Complete_IDs, address_list)}
            
            for col_name in load_selected.columns:
                load_selected /= surf_id_dict[col_name]
            
            if i== 0:
                Cons_loads[cons] = load_selected.copy()
            else : 
                df = Cons_loads[cons].copy() 
                df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
                Cons_loads[cons] = df
            
            #renaming columns with adresses 
            Cons_loads[cons].rename(columns=ID_mapping, inplace=True)
      
    return Cons_loads


def discriminate_typologies(Building_dict, LoadCurve_dict, Typo_list, normalization=True):
    Typo_loads = {}
    ID_mapping = {}
    
    if not Building_dict:  # Check if Building_dict is empty
        return Typo_loads, ID_mapping
    
    for i, (k, v) in enumerate(Building_dict.items()):
        Commune = Building_dict[k]
    
        for j, typo in enumerate(Typo_list): 
            Building_ID = Commune[Commune["Typo"]==typo]
            ID_list = Building_ID["Référence"].tolist()
            surface_list = Building_ID["Surface"].tolist()
            address_list = Building_ID["Emplacement"].tolist()
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
                
            surf_id_dict = {k: v for k, v in zip(Complete_IDs, surface_list)}
            address_id_dict = {k: v for k, v in zip(simple_IDs, address_list)}
            simple_id_dict = {k:v for k, v in zip(Complete_IDs, simple_IDs)}
            
            ID_mapping.update(address_id_dict)  # Update ID_mapping with simple_id_dict
            
            if normalization : 
                for col_name in load_selected.columns:
                    load_selected[col_name] /= surf_id_dict[col_name]
                
            if i== 0:
                Typo_loads[typo] = load_selected.copy()
            else : 
                df = Typo_loads[typo].copy() 
                df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
                Typo_loads[typo] = df
            
            Typo_loads[typo].rename(columns=simple_id_dict, inplace=True)
      
    return Typo_loads, ID_mapping  # Return Typo_loads and ID_mapping


def get_load_curves(total_cons=False):
    


    """
    

    Parameters
    ----------
    total_cons : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    LoadCurve_2023_dict : TYPE
        DESCRIPTION.
    LoadCurve_2022_dict : TYPE
        DESCRIPTION.
    Building_dict_2023 : TYPE
        DESCRIPTION.
    pv_2022_dict : TYPE
        DESCRIPTION.

    """
    
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
        
        if total_cons == True:
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


def sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, normalization=True):

    Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Buvette", "Parking"]
    print(type(Building_dict_2023), type(LoadCurve_2022_dict), type(Typo_list))
    #getting typologies from 2022
    Typo_loads_2022, _ = discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list, normalization)
    
    #getting typologies from 2023
    Typo_loads_2023, Correspondance = discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list, normalization)
    
    # creating overall dictionnary
    Typo_all_loads = {}
    for typo in Typo_list:
        Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
        
    #print(Typo_loads)
    
    return Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance