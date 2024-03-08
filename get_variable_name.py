# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:56:39 2024

@author: matheo

"""

""" import libraries"""
import os 
import numpy as np
import pandas as pd 

def get_variable_name(var, namespace):
    """returns the name of the variable in a string format"""
    
    for name, obj in namespace.items():
        if obj is var:
            return name
    return None


def get_load_curves():
    """returns all load curves for the for communes studied"""
    
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
        
        Commune_data.append(pd.read_excel(commune + "\\" + get_variable_name(commune, locals()) +"_courbes_de_charge_podvert_2023.xlsx", sheet_name=2))
    
    Commune_dict = {get_variable_name(Commune_paths[i], locals()): Commune_data[i] for i in range(len(Commune_paths))}


    return Commune_dict
