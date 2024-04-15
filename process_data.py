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

def discriminate_typologies(Building_dict, LoadCurve_dict, Typo_list):
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
    
        for typo in Typo_list: 
            
            Building_ID = Commune[Commune["Typo"]==typo]
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
                Typo_loads[typo] = load_selected.copy()
            else : 
                df = Typo_loads[typo].copy() 
                df[Complete_IDs] = load_selected.loc[:,Complete_IDs]
                Typo_loads[typo] = df
            
            #renaming columns with adresses 
            Typo_loads[typo].rename(columns=address_id_dict, inplace=True)
      
    return Typo_loads


def discriminate_conslevels(Building_dict, LoadCurve_dict, Cons_list):
    """
    

    Parameters
    ----------
    Building_dict : TYPE
        DESCRIPTION.
    LoadCurve_dict : TYPE
        DESCRIPTION.
    Cons_list : TYPE
        DESCRIPTION.

    Returns
    -------
    Cons_loads : TYPE
        DESCRIPTION.

    """
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
            Cons_loads[cons].rename(columns=address_id_dict, inplace=True)
      
    return Cons_loads