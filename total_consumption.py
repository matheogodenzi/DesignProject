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
            simple_id_address_dict = {k:v for k, v in zip(simple_IDs, address_list)}
            print(simple_id_address_dict)
            for col_name in load_selected.columns:
                load_selected[col_name]
            
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
      
    return Typo_loads, simple_id_address_dict



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
Typo_loads_2022, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list)

#getting typologies from 2023
Typo_loads_2023, Correspondance = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list)

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

#%% grading for comparison matrix - overload score 

x = Loads.columns
y = 4*mean_values

def get_score(typology_names, parameters):
    min_ = min(parameters)
    max_= max(parameters)
    
    grades = {}
    classes = {}
    
    for i, value in enumerate(parameters):
        grades[typology_names[i]] = 100*(parameters[i]-min_)/(max_-min_)
    
        if grades[typology_names[i]] <= 20 : 
            classes[typology_names[i]] = 1
        elif grades[typology_names[i]] <= 40 :
            classes[typology_names[i]] = 2
        elif grades[typology_names[i]] <= 60 :
            classes[typology_names[i]] = 3
        elif grades[typology_names[i]] <= 80 :
            classes[typology_names[i]] = 4
        elif grades[typology_names[i]] <= 100 :
            classes[typology_names[i]] = 5
    
    thresholds = [v/100*(max_-min_) + min_ for v in [0, 20, 40, 60, 80 , 100]]


    
    return grades, classes, thresholds 

grades, classes, thresholds = get_score(x, y)

plt.figure()
for i, (k, v) in enumerate(classes.items()):
    print(type(v))
    if v == 1:
        plt.bar(i,v, color="green")
    elif v == 2:
        plt.bar(i,v, color="yellow")
    elif v == 3:
        plt.bar(i,v, color="orange")
    elif v == 4:
        plt.bar(i,v, color="red" )
    elif v == 5:
        plt.bar(i,v, color="purple")
plt.grid(axis='y')
plt.xticks(range(len(x)), x)
plt.show()
#%% threshold calculation : 
    


#%%

#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#color list 
#color = ["darkblue", "royalblue", "green", "yellow", "orange", "red", "purple"]
color = sb.color_palette("husl", 29, 1)
colori = 0
#initiating figure
plt.figure()
for i, typo in enumerate(Typo_list) : 
    
    loads = Typo_all_loads[typo]
    # Obtain a typical year
    t_year = f.typical_period(loads,  "year")
    #obtain typical day 
    t_day = f.typical_period(t_year, "day")
    

    

    for j, col in enumerate(t_day.columns):
        t_day[col].plot(color=color[colori], label=col)
        colori += 1

plt.yscale("log")

# Manually set x-axis ticks to display only the hour component
#hours = mdates.HourLocator(interval=365)
#hours_fmt = mdates.DateFormatter("%H:%M")
#plt.gca().xaxis.set_major_locator(hours)
#plt.gca().xaxis.set_major_formatter(hours_fmt)
tick_labels = ["0"+str(i)+":00" if i < 10 else str(i)+":00" for i in range(0, 24, 2)]
tick_positions = [i*8 for i in range(12)]
plt.xticks(tick_positions, tick_labels, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Consumers")
plt.tight_layout(rect=[0, 0, 1, 2.3])
plt.xlabel("Hours")
plt.ylabel("Mean daily consumption [$kWh_{el}$]")
plt.title("Mean daily consumption")
plt.grid()
plt.show()

