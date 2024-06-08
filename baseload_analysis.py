# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:30:43 2024

@author: matheo
"""

"""libraries import"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

"""functions imports"""

import functions as f
import process_data as p



#%% data acquisition
#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, False)
#%% creating a benchmark over available years

# parameters to change
Typology = "Commune"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)

# Obtain a typical year averaged
typical_year = f.typical_period(Loads,  "year")

Loads_2022 = Typo_loads_2022[Typology]
Loads_2023 = Typo_loads_2023[Typology]

my_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']

#%% creating datasets of everything but schools

# # Loads_buv = Typo_all_loads["Buvette"]
# # Loads_sport = Typo_all_loads["Sport"]
# # Loads_parking = Typo_all_loads["Parking"]
# Loads_voirie = Typo_all_loads["Commune"]
# Loads_admin  = Typo_all_loads["Admin"]
# Loads_garderie  = Typo_all_loads["Apems"]
# Loads_culture  = Typo_all_loads["Culture"]
# Loads_unique = pd.concat([Loads_voirie, Loads_admin, Loads_garderie, Loads_culture], axis=1)
# Loads = Loads_unique #kW/15'/m2
# #df = 4*Loads_unique.astype(np.longdouble) #kW/m2
# my_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']

#%%
def get_baseload_2(df):
    """
    
    kWhel/15'/m2'
    Delineate annual tendencies over days, weeks, and months
    


    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    result : DataFrame
        DataFrame containing the mean of the 6 smallest values of each column.
    """

    num_rows = df.shape[0]
    averages = []

    # Iterate over the DataFrame in chunks of 96 rows
    chunk_size = 96
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get the current chunk of 96 rows
        
        # Calculate the 6 smallest values of each column
        smallest_values = chunk.iloc[:16, :] #if using 96 we get the daily average and if using nlargest(n) we get the n largest data points of the day
        #print(smallest_values)
        # Calculate the mean of the smallest values for each column
        average_of_smallest = smallest_values.mean()
        
        averages.append(average_of_smallest)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    return result

def get_daily_max(df):
    """
    
    kWhel/15'/m2'
    Delineate annual tendencies over days, weeks, and months
    


    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    result : DataFrame
        DataFrame containing the mean of the 6 smallest values of each column.
    """

    num_rows = df.shape[0]
    maxima = []

    # Iterate over the DataFrame in chunks of 96 rows
    chunk_size = 96
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get the current chunk of 96 rows
        
        # Calculate the 6 smallest values of each column
        #smallest_values = chunk.iloc[:16, :] #if using 96 we get the daily average and if using nlargest(n) we get the n largest data points of the day
        #print(smallest_values)
        # Calculate the mean of the smallest values for each column
        maximum = chunk.max()
        
        maxima.append(maximum)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(maxima, axis=1).T
    
    return result

#%%
df = Loads.astype(np.longdouble)

# Remove duplicate indices
#df_no_duplicates = df[~df.index.duplicated(keep='first')]
baseloads = 4*get_baseload_2(df) #going from kWh/15'/m2 to kW/m2

#print(df[df.index.duplicated()]) # duplicates come from time change 
df = 1000*baseloads #W/m2 or W depeding on normalization

# Define your color palette
palette = sns.color_palette("hls", df.shape[1])
fig, ax = plt.subplots(figsize=(6, 5))

coef_df =  pd.DataFrame({'slope': [], 'y-intercept': []})
relative_slope = []
yi_2023 = []
yf_2023 = []
# Perform linear regression and plot for each column
for i, column in enumerate(df.columns):
    #if i in [0, 3, 6, 8, 12]: #low-level
            #plt.ylim(0.0002, 0.0020)
    #if i in [1, 5, 7, 10, 13]: #medium level
            #plt.ylim(0.0001, 0.0012)
    #if i in [2, 4, 9, 11, 14]:
            #plt.ylim(0.0002, 0.0030)
        
            #plt.ylim(-5, 6)
            
            # Replace 0 values with NaN
            infra = df[column].copy()
            # Convert zeros to NaN
            infra.replace(0, np.nan, inplace=True)
            
            # Drop NaN values
            infra.dropna(inplace=True)
    
            X = np.array(infra.index).reshape(-1, 1)   # Independent variable
            #print(X)
            y = infra.values.reshape(-1, 1)# factor 4 to go from kWh/15'/m2 to kW/m2
            
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Extract regression coefficients
            coefficients = model.coef_
            intercept = model.intercept_
            
            specific_values = {'slope': coefficients[0][0] , 'y-intercept': intercept[0]}

            coef_df = coef_df.append(specific_values, ignore_index=True)
    
            # Plot regression line
            y_reg = model.predict(X)
            #print(type(y_reg))
            ax.plot(X, y_reg-specific_values['y-intercept'], c=my_colors[i], label=column, linewidth=2, alpha = 1)
            
            # Plot data points
            ax.scatter(X, y-specific_values['y-intercept'], color=my_colors[i], alpha=0.3, s=1)
            
            relative_slope.append(coefficients[0][0]/y_reg[0][0])
            yi_2023.append(y_reg[-365][0])
            yf_2023.append(y_reg[-1][0])
            # Set labels and title
            ax.set_title(f'{column}')
            ax.set_xlabel('Jours')
            ax.set_ylabel('Charge de base [$W_{el}/m^2$]')
            ax.legend()

#plt.ylim(0, 3e-12)
#plt.yscale("log")
plt.title("Evolution de la charge de base")
# Place legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clients")
plt.grid(which='both')
plt.tight_layout()
plt.show()


"""plotting 2023 variation"""
yi = np.array(yi_2023)
yf = np.array(yf_2023)
baseload_variations = 100*(yf-yi)/yi

plt.figure(figsize=(6,5))
plt.bar(range(len(baseload_variations)),baseload_variations)
plt.show()


#%%
num_rows = baseloads.shape[0]
chunk_size = 7 # if 7 > week, if 30 > month-ish 
df = baseloads 


averages = []
# Iterate over the DataFrame in chunks of 96 rows
for i in range(0, num_rows, chunk_size):
    chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
    chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
    averages.append(chunk_avg)  # Append the averages to the list
    print(i)
# Concatenate the averages into a single DataFrame
result = pd.concat(averages, axis=1).T


# Define your color palette
palette = sns.color_palette("bright", 2*result.shape[0])


plt.figure
plt.plot(result)
plt.grid()
plt.xlabel("weeks of 2022 and 2023")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.show()

plt.figure()
plt.plot((result.head(53).values+result.tail(53).values)/2)
plt.grid()
plt.xlabel("weeks of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
#plt.yscale("log")
plt.show()

#%% average daily baseload throughout the years

plt.figure
plt.plot(baseloads)
plt.grid()
plt.xlabel("weeks of the year")
plt.show()

plt.figure(figsize=(6,5))

for i, column in enumerate(result.columns):
    #if i in [0, 1, 4, 7, 9]: #low-level
            #plt.ylim(0.00005, 0.0003)
    #if i in [5, 8, 10, 11, 3]: #medium level
            #plt.ylim(0.00005, 0.0005)
    #if i in [2, 6,12]:
    
            if column == "V330" or column == "E202" or column == "E301" or column=="B140":
                mean = 1000*(baseloads[column].tail(365).values)
                plt.plot(mean, color=my_colors[i], label=column)
            else : 
                mean = 1000*(baseloads[column].head(365).values + baseloads[column].tail(365).values) / 2
                plt.plot(mean, color=my_colors[i], label=column)

            print(f"{column} : {np.max(mean)/np.min(mean)}")
            
#plt.yscale('log')

plt.grid(which="both", alpha=0.5)
plt.xlabel("Jours de l'année")
plt.ylabel("Charge de base - [$W_{el}/m^2$]")
plt.title("Profil annuel de la charge de base").set_position([0.55, 1])
plt.legend(title= "Clients", loc='upper left', bbox_to_anchor=(1, 1))
#plt.legend()
#plt.subplots_adjust(top=2)
plt.show()

#%% smoothing calculations

# Reset all rc settings to default
mpl.rcdefaults()

# Calculate mean for each column

average_array = 1000*(baseloads.iloc[:365,:].values+baseloads.iloc[365:,:].values)/2 #Wel/m2
baseloads_av = pd.DataFrame(average_array, columns=baseloads.columns)

for i, column in enumerate(baseloads_av.columns):
    
            if column == "V330" or column == "E202" or column == "E301" or column=="B140":
                baseloads_av[column] = 1000*(baseloads[column].tail(365).values)

means = baseloads_av.mean(axis=0)

# Plot boxplot for aboslute values
plt.figure(figsize=(6,5))
flierprops = dict(marker='*', markerfacecolor='b', markersize=4, linestyle='none', label="extrêmes")

boxplot = baseloads_av.boxplot(flierprops=flierprops)
plt.scatter(range(1, len(baseloads_av.columns) + 1), means, color='red', label='Mean', zorder=3, s=10)
plt.xticks(ticks=range(1, len(baseloads_av.columns) + 1), labels=baseloads_av.columns, rotation=45)
plt.xlabel("Identifiants des clients")
plt.ylabel("Charge [$W_{el}/m^2$]")
plt.title("Distribution annuelle moyenne de la charge de base journalière")
plt.grid(axis="x")

# Extracting the boxplot elements for creating legend
boxes = [item for item in boxplot.findobj(match=plt.Line2D)][::6]  # boxes
medians = [item for item in boxplot.findobj(match=plt.Line2D)][5::6]  # medians
whiskers = [item for item in boxplot.findobj(match=plt.Line2D)][2::6]  # whiskers
caps = [item for item in boxplot.findobj(match=plt.Line2D)][3::6]  # caps

# Create legend with labels
plt.legend([medians[0], caps[0], plt.Line2D([], [], color='red', marker='o', linestyle='None'),plt.Line2D([], [], color='black', marker='*', linestyle='None') ], 
           [ 'Mediane', 'Bornes', 'Moyenne', 'Extrêmes'],loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


y = baseload_variations
x = df.columns

# Create a bar plot
plt.figure(figsize=(6,5))
plt.bar(np.arange(len(y)), y, color="royalblue")

yy = baseload_variations

mi = min(yy)
ma = max(yy)

thresholds = [v/100*(ma-mi)+mi for v in [0, 20, 40, 60, 80, 100]]

# Adding labels under each bar
plt.xticks(np.arange(len(x)), x)
plt.tick_params(axis='both', which='major', labelsize=9, rotation=45)
plt.title("Variation annuelle de la charge de base")
plt.xlabel("Identifiants des consommateurs")
plt.ylabel("Variation [%]")
plt.grid(axis='y')

thresholds
#%% creating datasets of everything but schools

# Loads_buv = Typo_loads_2023["Buvette"]
# Loads_sport = Typo_loads_2023["Sport"]
# Loads_parking = Typo_loads_2023["Parking"]
# Loads_voirie = Typo_loads_2023["Commune"]
# Loads_admin  = Typo_loads_2023["Admin"]
# Loads_garderie  = Typo_loads_2023["Apems"]
# Loads_culture  = Typo_loads_2023["Culture"]
# Loads_unique = pd.concat([Loads_buv, Loads_sport, Loads_parking, Loads_voirie, Loads_admin, Loads_garderie, Loads_culture], axis=1)
# Loads = 1000*Loads_unique #W/15'/m2
# #df = 4*Loads_unique.astype(np.longdouble) #kW/m2


#%% baseload relative plot only 
"""If you want to have both years instead of their average, change typical_year by Loads"""
#df_nan = Loads.replace(0, np.nan)
max_load = (1000*4*get_daily_max(Loads)).mean(axis=0)
base_load = (1000*4*get_baseload_2(Loads)).mean(axis=0)


plt.figure(figsize=(6,5))
plt.bar(range(1, len(Loads.columns) + 1),max_load, color="royalblue")
plt.bar(range(1, len(Loads.columns) + 1),base_load, color="darkorange")
plt.xticks(ticks=range(1, len(Loads.columns) + 1), labels=Loads.columns, rotation=45)
plt.ylabel("Charge [$W_{el}/m^2$]")
plt.xlabel("Identifiants des consommateurs")
plt.title("Part relative de la charge de base")
plt.grid(axis="y")
plt.legend(["Charge maximale", "charge de base"])

baseload_ratio = base_load/max_load
#%%
import numpy as np
import matplotlib.pyplot as plt

# Vos données max_load et base_load
max_load = (1000 * 4 * get_daily_max(Loads)).mean(axis=0)
base_load = (1000 * 4 * get_baseload_2(Loads)).mean(axis=0)

# Calcul du pourcentage de base_load par rapport à max_load
percentage_base_load = (base_load / max_load) * 100

plt.figure(figsize=(6,5))
bars1 = plt.bar(range(1, len(Loads.columns) + 1), max_load, color="royalblue")
bars2 = plt.bar(range(1, len(Loads.columns) + 1), base_load, color="darkorange")
plt.xticks(ticks=range(1, len(Loads.columns) + 1), labels=Loads.columns, rotation=45)
plt.ylabel("Charge [$W_{el}/m^2$]")
plt.xlabel("Identifiants des consommateurs")
plt.title("Part relative de la charge de base")
plt.grid(axis="y")

# Ajout des annotations de pourcentage sur chaque barre
for bar1, bar2 in zip(bars1, bars2):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    percentage = (height2 / height1) * 100
    plt.text(bar1.get_x() + bar1.get_width() / 2, height1, f'{percentage:.1f}%', ha='center', va='bottom', color='black')

plt.legend(["Charge maximale", "Charge de base"])

plt.show()

#%% Baseload avoided or avoidable costs
"""Attention : remove noralization at the beginning to obtain this result"""

#old_baseload = base_load/(y/100 + 1)

energy_variation = yf-yi
#print(energy_variation)

HP_tarif = 8.44 #c/kWh
HC_tarif = 2.6 #c/kWh

HP_hours = 364*24 - (22-6)*52*5
HC_hours = 364*24 - HP_hours

#Economies per meter squared 
cost_variation = (energy_variation/1000)*(HP_tarif*HP_hours+HC_tarif*HC_hours)/100/2 #CHF/year
#dividing by 2 to make sure we do not atteibute more baseload to the orginal values than what there was 
cost_variation

#Normalihzation must not be performed when processing the data to have total variation results 
plt.figure()
plt.bar(range(1, len(Loads.columns) + 1),cost_variation, color="royalblue")
plt.xticks(ticks=range(1, len(Loads.columns) + 1), labels=Loads.columns, rotation=45)
plt.ylabel("cost variation [$CHF/year$]")
plt.grid(axis="y")
plt.show()

energy_variation*365*24/1000
#%% grading for comparison matrix - baseload trend score 

# # score pour variation annuelle 
# grades, classes, thresolds = f.get_score(x, y)

# score pour proportion de baseload 
grades, classes, thresholds = f.get_score(Loads.columns,baseload_ratio)

plt.figure()
for i, (k, v) in enumerate(classes.items()):
    print(type(v))
    if v == 1:
        plt.bar(i,v, color="darkgreen")
    elif v == 2:
        plt.bar(i,v, color="lightgreen")
    elif v == 3:
        plt.bar(i,v, color="yellow")
    elif v == 4:
        plt.bar(i,v, color="orange" )
    elif v == 5:
        plt.bar(i,v, color="red")
    else: 
        plt.bar(i,v, color="blue")
plt.grid(axis='y')
plt.xticks(range(len(x)), x, rotation=45)
plt.show()

print(f'Grades = {grades}')
print("++++++++++++++++++++++++++++")
print(f"thresholds = {thresholds}")
#%% past code 

"""
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

Typo_list = ["Ecole", "Culture", "Apems", "Commune", "Commune2", "Buvette", "Parking"]

#getting typologies from 2022
Typo_loads_2022, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2022_dict, Typo_list, True)

#getting typologies from 2023
Typo_loads_2023, _ = p.discriminate_typologies(Building_dict_2023, LoadCurve_2023_dict, Typo_list, True)

# creating overall dictionnary
Typo_all_loads = {}
for typo in Typo_list:
    Typo_all_loads[typo] = pd.concat([Typo_loads_2022[typo], Typo_loads_2023[typo]], axis=0)
    
#print(Typo_loads)
"""
#%%

"""
# smoothing calculations
Loads = Typo_all_loads[Typology]
Tendency = f.period_tendencies(Loads, Period)


#extracting 1 single load to compare with the benchmark and giving it the same smoothness 
single_load = Typo_all_loads[Typology].iloc[:, 0].to_frame()
#print(single_load)
smoothed_load = f.period_tendencies(single_load, Period)


# plotting 
updated_tendency = f.plot_mean_load(smoothed_load, Tendency, Period, Typology)
"""

#%% average weekly baseloads

"""
# Remove duplicate indices id needed
#df_no_duplicates = df[~df.index.duplicated(keep='first')]

baseloads = get_baseload_2(df)


# smoothing calculations
Loads = Typo_all_loads[Typology].iloc[:,:2]

num_rows = baseloads.shape[0]
chunk_size = 7
df = baseloads 

averages = []
# Iterate over the DataFrame in chunks of 96 rows
for i in range(0, num_rows, chunk_size):
    chunk = df.iloc[i:i+chunk_size]  # Get the current chunk of 96 rows
    chunk_avg = chunk.mean()  # Calculate the average for each column in the chunk
    averages.append(chunk_avg)  # Append the averages to the list
    print(i)
# Concatenate the averages into a single DataFrame
result = pd.concat(averages, axis=1).T


# Define your color palette
my_colors = sb.color_palette("hls", result.shape[1])
#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#palette = sns.color_palette("YlGnBu_r", result.shape[1])

plt.figure
plt.plot(result.head(53))
plt.plot(result.tail(53))
plt.grid()
plt.xlabel("weeks of the year")
plt.show()

plt.figure()

for i, column in enumerate(result.columns):
    plt.plot((result[column].head(53).values + result[column].tail(53).values) / 2, color=my_colors[i])

#plt.yscale('log')

plt.grid(which="both", alpha=0.5)
plt.xlabel("weeks of the year")
plt.ylabel("Baseload - [$kWh_{el}/m^2$]")
plt.title("Annual baseload variation - Schools")
plt.legend(result.columns, loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

"""
#%%% color plalettes
#my_colors = sb.color_palette("Spectral", result.shape[1])
#my_colors = sb.color_palette("magma", result.shape[1])

#my_colors = sb.color_palette("icefire", result.shape[1])
#my_colors = sb.color_palette("husl", result.shape[1])
#my_colors = sb.color_palette("rocket", result.shape[1])
#my_colors = sb.color_palette("viridis", result.shape[1])
#my_colors = sb.color_palette("mako", result.shape[1])
#my_colors = sb.color_palette("flare", result.shape[1])
#palette = sns.color_palette("YlGnBu_r", result.shape[1])
