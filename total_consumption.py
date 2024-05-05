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
from sklearn.linear_model import LinearRegression
"""functions imports"""

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


#True> total load, if False > only SIE load (without PV)
LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict = p.get_load_curves(False)

#%% get all typologies sorted for all provided year 

# if True > normalized load, if False > absolute load 
Typo_loads_2022, Typo_loads_2023, Typo_all_loads, Correspondance = p.sort_typologies(LoadCurve_2023_dict, LoadCurve_2022_dict, Building_dict_2023, pv_2022_dict, True)

#%% grading for comparison matrix - overload score 

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
        elif grades[typology_names[i]] <= 101 :
            classes[typology_names[i]] = 5
    
    thresholds = [v/100*(max_-min_) + min_ for v in [0, 20, 40, 60, 80 , 100]]


    
    return grades, classes, thresholds 


#%%
def get_mean_load_kW(df):
    """
    Delineate annual tendencies over days, weeks, and months and returns mean load in kW

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
    #if you want it per week you multiply the chunk_size by seven otherwise you keep 96 values per day
    chunk_size = 96
    for i in range(0, num_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get the current chunk of 96 rows
        chunk_kW = 4*chunk
        # Calculate the mean of the smallest values for each column
        average_of_smallest = chunk_kW.mean()
        
        averages.append(average_of_smallest)  # Append the averages to the list
    
    # Concatenate the averages into a single DataFrame
    result = pd.concat(averages, axis=1).T
    
    return result


#%% creating a benchmark over available years

# parameters to change
Typology = "Ecole"
Period = "day"

# smoothing calculations
Loads = Typo_all_loads[Typology]

# Obtain a typical year averaged
typical_year = f.typical_period(Loads,  "year")


Loads_2022 = Typo_loads_2022[Typology]
Loads_2023 = Typo_loads_2023[Typology]

# Replace zeros with NaN values
"""If you want to have both years instead of their average, change typical_year by Loads"""
df_nan = typical_year.replace(0, np.nan)

Daily_average_load = get_mean_load_kW(df_nan)

my_colors = sb.color_palette("hls", Daily_average_load.shape[1])

plt.figure()

for i in range(Daily_average_load.shape[1]):
    #if i in [0, 3, 7, 11]:
    #if i in [1, 4, 8, 12]:
    #if i in [2, 5, 6, 9, 10]:
    #if i in [6,12]:
        plt.plot(Daily_average_load.iloc[:,i], c=my_colors[i], label=Daily_average_load.columns[i])
plt.plot(Daily_average_load.mean(1), color="royalblue", label="Profil moyen", linewidth=7, alpha=0.5)
#plt.yscale("log")
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Etablissements")
plt.title("Consommation habdomadaire à travers l'année - écoles")
plt.xlabel("Semaines de l'année")
plt.ylabel("Charge moyenne [$KW_{el}/m^2$]")
plt.show()

#%% boxplot conso journalière moyenne sur une année

df = Daily_average_load

# Calculate mean for each column
means = df.mean()

# Plot boxplot
plt.figure()
boxplot = df.boxplot()
plt.scatter(range(1, len(df.columns) + 1), means, color='red', label='Mean', zorder=3, s=10)
plt.xticks(ticks=range(1, len(df.columns) + 1), labels=df.columns, rotation=45)
plt.xlabel("Identifiants des consommateurs")
plt.ylabel("Charge [$kW_{el}$]")
plt.title("Distribution annuelle de la charge journalière - Ecoles")
plt.grid(axis="x")

# Extracting the boxplot elements for creating legend
boxes = [item for item in boxplot.findobj(match=plt.Line2D)][::6]  # boxes
medians = [item for item in boxplot.findobj(match=plt.Line2D)][5::6]  # medians
whiskers = [item for item in boxplot.findobj(match=plt.Line2D)][2::6]  # whiskers
caps = [item for item in boxplot.findobj(match=plt.Line2D)][3::6]  # caps

# Create legend with labels
plt.legend([medians[0], caps[0], plt.Line2D([], [], color='red', marker='o', linestyle='None')], 
           [ 'Mediane', 'Bornes', 'Moyenne'])

plt.show()


#%% Linear regressions 

# Replace zeros with NaN values
"""If you want to have both years instead of their average, change typical_year by Loads"""
df_nan = Loads.replace(0, np.nan)

Daily_average_load = get_mean_load_kW(df_nan)

df  = Daily_average_load

# Define your color palette
palette = sb.color_palette("hls", df.shape[1])
# Create an iterator to cycle through the colors
#color_iterator = cycle(palette)
# Create subplots
fig, ax = plt.subplots(figsize=(8, 5))

coef_df =  pd.DataFrame({'slope': [], 'y-intercept': []})
relative_slope = []

# Perform linear regression and plot for each column
for i, column in enumerate(df.columns):
    #if i in [0, 1, 4, 7, 9]: #low-level
            #plt.ylim(0.0002, 0.0012)
    #if i in [5, 8, 10, 11, 3]: #medium level
            #plt.ylim(0.0002, 0.002)
    #if i in [2, 6,12]:
        
            #plt.ylim(0.0005, 0.004)
            
            # Replace 0 values with NaN
            infra = df[column].copy()
            # Convert zeros to NaN
            infra.replace(0, np.nan, inplace=True)
            
            # Drop NaN values
            infra.dropna(inplace=True)
    
            X = np.array(infra.index).reshape(-1, 1)   # Independent variable
            #print(X)
            y = 4*infra.values.reshape(-1, 1)              # Dependent variable
            
            # Plot data points
            #ax.scatter(X, y, color=palette[i], alpha=0.3, s=10)
            
            
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
            ax.plot(X, y_reg, c=palette[i], label=column, linewidth=2, alpha = 1)
            
            relative_slope.append(coefficients[0][0]/y_reg[0][0])
            # Set labels and title
            ax.set_title(f'{column}')
            ax.set_xlabel('Jours')
            ax.set_ylabel('Charge moyenne [$kW_{el}/m^2$]')
            ax.legend()

#plt.ylim(0, 3e-12)
#plt.yscale("log")
plt.title("Profile d'évolution de la consommation - écoles")
# Place legend outside the plot
plt.legend(title="Etablissements", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(which='both')
#plt.tight_layout()
plt.show()

coef_df.index = df.columns

#%% plotting scores 

#y for trends 
#y = np.array(relative_slope)*100*365

#y for mean values 

df_nan = typical_year.replace(0, np.nan)
Daily_average_load = get_mean_load_kW(df_nan)
Dailymeans = Daily_average_load.mean()

y = Dailymeans.values
print(y)
x= coef_df.index

grades, classes, thresholds = get_score(x, y)
print("*******************")
print(classes)
print("*******************")
print(grades)
print("*******************")
print(thresholds)
print("*******************")
plt.figure()
plt.bar(x,y)
plt.show()

plt.figure()
for i, (k, v) in enumerate(classes.items()):
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
    else:
        plt.bar(i, v, color="blue")
plt.grid(axis='y')
plt.xticks(range(len(x)), x)
plt.show()
#%% Previous code 
"""
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
plt.xticks(range(len(mean_values)), Loads.columns.tolist(), rotation=45)
#plt.yscale("log")
plt.ylabel("load [$kW_{el}/m^2$]")
plt.xlabel("Consumers' IDs")
plt.title("Overall average load per meter squared")
plt.grid(axis="y")
"""

#%% calculating mean and standard deviation for a typical day the year 
"""

# parameters to change
Typology = "Ecole"
Period = "week"

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
#%% average load tendency 
"""

two_years_load = Typo_all_loads[Typology]

# Assuming df is your DataFrame
# Replace zeros with NaN values
df_nan = two_years_load.replace(0, np.nan)

Daily_average_load = get_mean_load_kW(df_nan)

plt.figure()
for i in range(Daily_average_load.shape[1]):
    plt.scatter(Daily_average_load.index.to_list(), Daily_average_load.iloc[:,i].values, s=5, alpha=0.3)
plt.show()
"""