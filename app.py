# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:19:53 2024

@author: matheo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
from scipy.stats import shapiro
import seaborn as sb
from sklearn.linear_model import LinearRegression

import functions as f
import controls as c
import auto_analysis as aa
import process_data as p


def main():
    st.title("Analyse de votre consommation électrique annuelle")
    st.sidebar.header("Données d'entrée")
    
    # Allow user to upload a CSV file containing yearly electric load data
    uploaded_file = st.sidebar.file_uploader("Ajoutez un fichier de courbe de charge (format xlsx)", type=["xlsx"])
    building_useful_area = st.number_input("Entrez votre surface utile (en $m^2$)", min_value=0, max_value=100000, value=1000)
    
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        df_mean_load = f.get_mean_load_kW(df) #kW/m2
        # Assuming df is your DataFrame
        df_mean_load = df_mean_load.rename(columns={'Charge': 'Charge [kW]'})

        # Display the DataFrame
        st.subheader("Charge journalière moyenne:")
        st.dataframe(df_mean_load, height=300,  width=1000)
        
        # Plot the electric load series
        plot_load_series(df_mean_load, building_useful_area)
    
        #Plot mean load tendency
        plot_mean_load_trend(df)
    
    else:
        st.info("Pour commencer l'analyse, ajoutez un fichier .xlsx.")

def plot_load_series(df, area):
    # Assume the DataFrame has a 'Year' column and a 'Electric Load' column
    #df = df.set_index('Date')
    normalized_load = df['Charge [kW]'].copy()/area
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, normalized_load)
    plt.title('Charge électrique moyenne annuelle')
    plt.xlabel("Jours de l'année")
    plt.ylabel('Charge électrique [$kW/m^2$]')
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.subheader("Charge journalière moyenne par unité de surface:")
    st.pyplot(plt)
    
def plot_mean_load_trend(df): 
    
    my_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']

    df_nan = df.replace(0, np.nan)
    Daily_average_load = get_mean_load_kW(df_nan, "day") #kWel
    df  = Daily_average_load.copy()

    # Create subplots
    fig, ax = plt.subplots(figsize=(6, 5))
    coef_df =  pd.DataFrame({'slope': [], 'y-intercept': []})
    relative_slope = []
    
    # Perform linear regression and plot for each column
    for i, column in enumerate(df.columns):
            
                #plt.ylim(-6, 10)
                
                # Replace 0 values with NaN
                infra = df[column].copy()
                # Convert zeros to NaN
                infra.replace(0, np.nan, inplace=True)
                
                # Drop NaN values
                infra.dropna(inplace=True)
        
                X = np.array(infra.index).reshape(-1, 1)   # Independent variable
                #print(X)
                y = infra.values.reshape(-1, 1)              # Dependent variable
                
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
                # Set labels and title
                ax.set_title(f'{column}')
                ax.set_xlabel('Jours')
                ax.set_ylabel('Charge moyenne [$kW_{el}$]')
                ax.legend()
                
    
    #plt.ylim(0, 3e-12)
    #plt.yscale("log")
    plt.title("Profil d'évolution de la consommation")
    # Place legend outside the plot
    plt.legend(title="Clients", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(which='both')
    #plt.tight_layout()
    # Display the plot in Streamlit
    st.subheader("Tendance de la charge moyenne:")
    st.pyplot(plt)
    
    coef_df.index = df.columns
    
    return coef_df, relative_slope
    
def get_mean_load_kW(df, period="week"):
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
    if period == "week":
        chunk_size = 96*7
    if period == "day":
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

if __name__ == "__main__":
    main()
