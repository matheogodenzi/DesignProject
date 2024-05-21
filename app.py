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
    
    

if __name__ == "__main__":
    main()
