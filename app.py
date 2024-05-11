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
    st.title("Yearly Electric Load Series Plot")
    st.sidebar.header("Input Data")
    
    # Allow user to upload a CSV file containing yearly electric load data
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["xlsx"])
    
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        df_new = f.get_mean_load_kW(df)
        # Display the DataFrame
        st.subheader("Input Data:")
        display_table(df_new)
        
        # Plot the electric load series
        plot_load_series(df_new)
    else:
        st.info("Please upload a CSV file.")

def display_table(df):
    # Display the DataFrame with adjusted table width
    st.markdown(
        f"""
        <style>
            .dataframe {{ width: 800px; }}
        </style>
        """, unsafe_allow_html=True
    )
    st.dataframe(df, height=300,  width=1000)

def plot_load_series(df):
    # Assume the DataFrame has a 'Year' column and a 'Electric Load' column
    #df = df.set_index('Date')
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Charge'], marker='o', linestyle='-')
    plt.title('Yearly Electric Load Series')
    plt.xlabel('Year')
    plt.ylabel('Electric Load')
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.subheader("Electric Load Plot:")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
