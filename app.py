import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration and Data Loading ---

st.set_page_config(layout="wide", page_title="F1 Driver Performance Analysis")
st.title("F1 Driver Performance Analysis (PCA & Clustering)")

# Load the saved DataFrames
@st.cache_data # Cache data loading for performance
def load_data():
    try:
        df_driver_season = pd.read_csv('streamlit_data/df_driver_season.csv')
        df_pca_plot = pd.read_csv('streamlit_data/df_pca_plot.csv')
        return df_driver_season, df_pca_plot
    except FileNotFoundError:
        st.error("Data files not found! Please ensure 'streamlit_data' folder with 'df_driver_season.csv' and 'df_pca_plot.csv' is in the same directory as this script.")
        st.stop()

df_driver_season, df_pca_plot = load_data()

# --- Championship Years (Hardcoded for demonstration purposes) ---
championship_years = {
    'Lewis Hamilton': [2014, 2015, 2017, 2018, 2019, 2020],
    'Max Verstappen': [2021, 2022, 2023],
    'Sebastian Vettel': [], 
    'Fernando Alonso': [],
    'Nico Rosberg': [2016],
    'Jenson Button': [],
    'Charles Leclerc': [],
    'Carlos Sainz': [],
    # Add more drivers and their championship years if desired
}

# --- Driver Selection ---
driver_names = sorted(df_driver_season['driver_name'].unique().tolist())
selected_driver = st.sidebar.selectbox(
    "Select a Driver to Analyze:",
    driver_names
)

# Filter data for the selected driver
driver_stats = df_driver_season[df_driver_season['driver_name'] == selected_driver].sort_values('year')
driver_pca_data = df_pca_plot[df_pca_plot['driver_name'] == selected_driver].sort_values('year')

# --- Display Driver Statistics ---
st.header(f"Performance Statistics for {selected_driver}")
if not driver_stats.empty:
    st.dataframe(driver_stats.set_index('year'), use_container_width=True)
else:
    st.write("No detailed season statistics available for this driver in the dataset.")


# --- Plot PCA Trends ---
st.header(f"PCA Performance Trend for {selected_driver} Over Seasons")

if not driver_pca_data.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    fig.suptitle(f'PCA Performance Trend for {selected_driver} Over Seasons', fontsize=18)

    driver_championships = championship_years.get(selected_driver, [])

    # Plot PC1 trend
    sns.lineplot(data=driver_pca_data, x='year', y='PC1', marker='o', ax=axes[0], color='blue', linewidth=2, markersize=8)
    axes[0].set_title('Principal Component 1 (PC1) Trend', fontsize=14)
    axes[0].set_ylabel('PC1 Score', fontsize=12)
    axes[0].set_xlabel('Season Year', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Annotate championship years on PC1 plot
    for year in driver_championships:
        if year in driver_pca_data['year'].values:
            pc1_val = driver_pca_data[driver_pca_data['year'] == year]['PC1'].values[0]
            axes[0].annotate('🏆', (year, pc1_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=15, color='gold',
                             bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="k", lw=0.5))

    # Plot PC2 trend
    sns.lineplot(data=driver_pca_data, x='year', y='PC2', marker='o', ax=axes[1], color='red', linewidth=2, markersize=8)
    axes[1].set_title('Principal Component 2 (PC2) Trend', fontsize=14)
    axes[1].set_ylabel('PC2 Score', fontsize=12)
    axes[1].set_xlabel('Season Year', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Annotate championship years on PC2 plot
    for year in driver_championships:
        if year in driver_pca_data['year'].values:
            pc2_val = driver_pca_data[driver_pca_data['year'] == year]['PC2'].values[0]
            axes[1].annotate('🏆', (year, pc2_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=15, color='gold',
                             bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="k", lw=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
else:
    st.write("No PCA trend data available for this driver.")

