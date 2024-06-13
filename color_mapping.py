"""
This script loads and preprocesses data related to various metrics across different countries.
It uses custom color palettes for visualization and provides functions for creating color mappings,
color scales, and country abbreviation mappings.

Esther Fanyanàs i Ropero
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# Define a custom color palette
custom_palette2 = [
    "#a9a9a9", "#2f4f4f", "#556b2f", "#8b4513", "#191970",
    "#8b0000", "#808000", "#008000", "#3cb371", "#008b8b",
    "#4682b4", "#9acd32", "#00008b", "#32cd32", "#daa520",
    "#7f007f", "#8fbc8f", "#ff0000", "#ff8c00", "#ffd700",
    "#6a5acd", "#c71585", "#0000cd", "#00ff00", "#ba55d3",
    "#e9967a", "#dc143c", "#00ffff", "#00bfff", "#a020f0",
    "#adff2f", "#ff7f50", "#ff00ff", "#1e90ff", "#db7093",
    "#f0e68c", "#dda0dd", "#afeeee", "#98fb98", "#ffdab9", "#800080"
]

@st.cache_data
def load_data(filename:str) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file.

    :param filename: The path to the CSV file to be loaded.
    :return: The preprocessed data as a pandas DataFrame.
    """
    data = pd.read_csv(filename, sep=",", header=None, index_col=False)
    data.columns = data.iloc[0]
    data = data[1:]
    data.columns.name = None
    data.reset_index(drop=True, inplace=True)
    data['OBS_VALUE'] = pd.to_numeric(data['OBS_VALUE'], errors='coerce')
    return data

# Load data from CSV files and preprocess
df_teritary = load_data("data/teritary_educational_filter.txt")
df_teritary = df_teritary[df_teritary['geo'] != 'EU']
df_medical = load_data("data/medical_examination_filter.txt")
df_medical = df_medical[df_medical['geo'] != 'EU']
df_digital = load_data("data/basic_digital_noms.txt")
df_digital = df_digital[df_digital['geo'] != 'EU']

# Combine unique country names from all datasets
options_geo = np.union1d(df_teritary['name'].unique(), df_medical['name'].unique())
all_countries = np.union1d(options_geo, df_digital['name'].unique())

def create_color_mapping():
    """
    Create a mapping from country names to colors.

    Returns:
        dict: A dictionary mapping country names to colors.
    """
    color_mapping = {geo: color for geo, color in zip(all_countries, custom_palette2)}
    return color_mapping

def get_color_scale(selected_countries, color_mapping):
    """
    Create a color scale for the selected countries.

    Args:
        selected_countries (list): List of selected country names.
        color_mapping (dict): Dictionary mapping country names to colors.

    Returns:
        alt.Scale: An Altair color scale.
    """
    color_scale = alt.Scale(domain=selected_countries, range=[color_mapping[geo] for geo in selected_countries])
    return color_scale

def create_abbreviation_mapping():
    """
    Create a mapping from country names to their abbreviations.

    Returns:
        dict: A dictionary mapping country names to abbreviations.
    """
    return {
        "Albania": "AL", "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Czechia": "CZ",
        "Denmark": "DK", "Germany": "DE", "Estonia": "EE", "Ireland": "IE", "Greece": "EL",
        "Spain": "ES", "France": "FR", "Croatia": "HR", "Italy": "IT", "Cyprus": "CY",
        "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU", "Hungary": "HU", "Malta": "MT",
        "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovenia": "SI",
        "Slovakia": "SK", "Finland": "FI", "Sweden": "SE", "Iceland": "IS", "Norway": "NO",
        "Switzerland": "CH", "United Kingdom": "UK", "Montenegro": "ME", "North Macedonia": "MK",
        "Serbia": "RS", "Türkiye": "TR", "Bosnia and Herzegovina": "BA", "Kosovo": "XK"
    }
