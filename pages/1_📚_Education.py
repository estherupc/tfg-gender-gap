"""
This section of the script focuses on visualizing gender-related metrics in education using Altair.

The script performs the following tasks:
1. Load and preprocess data related to education metrics.
2. Create color mappings and scales for visualization.
3. Generate interactive visualizations to display gender differences in educational attainment.

Esther Fanyanàs I Ropero
"""
# ------- IMPORTS -------
import streamlit as st
import pandas as pd
import altair as alt
from color_mapping import create_color_mapping, get_color_scale, all_countries, create_abbreviation_mapping

# ------ FUNCTIONS ------
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

st.set_page_config(page_title="Education", page_icon="📚", layout="wide",
initial_sidebar_state="expanded",)
st.title("📚 Education")
st.markdown("---")
st.sidebar.header("Education")
st.write(
    """Education is a key factor in achieving gender equality and empowering all individuals. In this section, we delve into the gender gaps in educational attainment, focusing on metrics such as early school leavers, tertiary education rates, adult participation in learning and participation in early childhood education."""
)
st.markdown("---")

col1, col2 = st.columns([1, 2])

alt.data_transformers.disable_max_rows()


st.markdown("### Population aged 25-34 who have successfully completed tertiary studies")

color_mapping = create_color_mapping()
abbreviation_mapping = create_abbreviation_mapping()

with st.expander('## Legend of Region Colors'):
    cols = st.columns(4)  # Adjust the number of columns as needed
    color_items = list(color_mapping.items())
    num_items = len(color_items)
    items_per_col = 10  # Distribute items across columns

    for idx, (region, color) in enumerate(color_items):
        col_idx = idx // items_per_col
        with cols[col_idx]:
            abbreviation = abbreviation_mapping.get(region, "")
            st.markdown(f'{region} ({abbreviation}): ![{color}](https://via.placeholder.com/15/{color.strip("#")}/000000?text=+)')

df_teritary = load_data("data/teritary_educational.txt")
# Define dataframes and selections
df_reference = df_teritary[(df_teritary['geo'] == 'EU') & (df_teritary['sex'] == 'T')]
options_geo = [geo for geo in df_teritary['name'].unique() if geo not in ["EU"]]
options_geo.sort()
df_teritary = df_teritary[df_teritary['sex'].isin(['F', 'M'])]

nearest = alt.selection_point(nearest= True, on= 'mouseover', fields= ['TIME_PERIOD'])

color_mapping = create_color_mapping()

cols = st.columns([5, 1])

# Widget de multiselect
with cols[0]:
    selected_countries = st.multiselect(
        'Select regions:',
        options= options_geo,  # unique options from the DataFrame
        default=['Germany', 'Netherlands']
    )

filtered_data = df_teritary[df_teritary['name'].isin(selected_countries)]

color_scale = get_color_scale(selected_countries, color_mapping)

# Calculate the reference value for EU
df_reference['OBS_VALUE'] = pd.to_numeric(df_reference['OBS_VALUE'], errors='coerce')
df_reference['TIME_PERIOD'] = pd.to_numeric(df_reference['TIME_PERIOD'], errors='coerce')
df_teritary['TIME_PERIOD'] = pd.to_numeric(df_teritary['TIME_PERIOD'], errors='coerce')

# Now try calculating the average again
df_teritary = df_teritary.merge(df_reference[['TIME_PERIOD', 'OBS_VALUE']], on='TIME_PERIOD', suffixes=('', '_EU'))
df_teritary['Difference_from_EU'] = df_teritary['OBS_VALUE'] - df_teritary['OBS_VALUE_EU']
df_teritary['Difference_from_EU']= df_teritary['Difference_from_EU'].round(2)

# Create an Altair chart for average lines
mean_lines = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule().encode(y='y:Q')

# Update filtered data to include new difference column
filtered_data = df_teritary[
    (df_teritary['name'].isin(selected_countries)) &
    (df_teritary['name'] != 'EU')
]

filtered_data['sex'] = filtered_data['sex'].replace({'M': 'Male', 'F': 'Female'})
all_years = sorted(df_teritary['TIME_PERIOD'].unique())

# Create the base chart showing the difference instead of the OBS_VALUE values
base_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
    x=alt.X('TIME_PERIOD:O', title= 'Year', axis=alt.Axis(labelAngle=0), scale=alt.Scale(domain=all_years)),
    y=alt.Y('Difference_from_EU:Q', title='% Difference from European Union average', scale=alt.Scale(domain=(df_teritary['Difference_from_EU'].min(), df_teritary['Difference_from_EU'].max()))),
    color=alt.Color('name:N', scale=color_scale, title= 'Region'),
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
)

# Define the lines for each gender
lines_F = base_chart.transform_filter(
    alt.datum.sex == 'Female'
).encode(
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
    opacity=alt.value(0.5)
)

lines_M = base_chart.transform_filter(
    alt.datum.sex == 'Male'
).encode(
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
    opacity=alt.value(0.5)
)

# Define the interactive elements
selectors = alt.Chart(filtered_data).mark_point(color = 'gray').encode(
    x='TIME_PERIOD:O',
    opacity = alt.value(0),
    tooltip=[alt.Tooltip('TIME_PERIOD:O', title='Year')]
).add_params(
    nearest
)

rules = alt.Chart(filtered_data).mark_rule(color='gray').encode(
    x='TIME_PERIOD:O',
    tooltip=[alt.Tooltip('TIME_PERIOD:O', title='Year')]
).transform_filter(
    nearest
)

points = base_chart.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0)),  # Mostrar solo cuando hay una selección
    tooltip=[alt.Tooltip('TIME_PERIOD:O', title='Year'), alt.Tooltip('Difference_from_EU:Q', title='% Difference from EU'), alt.Tooltip('sex:N', title='Sex'), alt.Tooltip('name:N', title='Region')]
)

# Text displayed around the selected point
text = base_chart.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'Difference_from_EU:Q', alt.value(' ')),  # Mostrar solo cuando hay una selección
)

annotations = alt.Chart(pd.DataFrame({
    'y': [0, -2500],
    'text': ['Above EU Avg', 'Below EU Avg']
})).mark_text(
    align='right',
    baseline='middle',
    dx=60,
    dy=10,
    color='darkgray',
    fontSize = 12
).encode(
    x=alt.value(-100),  # To position the text on the left side of the graphic
    y=alt.Y('y:Q', axis=None),
    text='text:N'
)

# Merge all layers
ch = alt.layer(
    lines_F, lines_M, mean_lines, selectors, rules, points, text,
).properties(
    width=700,
    height=500,
    title= 'Yearly % Difference from EU27 Average by Region and Gender',
)

ch = alt.layer(
    ch, annotations
).resolve_scale(y='independent')


st.altair_chart(ch, use_container_width=True)

st.markdown("### Population aged 18 to 24 not involved in education or training")

url_geojson = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson"
data_url_geojson = alt.Data(url=url_geojson, format=alt.DataFormat(property="features"))

df_nini = load_data("data/nini_map.txt")

# Define dataframes and selections
df_reference = df_nini[df_nini['name'] == 'EU']
options_geo = df_nini['name'].unique().tolist()
df_nini = df_nini[df_nini['sex'].isin(['F', 'M'])]
df_nini['TIME_PERIOD'] = df_nini['TIME_PERIOD'].astype(int)
df_nini = df_nini[df_nini['sex'].isin(['F', 'M'])]

min_year = int("2009")
max_year = int(df_nini['TIME_PERIOD'].max())
all_years = sorted(df_nini['TIME_PERIOD'].unique())

# Group by ‘Year’, ‘geo’ and calculate the difference between M and F
df_diff = df_nini.pivot_table(index=['TIME_PERIOD', 'name'], columns='sex', values='OBS_VALUE', aggfunc='sum')
df_diff['Difference'] = df_diff['M'] -df_diff['F']

# Reset the index to make ‘Year’ and ‘geo’ columns regular again.
df_diff = df_diff.reset_index()

cols = st.columns([3, 2.5])

with cols[0]:
# Widget to select the year using a slider
    year = st.slider('Select a year:', min_value=min_year, max_value=max_year, value=min_year, step=1)

filtered_data = df_diff[df_diff['TIME_PERIOD'] == year]
all_countries_df = pd.DataFrame({
    'name': all_countries,
    'TIME_PERIOD': year
})
all_countries_df['name'] = all_countries_df['name'].replace({'Türkiye': 'Turkey', 'North Macedonia': 'The former Yugoslav Republic of Macedonia', 'Czechia': 'Czech Republic'})
combined_data = pd.merge(all_countries_df, filtered_data, on=['name', 'TIME_PERIOD'], how='left')

placeholder_value = -999
combined_data['Difference'].fillna(placeholder_value, inplace=True)
combined_data['tooltip'] = combined_data.apply(
    lambda row: 'No Data' if row['Difference'] == -999 else f"{row['Difference']:.2f}", axis=1
)

color_scale = alt.Scale(domain=[-20, -16,-12, -8, -4, 0, 4, 8, 12, 16, 20],  # Adjust the domain according to your data
                        range=['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',  '#f7f7f7',  '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'])

base = alt.Chart(data_url_geojson).mark_geoshape(
    stroke='black',
    strokeWidth=1
).encode(
    color=alt.condition(
        alt.datum.Difference != -999,
        alt.Color('Difference:Q', scale=color_scale, legend=alt.Legend(
            title="Difference (%)",
            titleFontSize=14,
            titlePadding=10,
            labelFontSize=12,
            labelPadding=10,
            orient='right',
            direction='vertical',
            gradientLength=200,
            gradientThickness=20,
            gradientStrokeWidth=0.5,
            tickCount=3,  # Limit the number of ticks
            labelExpr="datum.value == -20 ? ' More Female' : datum.value == 0 ? ' Equal' : datum.value == 20 ? ' More Male' : ''"
            )),
        alt.value('lightgray')  # Countries with no data in the year selected in grey
    ),
    tooltip=[
    alt.Tooltip('properties.NAME:N', title='Region'),
    alt.Tooltip('tooltip:N', title='Difference (%)')
]
).transform_lookup(
    lookup='properties.NAME',  # Set this to the right property in your GeoJSON
    from_=alt.LookupData(combined_data, 'name', ['Difference', 'TIME_PERIOD', 'tooltip'])
).properties(
    width=1000,
    height=700,
    title=f'Difference (Male - Female) in Early Leavers from Education and Training in {year}'
)
st.altair_chart(base, use_container_width=True)

st.markdown("### People aged 16 to 74 who have at least basic digital skills")

color_mapping = create_color_mapping()
abbreviation_mapping = create_abbreviation_mapping()

with st.expander('## Legend of Region Colors'):
    cols = st.columns(4)  # Adjust the number of columns as needed
    color_items = list(color_mapping.items())
    num_items = len(color_items)
    items_per_col = 10  # Distribute items across columns

    for idx, (region, color) in enumerate(color_items):
        col_idx = idx // items_per_col
        with cols[col_idx]:
            abbreviation = abbreviation_mapping.get(region, "")
            st.markdown(f'{region} ({abbreviation}): ![{color}](https://via.placeholder.com/15/{color.strip("#")}/000000?text=+)')

df_digital = load_data("data/basic_digital_filter.txt")

country_counts = df_digital.groupby('geo').size()

# Filter out countries that have data for only one year
countries_with_multiple_years = country_counts[country_counts > 3].index

df_digital = df_digital[df_digital['geo'].isin(countries_with_multiple_years)]

# Define dataframes and selections
df_reference = df_digital[df_digital['sex'].isin(['T'])]
options_geo = df_digital['name'].unique().tolist()
options_geo.sort()
df_digital = df_digital[df_digital['sex'].isin(['F', 'M'])]

nearest = alt.selection_point(nearest= True, on= 'mouseover', fields= ['TIME_PERIOD'])

cols = st.columns([5, 1])
# Multiselect widget
with cols[0]:
    selected_countries = st.multiselect(
        'Select regions:',
        options= options_geo,  # unique options from the DataFrame
        default=['Germany', 'Netherlands']
    )

filtered_data = df_digital[df_digital['name'].isin(selected_countries)]
color_scale = alt.Scale(domain=selected_countries, range=[color_mapping[geo] for geo in selected_countries])

# Calculate the reference value for EU
df_digital['TIME_PERIOD'] = pd.to_numeric(df_digital['TIME_PERIOD'], errors='coerce')
df_digital['sex'] = df_digital['sex'].replace({'M': 'Male', 'F': 'Female'})

# Update filtered data to include new difference column
filtered_data = df_digital[
    (df_digital['name'].isin(selected_countries)) &
    (df_digital['name'] != 'EU')
]

# Create the base chart showing the difference instead of the OBS_VALUE values
base_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
    x=alt.X('TIME_PERIOD:O', scale=alt.Scale(domain=[2021, 2023]), title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('OBS_VALUE:Q', scale=alt.Scale(domain=[df_digital['OBS_VALUE'].min(), df_digital['OBS_VALUE'].max()]), title='Percentage'),
    color=alt.Color('name:N', scale=color_scale, title='Region'),
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
)

# Define the lines for each gender
lines_F = base_chart.transform_filter(
    alt.datum.sex == 'Female'
).encode(
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
    opacity=alt.value(0.5)
)

lines_M = base_chart.transform_filter(
    alt.datum.sex == 'Male'
).encode(
    strokeDash=alt.StrokeDash('sex:N', legend=alt.Legend(title='Sex'), sort='descending'),
    opacity=alt.value(0.5)
)

# Define the interactive elements
selectors = alt.Chart(filtered_data).mark_point(color = 'gray').encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    opacity = alt.value(0)
).add_params(
    nearest
)

rules = alt.Chart(filtered_data).mark_rule(color='gray').encode(
    x=alt.X('TIME_PERIOD:O', axis=alt.Axis(labelAngle=0)),
)

points = base_chart.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0)),  # Show only when there is a selection
    tooltip=[alt.Tooltip('TIME_PERIOD:O', title = 'Year'), alt.Tooltip('OBS_VALUE:Q', title='Percentage'), alt.Tooltip('sex:N', title='Sex'), alt.Tooltip('name:N', title='Region')]
)

# Text displayed around the selected point
text = base_chart.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'OBS_VALUE:Q', alt.value(' ')),  # Show only when there is a selection
)

# Combine all layers
ch = alt.layer(
    lines_F, lines_M, selectors, rules, points, text
).properties(
    width=600,
    height=500,
    title = 'Trends in Digital Skills by Region and Gender (2021-2023)'
)
st.altair_chart(ch, use_container_width=True)

st.markdown("### Participation in early childhood education")

df_childhood = load_data("data/childhood_noms.txt")
df_childhood['TIME_PERIOD'] = pd.to_numeric(df_childhood['TIME_PERIOD'], errors='coerce')
cols = st.columns([1, 4])
unique_regions = df_childhood['name'].unique()
unique_regions.sort()
with cols[0]:
    # Country selection in Streamlit
    selected_country = st.selectbox('Select a region:', unique_regions)
df_childhood['sex'] = df_childhood['sex'].replace({'M': 'Male', 'F': 'Female'})
df_filtered = df_childhood[df_childhood['name'] == selected_country]

# Define fixed domains
x_domain = df_childhood['TIME_PERIOD'].unique()
y_domain = [0, df_childhood['OBS_VALUE'].max()]

color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

# Bar chart for the female population
bars = alt.Chart(df_filtered[df_filtered['sex'] == 'Female']).mark_bar().encode(
    x=alt.X('TIME_PERIOD:O', title='Year', scale=alt.Scale(domain=x_domain), axis=alt.Axis(labelAngle=0)),
    y=alt.Y('OBS_VALUE:Q', title='% of Population Aged 3 to Primary School Start', scale=alt.Scale(domain=y_domain)),
    color=alt.Color('sex:N', scale=color_sex, legend=alt.Legend(title="Sex")),
    tooltip=[ alt.Tooltip('OBS_VALUE:Q', title='Female Population'), alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
)

# Line graph for the male population
line = alt.Chart(df_filtered[df_filtered['sex'] == 'Male']).mark_line(point=True).encode(
    x=alt.X('TIME_PERIOD:O', title='Year', scale=alt.Scale(domain=x_domain)),
    y=alt.Y('OBS_VALUE:Q'),
    color=alt.Color('sex:N', scale=color_sex, legend=None),
    tooltip=[ alt.Tooltip('OBS_VALUE:Q', title='Male Population'), alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
)

# Combines both graphics
combined_chart = alt.layer(
    bars,
    line
).resolve_scale(
    y='shared'
).properties(
    width=800,
    height=400,
    title=''
).configure_point(
    size=100
).configure_axis(
    titleFontSize=14,
    labelFontSize=12
)

# Show the graph in Streamlit
st.altair_chart(combined_chart, use_container_width=True)