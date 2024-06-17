"""
This section of the script focuses on visualizing gender-related metrics in well-being using Altair.

The script performs the following tasks:
1. Load and preprocess data related to education metrics.
2. Create color mappings and scales for visualization.
3. Generate interactive visualizations to display gender differences in well-being attainment.

Esther Fanyanàs I Ropero
"""
# ------- IMPORTS -------
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go

st.set_page_config(page_title="Well-being", page_icon="🏥", layout="wide",
initial_sidebar_state="expanded")

st.title("🏥 Well-being")
st.markdown("---")
st.write(
    """Well-being is a multifaceted concept that encompasses physical, mental, and social health. This section presents data on indicators such as healthy life expectancy, self-reported unmet need for medical examination and care, self-reported well-being, fatal accidents at work and standardised death rate due to homicide. """
)

# ----- FUNCTIONS -------
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

st.markdown("---")
st.markdown("### Healthy life years at birth and Perceived good health")

cols = st.columns([1,1])
with cols[0]:
    df_healthy_life = load_data("data/healthy_life_noms.txt")
    df_healthy_life = df_healthy_life[df_healthy_life['sex'].isin(['F', 'M'])]
    df_healthy_life['sex'] = df_healthy_life['sex'].replace({'M': 'Male', 'F': 'Female'})

    df_health_good = load_data("data/health_good_noms.txt")
    df_health_good = df_health_good[df_health_good['sex'].isin(['F', 'M'])]
    df_health_good['sex'] = df_health_good['sex'].replace({'M': 'Male', 'F': 'Female'})
    df_health_good['Percentage_Display'] = df_health_good['OBS_VALUE'].apply(
        lambda x: f"{x:.2f}%"
    )
    common = np.intersect1d(df_healthy_life['name'].unique(), df_health_good['name'].unique())
    common.sort()

    col_reg = st.columns([1, 2.3])
    with col_reg[0]:
        # Country selection in Streamlit
        selected_geo = st.selectbox('Select a region:', common)

    # Combining DataFrames
    df_health_good['TIME_PERIOD'] = df_health_good['TIME_PERIOD'].astype(int)
    df_healthy_life['TIME_PERIOD'] = df_healthy_life['TIME_PERIOD'].astype(int)
    df_combined = pd.merge(df_healthy_life, df_health_good, on=['geo', 'name', 'TIME_PERIOD', 'sex'], suffixes=('_Life', '_Health'))

    # Filter data by geography selection
    df_filtered = df_combined[df_combined['name'] == selected_geo]
    min_value_life = df_combined ['OBS_VALUE_Life'].min()  # Slightly less than the minimum
    max_value_life = df_combined ['OBS_VALUE_Life'].max() # Slightly more than the maximum

    min_value_health = df_combined ['OBS_VALUE_Health'].min()
    max_value_health = df_combined ['OBS_VALUE_Health'].max()
    color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

    scatter_plot = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('OBS_VALUE_Life:Q', title='Healthy life years', scale=alt.Scale(domain=(min_value_life, max_value_life))),
        y=alt.Y('OBS_VALUE_Health:Q', title='Percentage of Percived Good Health', scale=alt.Scale(domain=(min_value_health, max_value_health))),
        color=alt.Color('sex:N', scale = color_sex, legend=alt.Legend(title="Sex")),
        tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year'), alt.Tooltip('OBS_VALUE_Life:Q', title='Healthy Life Years'), alt.Tooltip('Percentage_Display:N', title='Perceived Good Health')]
    ).properties(
        width=400,
        height=400,
        title='Correlation between Healthy Life Years at Birth and Perceived Health by Sex'
    )
    # Show the graph in Streamlit
    st.altair_chart(scatter_plot, use_container_width=True)

with cols[1]:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    chart = alt.Chart(df_health_good).mark_circle().encode(
        x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0, grid=True)),
        y=alt.Y('mean(OBS_VALUE)', title='Percentage of Percived Good Health'),
        color=alt.Color('sex:N', scale = color_sex, legend=alt.Legend(title="Sex")),
        tooltip=[alt.Tooltip('mean(OBS_VALUE):Q', title='% Regions\' Good Health', format='.2f'), alt.Tooltip('TIME_PERIOD:O', title='Year'), ]
    ).properties(
        width=400,
        height=400,
        title='Percentage of Regions\' Mean Perceived Good Health by Year and Sex'
    )
    # Show the graph in Streamlit
    st.altair_chart(chart, use_container_width=True)

st.markdown("### Smoking prevalence")
st.markdown(
    "<h6 style='text-align: center; color: #333333;'>Smoking prevalence by Year per Sex</h6>",
    unsafe_allow_html=True
)

df_smoking = load_data("data/smoking.txt")
df_smoking = df_smoking[df_smoking['sex'].isin(['F', 'M'])]
df_smoking['sex'] = df_smoking['sex'].replace({'M': 'Male', 'F': 'Female'})

pointpos_male = np.linspace(-1, -0.2, 18)
pointpos_female = np.linspace(0.2, 1, 18)
show_legend = [True] + [False] * 17  # Display the legend only once

fig = go.Figure()

unique_years_with_data = pd.unique(df_smoking['TIME_PERIOD'])

for i in range(len(unique_years_with_data)):
    fig.add_trace(go.Violin(x=df_smoking['TIME_PERIOD'][(df_smoking['sex'] == 'Female') &
                                        (df_smoking['TIME_PERIOD'] == pd.unique(df_smoking['TIME_PERIOD'])[i])],
                            y=df_smoking['OBS_VALUE'][(df_smoking['sex'] == 'Female')&
                                               (df_smoking['TIME_PERIOD'] == pd.unique(df_smoking['TIME_PERIOD'])[i])],
                            legendgroup='Female', scalegroup='Female', name='Female',
                            side='positive',
                            pointpos=pointpos_female[i],
                            line_color='#af8dc3',
                            showlegend=show_legend[i])
             )
    fig.add_trace(go.Violin(x=df_smoking['TIME_PERIOD'][(df_smoking['sex'] == 'Male') &
                                        (df_smoking['TIME_PERIOD'] == pd.unique(df_smoking['TIME_PERIOD'])[i])],
                            y=df_smoking['OBS_VALUE'][(df_smoking['sex'] == 'Male')&
                                               (df_smoking['TIME_PERIOD'] == pd.unique(df_smoking['TIME_PERIOD'])[i])],
                            legendgroup='Male', scalegroup='Male', name='Male',
                            side='negative',
                            pointpos=pointpos_male[i], # where to position points
                            line_color='#7fbf7b',
                            showlegend=show_legend[i])
             )

# Update characteristics shared by all traces
fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.05,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count
fig.update_layout(
    title_text= "",
    violingap=0.1,  # Increasing the space between violin graphics
    violingroupgap=0.48,  # Increasing the spacing between the violin graphic groups
    violinmode='overlay',
    width=1100,  # Adjust chart width
    height=700,  # Adjust chart height
    xaxis=dict(
        title='Year',
        tickangle=0,  # Rotate x-axis labels
        dtick=1,
        automargin=True  # Automatically adjust margins
    ),
    yaxis=dict(
        title='Percentage of total population',  # Label for the Y-axis
        automargin=True  # Adjust margins automatically
    ),
    margin=dict(t=0),
    legend=dict(
    title=dict(
            text="Sex",  # Title of the legend
            font=dict(
                size=14,
                color="grey"  # Text colour of the caption title
            ),
        ),
        font=dict(
            size=14,
            color="grey",  # Change the colour of the legend text to grey
        )
    )
    )
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Fatal accidents at work")

df_accidents = load_data("data/accidents_noms.txt")
df_accidents['TIME_PERIOD'] = pd.to_numeric(df_accidents['TIME_PERIOD'], errors='coerce')
df_accidents['sex'] = df_accidents['sex'].replace({'M': 'Male', 'F': 'Female'})

cols = st.columns([1.5,0.01,1])
# Allows the user to select two years for comparison
years = df_accidents['TIME_PERIOD'].unique()
with cols[0]:
    year1, year2 = st.select_slider(
        'Select two years to compare:',
        options=sorted(years),
        value=(years.min(), years.max())
    )

    # Filter data to include only selected years
    df_filtered = df_accidents[df_accidents['TIME_PERIOD'].isin([year1, year2])]

    df_female = df_filtered[df_filtered['sex'] == 'Female']
    df_male = df_filtered[df_filtered['sex'] == 'Male']

    # Pivot data using pivot_table to handle duplicates
    df_pivot = df_female.pivot_table(index='name', columns='TIME_PERIOD', values='OBS_VALUE', aggfunc='mean')
    df_pivot[year1] = pd.to_numeric(df_pivot[year1], errors='coerce')
    df_pivot[year2] = pd.to_numeric(df_pivot[year2], errors='coerce')

    df_pivot.dropna(inplace=True)

    df_pivot = df_pivot[(df_pivot[year1] != 0) & (df_pivot[year2] != 0)]

    # Calculate the difference
    df_pivot['difference'] = np.abs(df_pivot[year2] - df_pivot[year1])

    # Sort by the difference and take the top 10
    df_top10 = df_pivot.nlargest(10, 'difference').reset_index()
    df_filtered_top10_female = df_female[df_female['name'].isin(df_top10['name'])]
    df_filtered_top10_male = df_male[df_male['name'].isin(df_top10['name'])]

    # Ensure that the data is in the order of df_top10
    df_filtered_top10_female['name'] = pd.Categorical(df_filtered_top10_female['name'], categories=df_top10['name'], ordered=True)
    df_filtered_top10_male['name'] = pd.Categorical(df_filtered_top10_male['name'], categories=df_top10['name'], ordered=True)

    # Combining male and female data
    df_combined = pd.concat([
        df_filtered_top10_female.assign(sex='Female'),
        df_filtered_top10_male.assign(sex='Male'),
    ])
    df_pivot = df_combined.pivot_table(index=['name', 'TIME_PERIOD'], columns='sex', values='OBS_VALUE').reset_index()

    # Calculate the difference between the values ‘F’ and ‘M’.
    df_pivot['difference'] = df_pivot['Male'] - df_pivot['Female']

    # Unpivot the DataFrame back to its original form, adding the column ‘difference’.
    df_combined = pd.merge(df_combined, df_pivot[['name', 'TIME_PERIOD', 'difference']], on=['name', 'TIME_PERIOD'], how='left')

    # Add a column for the base of the lollipop (y=0)
    df_combined['y0'] = 0

    for name in df_combined['name'].unique():
        for period in df_combined['TIME_PERIOD'].unique():
            female_value = df_female[(df_female['name'] == name) & (df_female['TIME_PERIOD'] == period)]['OBS_VALUE']
            male_value = df_male[(df_male['name'] == name) & (df_male['TIME_PERIOD'] == period)]['OBS_VALUE']
            if not female_value.empty and not male_value.empty:
                middle_point = (female_value.values[0] + male_value.values[0]) / 2
                df_combined.loc[(df_combined['name'] == name) & (df_combined['TIME_PERIOD'] == period), 'middle_point'] = middle_point

    # Creating the Altair clustered bar chart for female values
    points = alt.Chart(df_combined).mark_point(size= 100, filled = True).encode(
        x=alt.X('OBS_VALUE:Q', title=''),
        y=alt.Y('TIME_PERIOD:O', title=''),
        color=alt.Color('TIME_PERIOD:N', legend=alt.Legend(title="Year")),
        shape=alt.Shape('sex:N', scale=alt.Scale(domain=['Female', 'Male'], range=['circle', 'square']), title = "Sex"),
        tooltip = [alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:N', title='Year'), alt.Tooltip('sex:N', title='Sex'), alt.Tooltip('OBS_VALUE:Q', title='Incidence Rate'), alt.Tooltip('difference:Q', title='Difference (M-F)')]
    ).properties(
        height=56  # Adjust the height of each facet
    )

    lines = alt.Chart(df_combined).mark_line().encode(
        x=alt.X('OBS_VALUE:Q', title='Incidence rate'),
        y=alt.Y('TIME_PERIOD:O', title=''),
        color=alt.value('grey'),
        detail=['name:N', 'TIME_PERIOD:N']
    )

    # Adding difference labels
    text = alt.Chart(df_combined).mark_text(dy=-5, color='black').encode(
        x=alt.X('middle_point:Q'),
        y=alt.Y('TIME_PERIOD:O'),
        text=alt.Text('difference:Q', format='.2f'),
        detail=['name:N', 'TIME_PERIOD:N']
    )

    # Overlay both graphs
    layered_chart = alt.layer(lines, points, text).resolve_scale(
        y='shared',
        x='shared'
    )

    # Show the faceted chart
    final_chart = alt.layer(layered_chart, data=df_combined).facet(
        row=alt.Row('name:N', sort=df_combined['geo'].unique().tolist(), header=alt.Header(labelAngle=0, labelAlign='left'), title='Region and Year')
    ).resolve_scale(
        y='shared',
        x='shared'
    ).configure_facet(
        spacing=0
    ).configure_header(
        titleFontSize=14,
    ).properties(
    title= f'Top 10 Regions with Highest Differences between {year1} and {year2}'
    )

    st.altair_chart(final_chart, use_container_width=True)

with cols[2]:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Ensure TIME_PERIOD is in time format.
    idx = df_accidents.groupby('TIME_PERIOD')['OBS_VALUE'].idxmax()
    df_max = df_accidents.loc[idx].reset_index(drop=True)
    df_accidents['TIME_PERIOD'] = df_accidents['TIME_PERIOD'].astype(str)

    # Create separate graphs for each sex
    chart_female = alt.Chart(df_accidents[df_accidents['sex'] == 'Female']).mark_area(opacity=0.8).encode(
        x=alt.X('TIME_PERIOD:O', title='Year'),
        y=alt.Y('mean(OBS_VALUE):Q', title='Mean Rate Homicide'),
        color=alt.Color('sex:N', scale=color_sex, title='Sex'),
        tooltip = [ alt.Tooltip('mean(OBS_VALUE):Q', title= 'Regions\' Rate Homicide', format=".2f"), alt.Tooltip('sex:N', title='Sex'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
    )

    chart_male = alt.Chart(df_accidents[df_accidents['sex'] == 'Male']).mark_area(opacity=0.8).encode(
        x=alt.X('TIME_PERIOD:O', title='Year'),
        y=alt.Y('mean(OBS_VALUE):Q', title='Mean Rate Homicide'),
        color=alt.Color('sex:N', scale=color_sex, title='Sex'),
        tooltip = [alt.Tooltip('mean(OBS_VALUE):Q', title= 'Regions\' Rate Homicide', format=".2f"), alt.Tooltip('sex:N', title='Sex'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
    )

    # Combine graphics using alt.layer
    chart = alt.layer(chart_male, chart_female).resolve_scale(
        y='shared'
    ).properties(
        width=800,
        height=340,
        title='Mean Regions\' Rate of Fatal Accidents by Year and Sex'
    )

    st.altair_chart(chart, use_container_width=True)

    df_erate = load_data("data/employment_rate_noms.txt")
    df_erate['TIME_PERIOD'] = pd.to_numeric(df_erate['TIME_PERIOD'], errors='coerce')
    df_accidents['TIME_PERIOD'] = pd.to_numeric(df_accidents['TIME_PERIOD'], errors='coerce')
    df_erate['sex'] = df_erate['sex'].replace({'M': 'Male', 'F': 'Female'})
    df_erate = df_erate[df_erate['sex']!= "T"]
    df_erate['Percentage_Display'] = df_erate['OBS_VALUE'].apply(
        lambda x: f"{x:.2f}%"
    )
    df_accidents = df_accidents[df_accidents['sex']!= "T"]

    df_merge = pd.merge(df_erate, df_accidents, on=['name', 'TIME_PERIOD', 'sex'], suffixes=('_Emp', '_Acc'))

    scatter_plot = alt.Chart(df_merge).mark_point(filled=True).encode(
        x = alt.X('OBS_VALUE_Emp:Q', title='Percentage of population employed'),
        y= alt.Y('OBS_VALUE_Acc:Q', title= 'Rate of incidents'),
        color=alt.Color('sex:N', scale=color_sex, title='Sex'),
        tooltip=[alt.Tooltip('Percentage_Display:N', title='Employment'), alt.Tooltip('OBS_VALUE_Acc:Q', title='Rate of incidents'), alt.Tooltip('sex:N', title='Sex')]
    ).properties(
        width=270,
        height=270,
        title='Correlation Employed Population vs Accidents Rate'
    )
    st.altair_chart(scatter_plot, use_container_width=True)

st.markdown("### Self-reported unmet need for medical examination and care")
st.write("The share of the population aged 16 and over reporting unmet needs for medical care due to is too expensive or too far to travel or waiting list.")

df_medical = load_data("data/medical_examination_filter.txt")
df_medical['TIME_PERIOD'] = pd.to_numeric(df_medical['TIME_PERIOD'], errors='coerce')

all_years = sorted(df_medical['TIME_PERIOD'].unique())

df_medical = df_medical[df_medical['sex'].isin(['F', 'M'])]

df_medical['sex'] = df_medical['sex'].replace({'M': 'Male', 'F': 'Female'})

years = df_medical['TIME_PERIOD'].unique()
cols = st.columns([2,2])
color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

with cols[0]:
    selected_year = st.slider(
        "Select a year:",
        min_value=int(df_medical['TIME_PERIOD'].min()),
        max_value=int(df_medical['TIME_PERIOD'].max()),
        value=int(df_medical['TIME_PERIOD'].min()),
        step=1
    )
df_pivot = df_medical.pivot_table(index=['geo', 'name', 'TIME_PERIOD'], columns='sex', values='OBS_VALUE').reset_index()

# Calculate the difference
df_pivot['Difference'] = df_pivot['Male'] - df_pivot['Female']

# Merge the difference back into the original dataframe
df_diff = df_pivot[['geo', 'name', 'TIME_PERIOD', 'Difference']]

# Filter the data for the selected year
df_filtered = df_diff[df_diff['TIME_PERIOD'] == selected_year]

all_regions = df_medical['name'].unique()
all_regions.sort()
regions_with_values = df_filtered['name'].unique()
regions_without_values = list(set(all_regions) - set(regions_with_values))

df_missing = pd.DataFrame({'name': regions_without_values, 'Difference': [0]*len(regions_without_values)})

df_filtered['Category'] = df_filtered['Difference'].apply(lambda x: 'Male > Female' if x > 0 else 'Female > Male' if x < 0 else 'No Difference')
df_missing['Category'] = 'No Data'

# Concatenate the filtered and missing dataframes
df_combined = pd.concat([df_filtered, df_missing])


x_domain = [df_diff ['Difference'].min(), df_diff ['Difference'].max()]
# Create the base chart
pyramid_chart = alt.Chart(df_filtered[df_filtered['Difference'] != 0]).mark_bar(size=10).encode(
    y=alt.Y('name:N', title='Region', scale=alt.Scale(domain=all_regions)),
    x=alt.X('Difference:Q', title='% Difference in Population (Male - Female)', scale=alt.Scale(domain=x_domain)),
    color=alt.Color('Category:N', scale=alt.Scale(domain=['Male > Female', 'Female > Male', 'No Difference', 'No Data'], range=['#7fbf7b', '#af8dc3', '#ec432c', 'gray']), legend=alt.Legend(title="Category")),
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('Difference:Q', title='% Difference (M - F)')
    ]
)

# Create the point chart for zero differences
point_chart = alt.Chart(df_filtered[df_filtered['Difference'] == 0]).mark_point(
    size=40, filled=True
).encode(
    x=alt.X('Difference:Q', title=''),
    y=alt.Y('name:N', title='Region', sort=alt.EncodingSortField(field='Difference', order='descending')),
    color=alt.value('#ec432c'),
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('Difference:Q', title='% Difference (M - F)')
    ]
)

missing_chart = alt.Chart(df_missing).mark_point(
    shape='stroke',
    color='gray',
    size=100
).encode(
    x=alt.X('Difference:Q'),
    y=alt.Y('name:N', title='Region', sort=alt.EncodingSortField(field='Difference', order='descending')),
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('Category:N', title='')
    ]
)

# Layer the bar and point charts
combined_chart = alt.layer(pyramid_chart, point_chart, missing_chart).properties(
    width=600,
    height=710,
    title=f'Gender Difference (Male - Female) in Unmet Medical Needs by Region in {selected_year}'
).configure_axis(
    labelFontSize=9,  # Smaller font size for labels
)

# Display the chart in Streamlit
st.altair_chart(combined_chart, use_container_width=True)

st.markdown("### Standardised death rate due to homicide")
st.write("The standardised death rate of homicide and injuries inflicted by another person with the intent to injure or kill by any means, including ‘late effects’ from assault.")

df_homicide = load_data("data/homicide_noms.txt")
df_homicide['TIME_PERIOD'] = pd.to_numeric(df_homicide['TIME_PERIOD'], errors='coerce')
df_homicide['sex'] = df_homicide['sex'].replace({'F': 'Female', 'M': 'Male'})

df_female = df_homicide[df_homicide['sex'] == 'Female']
df_male = df_homicide[df_homicide['sex'] == 'Male']

# Find the maximum value and the corresponding year for each region and gender
idx_female = df_female.groupby('name')['OBS_VALUE'].idxmax()
df_max_female = df_female.loc[idx_female].reset_index(drop=True)

idx_male = df_male.groupby('name')['OBS_VALUE'].idxmax()
df_max_male = df_male.loc[idx_male].reset_index(drop=True)

# Combine results in a single DataFrame
df_max = pd.concat([df_max_female, df_max_male])

df_pivot = df_homicide.pivot_table(index=['geo', 'name', 'TIME_PERIOD'], columns='sex', values='OBS_VALUE').reset_index()

df_pivot['Total'] = df_pivot['Female'] + df_pivot['Male']

# Calculate the percentage for each gender
df_pivot['Female_Percentage'] = (df_pivot['Female'] / df_pivot['Total']) * 100
df_pivot['Male_Percentage'] = (df_pivot['Male'] / df_pivot['Total']) * 100

# Unpivot the dataframe for Altair compatibility
df_unpivot_percentage = df_pivot.melt(id_vars=['geo', 'name', 'TIME_PERIOD'], value_vars=['Female_Percentage', 'Male_Percentage'],
                           var_name='sex', value_name='Percentage')

# Replace the values in 'sex' to match the original dataset
df_unpivot_percentage['sex'] = df_unpivot_percentage['sex'].replace({'Female_Percentage': 'Female', 'Male_Percentage': 'Male'})

df_unpivot_value = df_pivot.melt(id_vars=['geo', 'name', 'TIME_PERIOD'], value_vars=['Female', 'Male'],
                                 var_name='sex', value_name='OBS_VALUE')

df_homicide = pd.merge(df_unpivot_percentage, df_unpivot_value, on=['geo', 'name', 'TIME_PERIOD', 'sex'])

# Filter regions with data for at least 10 different years
regions_with_enough_data = df_homicide.groupby('name')['TIME_PERIOD'].nunique()
regions_with_enough_data = regions_with_enough_data[regions_with_enough_data >= 12].index

df_homicide_l = df_homicide[df_homicide['TIME_PERIOD'] > 2002]

years = np.sort(df_homicide['TIME_PERIOD'].unique())
all_regions = df_homicide_l['name'].unique()
all_regions.sort()
color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

cols = st.columns([1,1])
with cols[0]:
    selected_year = st.slider(
        "Select a year:",
        min_value=int(df_homicide_l['TIME_PERIOD'].min()),
        max_value=int(df_homicide_l['TIME_PERIOD'].max()),
        value=int(df_homicide_l['TIME_PERIOD'].min()),
        step=1
    )

    df_filtered = df_homicide_l[df_homicide_l['TIME_PERIOD'] == selected_year]

    all_years = df_homicide_l['TIME_PERIOD'].unique()
    all_regions = df_homicide_l['name'].unique()
    all_regions.sort()
    all_sex = df_homicide_l['sex'].unique()
    full_index = pd.MultiIndex.from_product([all_regions, all_sex], names=['name', 'sex'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Completar el DataFrame original
    df_complete = pd.merge(full_df, df_filtered, on=['name', 'sex'], how='left')

    placeholder_value = -999
    df_complete['OBS_VALUE'].fillna(placeholder_value, inplace=True)
    df_complete['OBS_VALUE_Display'] = df_complete['OBS_VALUE'].apply(
        lambda x: f"{x:.2f}" if x != placeholder_value else "No Data"
    )
    df_complete['Percentage'].fillna(placeholder_value, inplace=True)
    df_complete['Percentage_Display'] = df_complete['Percentage'].apply(
        lambda x: f"{x:.2f}%" if x != placeholder_value else "No Data"
    )


    # Create the chart with the order based on ‘max_value’.
    chart = alt.Chart(df_complete).mark_bar().encode(
        y=alt.Y('name:N', title = "Region",  scale=alt.Scale(domain=all_regions)),  # Sort from highest to lowest
        x=alt.X('OBS_VALUE:Q', stack='normalize', title='Percentage of homicide'),
        color=alt.condition(alt.datum.OBS_VALUE != -999,
        alt.Color('sex:N', scale = color_sex, title = "Sex"), alt.value('lightgray')),
        tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('sex:N', title='Sex'),
        alt.Tooltip('OBS_VALUE_Display:N', title='Rate of homicide'),
        alt.Tooltip('Percentage_Display:N', title='% of homicide')
    ]
    ).properties(
        width=550,
        height=730,
        title= alt.TitleParams(
        text=f'Standardised Death Rate Due to Homicide by Gender in {selected_year}',
        )
    ).configure_axis(
        labelFontSize=9,  # Smaller font size for labels
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

with cols[1]:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])
    selection = alt.selection_multi(fields=['sex'], bind='legend')

    nearest = alt.selection_point(nearest= True, on= 'mouseover', fields= ['TIME_PERIOD'])

    chart = alt.Chart(df_max).mark_circle().encode(
        x=alt.X('TIME_PERIOD:N', title='Year', scale=alt.Scale(domain=years)),
        y=alt.Y('name:N', title = "Region",  scale=alt.Scale(domain=all_regions)),
        color=alt.Color('sex:N', scale=color_sex, title='Sex'),
        opacity=alt.condition(selection, alt.value(0.6), alt.value(0)),
        size=alt.Size('OBS_VALUE:Q', title="Rate of Homicide"),  # Dot size proportional to value
        tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('TIME_PERIOD:O', title='Year'),
        alt.Tooltip('OBS_VALUE:Q', title='Highest Homicide Rate'),
        alt.Tooltip('sex:N', title='Sex')
    ]
    ).properties(
        width=550,
        height=730,
        title= alt.TitleParams(
        text='Year of Highest Female Index Value by Region',
        )
    ).add_params(
    nearest,
    selection
    )

    rules = alt.Chart(df_homicide).mark_rule().encode(
        x='TIME_PERIOD:O',
        color=alt.value('gray'),
        opacity = alt.value(0.01),
        tooltip=[alt.Tooltip('TIME_PERIOD:O', title='Year')]
    ).transform_filter(
        nearest
    )

    ch = alt.layer(chart, rules).configure_axis(
        labelFontSize=9,  # Smaller font size for labels
    )
    # Display the chart in Streamlit
    st.altair_chart(ch, use_container_width=True)
