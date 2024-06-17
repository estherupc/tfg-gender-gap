"""
This section of the script focuses on visualizing gender-related metrics in employment using Altair.

The script performs the following tasks:
1. Load and preprocess data related to education metrics.
2. Create color mappings and scales for visualization.
3. Generate interactive visualizations to display gender differences in employment attainment.

Esther Fanyanàs I Ropero
"""
# ------- IMPORTS -------
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from PIL import Image
import numpy as np


st.set_page_config(page_title="Employment", page_icon="📊", layout="wide",
initial_sidebar_state="expanded")

st.title("📊 Employment")
st.markdown("---")
st.write(
    """The labor market remains one of the critical areas where gender disparities are evident. This section examines the gender gap in employment rates, types of employment, positions held by women in senior management and in government, the gender pay gap and the labour force due to caring responsabilities. """
)

st.markdown("---")

def assign_color_category(row):
        if row['name'] == 'European Union':
            return row['name']
        elif row['OBS_VALUE'] > 0:
            return 'Male > Female'
        else:
            return 'Female > Male'

data = pd.read_csv('data/government_2023.txt')

st.markdown("### Seats held by women in national parliaments and governments")

cols = st.columns([1, 4])
unique_regions = data['name'].unique()
unique_regions.sort()
with cols[0]:
    # Country selection in Streamlit
    selected_country = st.selectbox('Select a region:', unique_regions)

col1, col2= st.columns([5.9,  2])
with col1:
    tab1, tab2 = st.tabs(["Parliament", "Government"])
    st.write("")
    with tab1:
        legend_html = """
            <div style='position: relative; display: flex; justify-content: flex-end;'>
                <div style='position: absolute; top: 50px; right: 40px; background-color: white; padding: 10px; border-radius: 15px;font-size: 12px;color: grey;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 20px; height: 20px; background-color: #af8dc3; margin-right: 10px;'></div>
                        <span>Female</span>
                    </div>
                    <div style='display: flex; align-items: center; margin-top: 5px;'>
                        <div style='width: 20px; height: 20px; background-color: #7fbf7b; margin-right: 10px;'></div>
                        <span>Male</span>
                    </div>
                </div>
            </div>
        """
        # Display the legend and pictograms in Streamlit
        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("###### Parliament Data (2023)")
        data_parl = data[data['org_inst'] == 'PARL_NAT']
        percentage_women = data_parl[data_parl['name'] == selected_country]['OBS_VALUE'].iloc[0]
        st.write(f"The percentage of seats held by women in {selected_country}'s parliament in 2023 is {percentage_women}%.")
        num_chairs = 100  # Total number of chairs

        # Configurations for semicircles
        radii = [0.5, 1.0, 1.5, 2.0, 2.5]  # Radii for semicircles
        seats_per_circle = [8, 12, 20, 28, 32]  # Number of seats per semicircle
        total_seats = sum(seats_per_circle)

        # Create figure
        fig = go.Figure()

        # Prepare entry data in all semicircles
        seat_data = []

        # Calculate the position of the seats for each semicircle and store them.
        for radius, seats in zip(radii, seats_per_circle):
            angles = np.linspace(0, np.pi, seats)
            seat_data.extend([(radius * np.cos(angle), radius * np.sin(angle), angle) for angle in angles])

        # Sort seat data by angle to fill them like clockwork
        seat_data.sort(key=lambda x: -x[2])  # Descending order by angle

        # Separate x-, y-coordinates and colours after ordering
        all_x = [x[0] for x in seat_data]
        all_y = [x[1] for x in seat_data]
        colors = ['#af8dc3' if i < percentage_women else '#7fbf7b' for i in range(total_seats)]

        # Add the seats to the figure
        fig.add_trace(go.Scatter(
            x=all_x,
            y=all_y,
            marker=dict(color=colors, size=15),
            mode='markers',
            hoverinfo='none',
            showlegend=False  # Do not show this trace in the legend
        ))

        # Add a vertical line in the centre
        fig.add_shape(type="line",
            x0=0, y0=-0.2, x1=0, y1=max(radii) + 0.1,
            line=dict(color="Black",width=1)
        )

        # Adjust the plot layout and axes visibility
        fig.update_xaxes(visible=False)  # Hide the x-axis
        fig.update_yaxes(visible=False)  # Hide the y-axis

        # Ajustar los límites del gráfico y los ejes
        fig.update_xaxes(range=[-max(radii) - 0.1, max(radii) + 0.1], showline=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-0.2, max(radii) + 0.1], showline=False, showgrid=False, zeroline=False)

        fig.update_layout(
            plot_bgcolor='white',
            margin=dict(t=0, b=0, l=0, r=0),
            width=500,
            height=400,
            legend=dict(
                font=dict(
                    color='gray'
                )
            )
        )

        config = {
            'displayModeBar': False  # This hides the modebar
        }

        # Mostrar la figura en Streamlit
        st.plotly_chart(fig, use_container_width=True, config=config)

    with tab2:

        legend_html = """
            <div style='position: relative; display: flex; justify-content: flex-end;'>
                <div style='position: absolute; top: 50px; right: 40px; background-color: white; padding: 10px; border-radius: 15px;font-size: 12px; color: grey;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 20px; height: 20px; background-color: #af8dc3; margin-right: 10px;'></div>
                        <span>Female</span>
                    </div>
                    <div style='display: flex; align-items: center; margin-top: 5px;'>
                        <div style='width: 20px; height: 20px; background-color: #7fbf7b; margin-right: 10px;'></div>
                        <span>Male</span>
                    </div>
                </div>
            </div>
        """
        # Display the legend and pictograms in Streamlit
        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("###### Government Data (2023)")
        data_gov = data[data['org_inst'] == 'GOV_NAT']
        percentage_women = data_gov[data_gov['name'] == selected_country]['OBS_VALUE'].iloc[0]
        st.write(f"The percentage of seats held by women in {selected_country}'s government in 2023 is {percentage_women}%.")

        # Upload images
        man_logo_path = "images/user-green.png"
        woman_logo_path = "images/user-purple.png"
        man_logo = Image.open(man_logo_path)
        woman_logo = Image.open(woman_logo_path)

        # Creating the figure
        fig = go.Figure()

        # Coordinates for people
        x_coords_row1 = [i * 1 for i in range(1, 9)]
        x_coords_row2 = [i * 1 for i in range(1, 9)]
        y_coords_row1 = [1] * 8
        y_coords_row2 = [0] * 8

        total_figures = 16
        num_purple = round(total_figures * (percentage_women / 100))
        num_green = total_figures - num_purple

        # Create the list of images by percentage
        images = [woman_logo] * num_purple + [man_logo] * num_green

        # Distribute the images in two rows
        images_row1 = images[:8]
        images_row2 = images[8:]


        # Adding images to the figure
        for x, img in zip(x_coords_row1, images_row1):
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=x,
                    y=1.25,
                    sizex=1,
                    sizey=7,
                    xanchor="center",
                    yanchor="middle"
                )
            )

        # Add images to the figure for the second row
        for x, img in zip(x_coords_row2, images_row2):
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=x,
                    y=-0.05,
                    sizex=1,
                    sizey=7,
                    xanchor="center",
                    yanchor="middle"
                )
            )

        # Adjust layout
        fig.update_layout(
            xaxis=dict(
                visible=False,
                range=[0, 11]
            ),
            yaxis=dict(
                visible=False,
                range=[-1.5, 2.5]
            ),
            margin=dict(l=10, r=20, t=10, b=10),
            width=1600,
            height=400
        )

        # Show the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True, config=config)

with col2:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    data['inst'] = data['org_inst'].replace({'PARL_NAT': 'Parliament', 'GOV_NAT': 'Government'})
    data['org_inst'] = data['org_inst'].replace({'PARL_NAT': 'PARL', 'GOV_NAT': 'GOV'})
    filtered_data = data[data['name'] == selected_country]

    chart_global = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('org_inst:N', title = "Institution" , axis=alt.Axis(labelAngle=0)),  # Ordenar de mayor a menor
        y=alt.Y('OBS_VALUE:Q', scale = alt.Scale(domain = [data['OBS_VALUE'].min(), data['OBS_VALUE'].max()]), title='Percentage of Women'),
        color=alt.Color('inst:N', scale= alt.Scale(domain =[ "Government", "Parliament"], range = ["#972554", "#77c4d3"]), title = "Institution"),
        tooltip=[
        alt.Tooltip('inst:N', title='Institution'),
        alt.Tooltip('OBS_VALUE:Q', title='% of Women', format='.2f'),
        alt.Tooltip('name:N', title='Region')
    ]
    ).properties(
        width=400,
        height=500,
        title = alt.TitleParams(
        text='Women in Government vs Parliament',
        )
    )
    st.altair_chart(chart_global, use_container_width=True)

df_management = pd.read_csv("data/management_noms.txt", sep=",", header=None, index_col=False)
df_management.columns = df_management.iloc[0]
df_management = df_management[1:]
df_management.columns.name = None
df_management.reset_index(drop=True, inplace=True)
df_management['OBS_VALUE'] = pd.to_numeric(df_management['OBS_VALUE'], errors='coerce')
df_management['TIME_PERIOD'] = pd.to_numeric(df_management['TIME_PERIOD'], errors='coerce')
st.markdown("### Positions held by women in senior management positions")
df_management = df_management.loc[df_management['TIME_PERIOD'] >= 2012]
cols = st.columns([3, 2.5])

with cols[0]:
    selected_time_period = st.slider(
        "Select a year:",
        min_value=int(df_management['TIME_PERIOD'].min()),
        max_value=int(df_management['TIME_PERIOD'].max()),
        value=int(df_management['TIME_PERIOD'].min()),
        step=1
    )
df_management = df_management[df_management['TIME_PERIOD'] == selected_time_period]
df_management['Percentage_Display'] = df_management['OBS_VALUE'].apply(
    lambda x: f"{x:.2f}%"
)
mean_value = df_management['OBS_VALUE'].mean()

st.markdown("➡️ Click on a bar to see the detailed breakdown of professional positions for the selected region.")

# Overall percentage of women in senior positions by region
overall_chart = alt.Chart(df_management).mark_bar().encode(
    x=alt.X('name:N', sort='-y', title='Region'),
    y=alt.Y('mean(OBS_VALUE):Q', scale=alt.Scale(domain=[0, 40]), title='Percentage of Women in Senior Positions'),
    color=alt.condition(
        alt.datum.name == 'European Union',
        alt.value('#444444'),
        alt.value('#ec432c')
    ),
    tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('mean(OBS_VALUE):Q', title='Total Percentage'),  alt.Tooltip('TIME_PERIOD:O', title='Year')],
).properties(
    width=700,
    height=400,
)

mean_line = alt.Chart(df_management).mark_rule(color='black', strokeDash=[5, 5]).encode(
    y=alt.datum(mean_value)
)

median_text = alt.Chart(pd.DataFrame({'y': [mean_value]})).mark_text(
    align='center',
    baseline='middle',
    dx=600,  # Adjust to move the text away from the line
    dy=-10
).encode(
    y=alt.Y('y:Q'),
    x=alt.value(10),  # Adjust based on where you want to place the text
    text=alt.value(f'Median of Women: {mean_value:.2f}%')
)

overall_chart = overall_chart + mean_line + median_text

# Create a selection
selection = alt.selection_single(empty='none', fields=['name'])

# Add selection to the overall chart
overall_chart = overall_chart.add_selection(selection)

# Detailed chart for the selected region
detailed_chart = alt.Chart(df_management).transform_filter(
    selection
).mark_bar().encode(
    x=alt.X('prof_pos:N', axis=alt.Axis(labelAngle=0), title='Professional Position'),
    y=alt.Y('OBS_VALUE:Q', title='Percentage of Women', scale=alt.Scale(domain=[0, 50])),
    column=alt.Column('name:N', title='Region'),
    color=alt.Color('pos:N',  scale=alt.Scale(domain=["Board members", "Executives"],range=["#843d34", "#c0e15c"]), title="Professional Position"),
    tooltip=[alt.Tooltip('pos:N', title='Professional Position'), alt.Tooltip('Percentage_Display:N', title='Percentage'), alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
).properties(
    width={'step': 60},
    height=370,
)

combined_chart = alt.hconcat(overall_chart, detailed_chart).resolve_legend(
    color="independent"
).configure_legend(
    orient='bottom',  # Move the legend to the bottom
    titleOrient='top',
    columns=2,  # Configure columns to make the legend more compact
    labelFontSize=12,
    titleFontSize=14
)

st.markdown(
    f"<h6 style='color: #333333;''>Percentage of Women in Senior Positions by Region in {selected_time_period} and Detailed Senior Positions</h6>",
    unsafe_allow_html=True
)
st.altair_chart(combined_chart)

df_labour_force = pd.read_csv("data/labour_force_noms.txt", sep=",", header=None, index_col=False)
df_labour_force.columns = df_labour_force.iloc[0]
df_labour_force = df_labour_force[1:]
df_labour_force.columns.name = None
df_labour_force.reset_index(drop=True, inplace=True)

st.markdown("### Persons outside the labour force due to caring responsibilities")

df_labour_force = df_labour_force[df_labour_force['sex'].isin(['F', 'M'])]

df_labour_force['TIME_PERIOD'] = pd.to_numeric(df_labour_force['TIME_PERIOD'], errors='coerce')
df_labour_force['OBS_VALUE'] = pd.to_numeric(df_labour_force['OBS_VALUE'], errors='coerce')
df_labour_force = df_labour_force.loc[df_labour_force['TIME_PERIOD'] >= 2009]

all_years = df_labour_force['TIME_PERIOD'].unique()
all_regions = df_labour_force['name'].unique()
full_index = pd.MultiIndex.from_product([all_years, all_regions], names=['TIME_PERIOD', 'name'])
full_df = pd.DataFrame(index=full_index).reset_index()

# Pivoting the table to calculate the difference between women and men
pivot_table = df_labour_force.pivot_table(
    index=['geo', 'name', 'TIME_PERIOD'],
    columns='sex',
    values='OBS_VALUE'
).reset_index()

# Calculate the difference (Female - Male)
pivot_table['Difference'] = pivot_table['M'] - pivot_table['F']

# Create a new DataFrame with the differences
difference_data = pivot_table[['geo', 'name', 'Difference', 'TIME_PERIOD']]

# Get the full list of regions
all_regions = df_labour_force['geo'].unique().tolist()

all_names = df_labour_force['name'].unique().tolist()

geo_name_dict = dict(zip(all_regions, all_names))

# Convert 'No Data' to NaN for the Altair chart
difference_data['Difference'] = pd.to_numeric(difference_data['Difference'], errors='coerce')

difference_data['name'] = difference_data['geo'].map(geo_name_dict)

# Combine with the original DataFrame
df_complete = pd.merge(full_df, difference_data, on=['TIME_PERIOD', 'name'], how='left')

placeholder_value = -999
df_complete['Difference'].fillna(placeholder_value, inplace=True)
df_complete['Difference_Display'] = df_complete['Difference'].apply(
    lambda x: f"{x:.2f}%" if x != placeholder_value else "No Data"
)

# Define the colour scale for the difference
color_scale = alt.Scale(domain=[-4.5, -3.6,-2.7, -1.8, -0.9, 0, 0.9, 1.8, 2.7, 3.6, 4.5],  # Adjust the domain according to your data
                        range=[ '#40004b',  '#762a83'  ,'#9970ab' , '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61','#1b7837',   '#00441b',])

# Create the heatmap using the differences
heatmap = alt.Chart(df_complete).mark_rect().encode(
    x = alt.X('TIME_PERIOD:O', title='Year'),
    y=alt.Y('name:N', title='Region', sort=all_regions),
    color=alt.condition(
        alt.datum.Difference != -999,
        alt.Color('Difference:Q', scale=color_scale, title='Difference', legend=alt.Legend(
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
            labelExpr="datum.value == -4 ? ' More Female' : datum.value == 0 ? ' Equal' : datum.value == 4 ? ' More Male' : ''"
            )),
        alt.value('lightgray')
    ),
    tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('Difference_Display:N', title='Difference'),  alt.Tooltip('TIME_PERIOD:O', title='Year')]
).properties(
    width=200,
    height=730,
    title=f'Percentage of Gender Differences (Male - Female) by Region and Year'
).configure_axis(
    labelFontSize=9,  # Smaller font size for labels
)
col1, col2 = st.columns([1, 1.2])

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.altair_chart(heatmap, use_container_width=True)

with col1:
    selected_time_period = st.slider(
        "Select a year:",
        min_value=int(df_labour_force['TIME_PERIOD'].min()),
        max_value=int(df_labour_force['TIME_PERIOD'].max()),
        value=int(df_labour_force['TIME_PERIOD'].min()),
        step=1
    )
# Mostrar el gráfico en Streamlit
filtered_data = df_labour_force[df_labour_force['TIME_PERIOD'] == selected_time_period]

filtered_data = filtered_data[filtered_data['sex'] == 'F']

# You can now select the top and bottom 10 countries based on this new metric.
top_ratio_countries = filtered_data.nlargest(10, 'OBS_VALUE')
bottom_ratio_countries = filtered_data.nsmallest(10, 'OBS_VALUE')

combined_ratio_countries = pd.concat([top_ratio_countries, bottom_ratio_countries], axis=0)
combined_ratio_countries['Category'] = ['More'] * len(top_ratio_countries) + ['Less'] * len(bottom_ratio_countries)
combined_ratio_countries['Percentage_Display'] = combined_ratio_countries['OBS_VALUE'].apply(
    lambda x: f"{x:.2f}%"
)
# Create the Altair graphic
chart = alt.Chart(combined_ratio_countries).mark_bar().encode(
    x=alt.X('OBS_VALUE:Q', title='% Women outside the labour force', scale=alt.Scale(domain=[df_labour_force['OBS_VALUE'].min(), df_labour_force['OBS_VALUE'].max()])),
    y=alt.Y('name:N', title='Region', sort=alt.EncodingSortField(field= 'OBS_VALUE', order='descending')),
    color=alt.Color('Category:N', title='Category', scale=alt.Scale(domain=['More', 'Less'], range=['#F9B7B2', '#ec432c'])),
    tooltip=[alt.Tooltip('name:N', title='Region'),alt.Tooltip('Percentage_Display:N', title='% of women'),  alt.Tooltip('TIME_PERIOD:O', title='Year')]
).properties(
    width=500,
    height=700,
    title=f'More/Less 10 Women Outside Labour for {selected_time_period}'
)
with col1:
    # Show the graph in Streamlit
    st.altair_chart(chart, use_container_width=True)

st.markdown("### Gender pay gap in unadjusted form")
col1, col2 = st.columns([1,1.2])

df_gender_pay = pd.read_csv("data/pay_gap_noms.txt", sep=",", header=None, index_col=False)
df_gender_pay.columns = df_gender_pay.iloc[0]
df_gender_pay = df_gender_pay[1:]
df_gender_pay.columns.name = None
df_gender_pay.reset_index(drop=True, inplace=True)

df_gender_pay['TIME_PERIOD'] = pd.to_numeric(df_gender_pay['TIME_PERIOD'], errors='coerce')
df_gender_pay['OBS_VALUE'] = pd.to_numeric(df_gender_pay['OBS_VALUE'], errors='coerce')
df_gender_pay = df_gender_pay[df_gender_pay['TIME_PERIOD'] > 2005]

with col1:
    color_scale = alt.Scale(domain=[-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30],  # Ajustar el dominio según tus datos
                            range=[ '#40004b',  '#762a83'  ,'#9970ab' , '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61','#1b7837',   '#00441b',])

    years = df_gender_pay['TIME_PERIOD'].unique()
    max_value =  max(df_gender_pay['OBS_VALUE'])
    min_value = min(df_gender_pay['OBS_VALUE'])
    selected_year = st.slider('Select a year:', int(years.min()), int(years.max()), int(years.max()))

    # Filter the DataFrame based on the selected year
    filtered_data = df_gender_pay[df_gender_pay['TIME_PERIOD'] == selected_year]
    filtered_data['Percentage_Display'] = filtered_data['OBS_VALUE'].apply(
        lambda x: f"{x:.2f}%"
    )
    # Apply the function to create a new color category column
    filtered_data['color_category'] = filtered_data.apply(assign_color_category, axis=1)

    # Create a color scale
    color_scale = alt.Scale(
        domain=['European Union', 'Male > Female', 'Female > Male'],
        range=['#444444', '#7fbf7b', '#af8dc3']
    )


    sorted_geo = filtered_data.sort_values('OBS_VALUE', ascending=False)['name'].tolist()
    points = alt.Chart(filtered_data).mark_point(size=40, filled=True).encode(
        x=alt.X('OBS_VALUE:Q', title='% of average gross hourly earnings of men',
                scale=alt.Scale(domain=[min_value , max_value])),  # Escala ajustada para visualización
        y=alt.Y('name:N', title='Region', sort=sorted_geo),
        color = alt.Color('color_category:N', scale=color_scale, legend=alt.Legend(title='Category')),
        tooltip=[alt.Tooltip('name:N', title='Region'),alt.Tooltip('Percentage_Display:N', title='Gender pay gap'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
    ).properties(
        width=600,
        height=730
    )

    filtered_data['y0'] = 0

    # Lines (lollipop)
    lines = alt.Chart(filtered_data).mark_rule().encode(
        y=alt.Y('name:N', title='Region', sort=sorted_geo),  # Ensure the sorting key is consistent
        x=alt.X('y0:Q', axis=alt.Axis(grid=True), scale=alt.Scale(domain=[min_value , max_value])),
        x2=alt.X2('OBS_VALUE:Q'),
        color=alt.Color('color_category:N', scale=color_scale),
        tooltip=[alt.Tooltip('name:N', title='Region'),alt.Tooltip('Percentage_Display:N', title='Gender pay gap'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
    )

    # Pay gap target (0%)
    target = alt.Chart(filtered_data).mark_rule(color='black').encode(
        x=alt.X('y0:Q', scale=alt.Scale(domain=[min_value , max_value]))
    )

    mean_line = alt.Chart(filtered_data).mark_rule(color='black', strokeDash=[5, 5]).encode(
        x=alt.datum(mean_value)
    )

    median_text = alt.Chart(pd.DataFrame({'x': [mean_value]})).mark_text(
        align='center',
        baseline='middle',
        dx=0,  # Adjust to move the text away from the line
        dy=-20
    ).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[min_value , max_value])),
        y=alt.value(10),  # Adjust based on where you want to place the text
        text=alt.value(f'Median Pay Gap: {mean_value:.2f}%')
    )

    # Merge chart layers
    final_chart = (points + lines + target + mean_line + median_text).properties(
        title=f"Gender Pay Gap by Region for {selected_year}"
    ).configure_axis(
        labelFontSize=8,  # Smaller font size for labels
    )

    # Show the chart
    st.altair_chart(final_chart, use_container_width=True)

with col2:
    color_scale = alt.Scale(domain=[-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30],  # Adjust the domain according to data
                            range=[ '#40004b',  '#762a83'  ,'#9970ab' , '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61','#1b7837',   '#00441b',])
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    all_years = df_gender_pay['TIME_PERIOD'].unique()
    all_regions = df_gender_pay['name'].unique()
    full_index = pd.MultiIndex.from_product([all_years, all_regions], names=['TIME_PERIOD', 'name'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Combine with the original DataFrame
    df_complete = pd.merge(full_df, df_gender_pay, on=['TIME_PERIOD', 'name'], how='left')

    placeholder_value = -999
    df_complete['OBS_VALUE'].fillna(placeholder_value, inplace=True)
    df_complete['OBS_VALUE_Display'] = df_complete['OBS_VALUE'].apply(
        lambda x: f"{x:.2f}%" if x != placeholder_value else "No Data"
    )

    heatmap = alt.Chart(df_complete).mark_rect().encode(
        x=alt.X('TIME_PERIOD:O', title='Year'),
        y=alt.Y('name:N', title='Region'),
        color=alt.condition(
            alt.datum.OBS_VALUE != -999,
            alt.Color('OBS_VALUE:Q', scale=color_scale, title='% Gender Pay Gap', legend=alt.Legend(
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
            alt.value('lightgrey')  # Color for missing values
        ),
        tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('OBS_VALUE_Display:N', title='Gender pay gap'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
    ).properties(
        width=500,
        height=730,
        title=f'Percentage of Gender Pay Gap by Region and Year'
    ).configure_axis(
        labelFontSize=7.5,  # Smaller font size for labels
    )

    st.altair_chart(heatmap, use_container_width=True)

st.markdown("### Gender employment gap by type of employment")

df_employment = pd.read_csv("data/employment_noms.txt", sep=",", header=None, index_col=False)
df_employment.columns = df_employment.iloc[0]
df_employment = df_employment[1:]
df_employment.columns.name = None
df_employment.reset_index(drop=True, inplace=True)
df_employment['TIME_PERIOD'] = pd.to_numeric(df_employment['TIME_PERIOD'], errors='coerce')
df_employment['OBS_VALUE'] = pd.to_numeric(df_employment['OBS_VALUE'], errors='coerce')

color_scale = alt.Scale(range=['#7eb0d5', '#bd7ebe', '#fd7f6f', '#fdcce5'])
years = df_employment['TIME_PERIOD'].unique()
max_value =  max(df_employment['OBS_VALUE'])
min_value = min(df_employment['OBS_VALUE'])

# Slider to select the year
years = df_employment['TIME_PERIOD'].unique()
cols = st.columns([3, 2.5])

with cols[0]:
    selected_year = st.slider('Select a year', int(years.min()), int(years.max()), int(years.max()))

st.write("➡️ Click on the legend to filter and view only the selected category.")

# Filter the DataFrame based on the selected year
filtered_data = df_employment[df_employment['TIME_PERIOD'] == selected_year]
filtered_data['Percentage_Display'] = filtered_data['OBS_VALUE'].apply(
    lambda x: f"{x:.2f}%"
)

selection = alt.selection_multi(fields=['type'], bind='legend')

# Create the chart with the order based on ‘max_value’
points = alt.Chart(filtered_data).mark_point(size= 40, filled=True).encode(
    x=alt.X('name:O', title = "Region"),  # Order from highest to lowest
    y=alt.Y('OBS_VALUE:Q', title='Employment Gap (Percentage Points)',  scale = alt.Scale(domain = [min_value, max_value])),
    color=alt.Color('type:N', scale = color_scale, legend=alt.Legend(title="Type of Employment", labelLimit=500, orient='top')),
    tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year'), alt.Tooltip('type:N', title='Type'), alt.Tooltip('Percentage_Display:N', title='Difference')],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).properties(
    width=800,
    height=600,
    ).add_params(selection)

filtered_data['y0'] = 0

# Lines (lollipop)
lines = alt.Chart(filtered_data).mark_rule().encode(
    x='name:O',
    y=alt.Y('y0:Q', axis=alt.Axis(grid=True)),
    y2='OBS_VALUE:Q',
    color=alt.Color('type:N', scale= color_scale, legend=alt.Legend(title="Type of Employment", labelLimit=500, orient='top')),
    tooltip=[alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year'), alt.Tooltip('type:N', title='Type')],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
).add_params(selection)

annotations = alt.Chart(pd.DataFrame({
    'y': [80, 0, 1900, 1820],
    'text': ['Female', 'Higher Rate', 'Male', 'Higher Rate']
})).mark_text(
    align='left',
    baseline='middle',
    dx=5,
    dy=-10,
    color='darkgray'
).encode(
    x=alt.value(-100),  # To position the text on the left side of the graphic
    y=alt.Y('y:Q', axis=None),
    text='text:N'
)

# Combining dots and lines
lollipop_chart = alt.layer(lines, points).properties(
    title=f'% Difference between the employment rates of men and women aged 20 to 64 by Age Group and Region for {selected_year}'
)

# Combining text annotations with the main graphic
final_chart = alt.layer(lollipop_chart, annotations).resolve_scale(
    y='independent'
).configure_title(
    fontSize=16,
    anchor='middle',
    offset=20
)

# Show the graph in Streamlit
st.altair_chart(final_chart, use_container_width=True)

st.markdown("### Long-term unemployment")

df_unemployment = pd.read_csv("data/unemployment_noms.txt", sep=",", header=None, index_col=False)
df_unemployment.columns = df_unemployment.iloc[0]
df_unemployment = df_unemployment[1:]
df_unemployment.columns.name = None
df_unemployment.reset_index(drop=True, inplace=True)
df_unemployment['TIME_PERIOD'] = pd.to_numeric(df_unemployment['TIME_PERIOD'], errors='coerce')
df_unemployment['OBS_VALUE'] = pd.to_numeric(df_unemployment['OBS_VALUE'], errors='coerce')
df_unemployment['sex'] = df_unemployment['sex'].replace({'M': 'Male', 'F': 'Female'})
# Obtain the data corresponding to these indices
idx_female = df_unemployment[df_unemployment['sex'] == 'Female'].groupby('TIME_PERIOD')['OBS_VALUE'].idxmax()
df_max_female = df_unemployment.loc[idx_female].reset_index(drop=True)

# Obtain the corresponding values for men in the same country and year
df_max_male = pd.merge(df_max_female[['TIME_PERIOD', 'name']],
                       df_unemployment[df_unemployment['sex'] == 'Male'],
                       on=['TIME_PERIOD', 'name'],
                       how='left')

# Combine data in a single DataFrame
df_combined = pd.concat([df_max_female, df_max_male]).reset_index(drop=True)
df_labels = df_combined.loc[df_combined.groupby('TIME_PERIOD')['OBS_VALUE'].idxmax()]

color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

# Crear la gráfica con Altair
chart_f = alt.Chart(df_combined[df_combined['sex']=='Male']).mark_bar(size=40).encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('OBS_VALUE:Q', title='Unemployment Rate', stack=None),
    color=alt.Color('sex:N', title='Sex', scale=color_sex),
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('sex:N', title='Sex'),
        alt.Tooltip('OBS_VALUE:Q', title='Unemployment Rate', format='.2f')
    ]
)

chart_m = alt.Chart(df_combined[df_combined['sex']=='Female']).mark_bar(size=20).encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('OBS_VALUE:Q', title='Unemployment Rate', stack=None),
    color=alt.Color('sex:N', title='Sex', scale=color_sex),
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('sex:N', title='Sex'),
        alt.Tooltip('OBS_VALUE:Q', title='Unemployment Rate', format='.2f')
    ]
)

# Adding value labels only at the maximum points
text = alt.Chart(df_labels).mark_text(
    align='center',
    baseline='bottom',
    dx=0,
    dy=-10  # Set this value to position the labels correctly
).encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('OBS_VALUE:Q', title='Unemployment Rate', stack=None),
    text=alt.Text('geo:N'),
    tooltip= [alt.Tooltip("name:N", title="Region"), alt.Tooltip("TIME_PERIOD:O", title="Year"), alt.Tooltip("OBS_VALUE:Q", title="Unemployment Rate")]
)

# Show the graph in Streamlit
final_chart = (chart_f + chart_m + text).properties(
    width=800,
    height=400,
    title='Highest Female Unemployment Rate Regions by Year with Corresponding Male Rate'
)

st.altair_chart(final_chart, use_container_width=True)

st.markdown("### Employment rate")
df_erate = pd.read_csv("data/employment_rate_noms.txt", sep=",", header=None, index_col=False)
df_erate.columns = df_erate.iloc[0]
df_erate = df_erate[1:]
df_erate.columns.name = None
df_erate.reset_index(drop=True, inplace=True)
df_erate['OBS_VALUE'] = pd.to_numeric(df_erate['OBS_VALUE'], errors='coerce')
df_erate = df_erate[df_erate['sex'] != 'T']
df_female = df_erate[df_erate['sex'] == 'F']
df_male = df_erate[df_erate['sex'] == 'M']

# Calculating the gender gap for each region and year
df_gap = df_female.merge(df_male, on=['TIME_PERIOD', 'geo', 'name'], suffixes=('_F', '_M'))
df_gap['gap'] = df_gap['OBS_VALUE_M'] - df_gap['OBS_VALUE_F']

# Calculate the overall average for each year
global_mean = df_gap.groupby('TIME_PERIOD')['gap'].mean().reset_index()
global_mean['geo'] = 'Global Mean'

# Combining data from the regions with the global average
df_combined = pd.concat([df_gap, global_mean], ignore_index=True)
df_gap['Gap_Display'] = df_gap['gap'].apply(
    lambda x: f"{x:.2f}%"
)
global_mean['Gap_Display'] = global_mean['gap'].apply(
    lambda x: f"{x:.2f}%"
)

mean_lines = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#ec432c', size=2).encode(y='y:Q')

regions = df_gap['name'].unique()
regions.sort()

# Create a drop-down menu in Streamlit to select region
selected_region = st.selectbox('Select a region:', regions)

# Creating the Streamgraph with Altair
select_lines = alt.Chart(df_gap[df_gap['name'] == selected_region]).mark_line().encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('gap:Q', title='% Gender Employment Rate Gap (Male - Female)'),
    color=alt.Color('name:N', scale=alt.Scale(range=['#1E90FF']), legend=alt.Legend(title="Region")),
    detail='geo:N',
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('TIME_PERIOD:O', title='Year'),
        alt.Tooltip('Gap_Display:N', title='Employment Rate Gap'),
    ]
).properties(
    width=800,
    height=400,
    title='Gender Employment Rate Gap by Region and Year'
)

noselect_lines = alt.Chart(df_gap[df_gap['name'] != selected_region]).mark_line().encode(
    x=alt.X('TIME_PERIOD:O', title='Year', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('gap:Q', title='% Gender Employment Rate Gap (Male - Female)'),
    color=alt.value('lightgray'),
    detail='geo:N',
    tooltip=[
        alt.Tooltip('name:N', title='Region'),
        alt.Tooltip('TIME_PERIOD:O', title='Year'),
        alt.Tooltip('gap:Q', title='Employment Rate Gap'),
    ]
).properties(
    width=800,
    height=400,
    title='Gender Employment Rate Gap by Region and Year'
)

# Overall average line
mean_line = alt.Chart(global_mean).mark_line(color='black').encode(
    x=alt.X('TIME_PERIOD:O', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('gap:Q'),
    detail='geo:N',
    color=alt.Color('legend:N',  scale=alt.Scale(range=['black']), legend=alt.Legend(title=" ")),
    tooltip=[
        alt.Tooltip('TIME_PERIOD:O', title='Year'),
        alt.Tooltip('Gap_Display:N', title='Global Mean Employment Rate Gap'),
    ]
).transform_calculate(
    legend='"Global Mean"'
)

text = alt.Chart(pd.DataFrame({'x': [df_gap['TIME_PERIOD'].max()], 'y': [0], 'text': ['Parity']})).mark_text(
    align='left',
    baseline='top',
    dx=3,  # Displacement in x
    dy=5,  # Displacement in y
    fontSize=13
).encode(
    x='x:O',
    y='y:Q',
    text='text:N',
    color=alt.value('#ec432c'),
)

nearest = alt.selection_point(nearest= True, on= 'mouseover', fields= ['TIME_PERIOD'])

selectors = alt.Chart(df_gap).mark_point(color = 'gray').encode(
    x=alt.X('TIME_PERIOD:O', title='Year'),
    opacity = alt.value(0)
).add_params(
    nearest
)

rules = alt.Chart(df_gap).mark_rule(color='gray').encode(
    x='TIME_PERIOD:O',
    tooltip=[alt.Tooltip('TIME_PERIOD:O', title='Year')]
).transform_filter(
    nearest
)

points = alt.Chart(df_gap[df_gap['name'] == selected_region]).mark_point(filled=True).encode(
    x=alt.X('TIME_PERIOD:O'),
    y=alt.Y('gap:Q'),
    color=alt.Color('name:N', scale=alt.Scale(range=['#1E90FF']), legend=None),  # Evitar la leyenda duplicada
    opacity=alt.condition(nearest, alt.value(1), alt.value(0)),  # Mostrar solo cuando hay una selección
    tooltip=[alt.Tooltip('Gap_Display:N', title='Employment Rate Gap'), alt.Tooltip('name:N', title='Region'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
).transform_filter(
    nearest
)

points2 = alt.Chart(global_mean).mark_point(filled=True).encode(
    x=alt.X('TIME_PERIOD:O'),
    y=alt.Y('gap:Q'),
    detail='geo:N',
    color=alt.Color('legend:N',  scale=alt.Scale(range=['black']), legend=None),  # Evitar la leyenda duplicada
    opacity=alt.condition(nearest, alt.value(1), alt.value(0)),  # Mostrar solo cuando hay una selección
    tooltip=[alt.Tooltip('Gap_Display:N', title='Global Mean Rate Gap'), alt.Tooltip('TIME_PERIOD:O', title='Year')]
).transform_filter(
    nearest
).transform_calculate(
    legend='"Global Mean"'
)

ch = alt.layer(
     noselect_lines,select_lines, mean_lines, mean_line, text, rules, selectors, points, points2
).properties(
    width=700,
    height=500,
    title= 'Yearly Rate Difference Male and Female in Employment',
).resolve_scale(
    color='independent'
)

# Show the graph in Streamlit
st.altair_chart(ch, use_container_width=True)
