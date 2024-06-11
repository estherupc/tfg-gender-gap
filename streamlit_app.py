"""
This script creates the home page for a Gender Gap Visualization Tool using Streamlit.
The tool aims to provide an interactive platform for exploring various metrics related to gender disparities across different regions and over time.
The script is structured to include an introduction, a section on physical and violence against women in 2012, and plots about all metrics across all regions.

Esther Fanyanàs I Ropero
"""
# ------- IMPORTS -------
import streamlit as st
import altair as alt
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import plotly.graph_objects as go
import numpy as np
from color_mapping import create_color_mapping, create_abbreviation_mapping
from st_pages import Page, show_pages
# ------ FUNCTIONS ------
st.set_page_config(
        page_title="Gender Gap Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

show_pages(
    [
        Page("streamlit_app.py", "Home", "🏠"),
        Page("pages/1_📚_Education.py", "Education", "📚"),
        Page("pages/2_📊_Employment.py", "Employment", "📊"),
        Page("pages/3_🏥_Well-being.py", "Well-being", "🏥"),
        Page("pages/4_❓_Help.py", "Help", "❓"),
    ]
)

def page():
    """
    Configure the settings for the Streamlit application.
    """
    st.title("Gender Gap Visualizations")
    st.markdown("---")
    st.write("**Welcome to the Gender Gap Visualization Dashboard**. This tool is designed to provide insights and analyses on gender disparities across various regions of Europe and through differents years. Use the sidebar to navigate through different visualizations that highlight key aspects of gender equality.")
    image_path = "images/SDG_5.jpg"  # Reemplaza con la ruta a tu imagen
    image = Image.open(image_path)

    # Convert the image to base64 for embedding in HTML
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    html_code = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex: 1;">
        <img src="data:image/png;base64,{img_str}" width="150" style="margin-bottom: 10px;">
        </div>
        <div style="flex: 5; margin-left: 1px;">
            <h4> Achieve gender equality and empower all women and girls </h4>
            <h5> Goal 5 of the Sustainable Development Goals (SDGs) </h5>
            <p>
                 This goal emphasizes the need to address various dimensions of gender inequality, including access to education, economic opportunities, political representation, and health services.
            </p>
        </div>
    </div>
    """
    # Render HTML
    st.markdown(html_code, unsafe_allow_html=True)

    st.write("The **Sustainable Development Goals** (SDGs) are a set of 17 global goals established by the United Nations in 2015 as part of the 2030 Agenda for Sustainable Development. These goals are designed to address a wide range of global challenges, including poverty, inequality, climate change, environmental degradation, peace, and justice.")
    url = "https://ec.europa.eu/eurostat/databrowser/explore/all/all_themes?lang=en&display=list&sort=category"
    st.write("The data used is from the section Sustainable Development Indicators from [Eurostat](%s), the statistical office of the European Union." % url)
    st.write("The **dashboard** is organized into four distinct sections. The first section, located on this page, features two tabs: one for examining **physical and sexual violence against women by age group in 2012**, and another for **general metrics across women in all regions**. In the sidebar, you will find three additional sections, each grouping metrics related to **education**, **employment**, and **well-being**. Lastly, there is a **help guide** to assist you in navigating and understanding the dashboard.")
    st.markdown("---")

def ytick(yvalue: float, field: str, dy_adjust: float, base_chart: alt.Chart) -> alt.LayerChart:
    """
    Create a custom y-axis tick mark in an Altair chart.

    :params yvalue: The y-coordinate value where the tick mark should be placed.
    :params field: The field from which the minimum value is taken to be displayed as the label.
    :params dy_adjust: The adjustment for the vertical position of the text label.
    :params base_chart: The base Altair chart to which the tick mark should be added.
    :return: A layered Altair chart with the custom y-axis tick mark and label.
    """
    scale = base_chart.encode(x='Simp_Domain:N', y=alt.value(yvalue), text=alt.Text(f"min({field}):Q", format=".2f"))
    return alt.layer(
        scale.mark_text(baseline="middle", align="right", dx=-5, dy=dy_adjust, tooltip=None),
    )

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

def main():

    page()

    tab1, tab2 = st.tabs(["Physical and sexual violence to women by age group in 2012", "Comprehensive Analysis of Women's Status Across Key Metrics"])
    with tab1:
        st.markdown("### Physical and sexual violence to women by age group in 2012")
        st.markdown("")
        df_violence = load_data("data/violence_noms.txt")
        df_filtered = df_violence[df_violence['age'] != "15-74"]

        max_values = df_filtered.groupby('geo').OBS_VALUE.transform(max)

        # Add this information to the original DataFrame so that it can be sorted by it
        df_filtered['max_value'] = max_values
        df_filtered['is_highest'] = df_filtered.groupby('geo')['OBS_VALUE'].transform(lambda x: x == x.max())

        color_scale = alt.Scale(range=['#fdb562', '#c2df7d','#bfbbdc', '#8cd4c7',  '#bf5846'])

        filtered_df = df_violence[df_violence['age'] == "15-74"]

        # Calculate the total number of victims for the selected age range
        total_victims = filtered_df['OBS_VALUE'].mean()

        # Create a DataFrame for the donut chart
        donut_data = pd.DataFrame({
            'category': ['Victims', 'Non-victims'],
            'value': [total_victims, 100 - total_victims]
        })

        # Create the donut chart
        donut_chart = alt.Chart(donut_data).mark_arc(innerRadius=80, outerRadius = 100).encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="category", type="nominal", title='Category', scale=alt.Scale(range=["#F9B7B2", "#ec432c"])),
            tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('value:Q', title='% of Women', format='.2f')
        ]
        ).properties(
            width=260,
            height=600,
            title = alt.TitleParams(
            text='Global Women Aged 15 - 74 Victims of Violence',
            )
        )

        # Add text in the centre
        percentage_text = alt.Chart(pd.DataFrame({'text': [f'{donut_data["value"].values[0]:.2f}%']})).mark_text(
            align='center',
            baseline='middle',
            fontSize=35,
            fontWeight='bold',
            color = "#ec432c"
        ).encode(
            text='text:N',
            tooltip=[
            alt.Tooltip('text:N', title='% of Women Victims')
        ]
        ).properties(
            width=260,
            height=600
        )

        # Overlay donut graphic and text
        final_chart = alt.layer(donut_chart, percentage_text)

        url_geojson = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson"
        data_url_geojson = alt.Data(url=url_geojson, format=alt.DataFormat(property="features"))

        color_map = alt.Scale(domain=[0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], range=[ '#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'])

        base = alt.Chart(data_url_geojson).mark_geoshape(
            stroke='black',
            strokeWidth=1
        ).encode(
            color=alt.Color('OBS_VALUE:Q', scale=color_map, legend=alt.Legend(title="% of Victims")),
            tooltip=[
            alt.Tooltip('properties.NAME:N', title='Region'),
            alt.Tooltip('OBS_VALUE:Q', title='% of Women Victims', format='.2f')
        ]
        ).transform_lookup(
            lookup='properties.NAME',  # Ajusta esto a la propiedad correcta en tu GeoJSON
            from_=alt.LookupData(filtered_df, 'name', ['OBS_VALUE'])
        ).properties(
            width=600,
            height=600,
            title = alt.TitleParams(
            text='% of Women Aged 15 - 74 Victims of Violence by Region',
            align='center',
            anchor='middle'
            )
        )

        chart_global = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('age:N', sort = '-y', title = "Age Range", axis=alt.Axis(labelAngle=0) ),  # Ordenar de mayor a menor
            y=alt.Y('mean(OBS_VALUE):Q', title='% of Women Victims of Violence '),
            color=alt.Color('age:N', scale = color_scale, title = "Age Range", legend=None ),
            tooltip=[
            alt.Tooltip('age:N', title='Age Range'),
            alt.Tooltip('mean(OBS_VALUE):Q', title='% of Women Victims', format='.2f')
        ]
        ).properties(
            width=250,
            height=500,
            title = alt.TitleParams(
            text='% of Women Victims by Age Range',
            align='center',
            anchor='middle'
            )
        )

        # Create the chart with the order based on ‘max_value’.
        chart = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X('geo:N', sort=alt.EncodingSortField(field='max_value', order='descending'), title = "Region"),  # Ordenar de mayor a menor
            y=alt.Y('OBS_VALUE:Q', stack='normalize', title='% of Women Victims of Violence '),
            color=alt.Color('age:N', scale = color_scale, title = "Age Range"),
            opacity=alt.condition(
                alt.datum.is_highest == True,
                alt.value(1),  # Highlight with full opacity
                alt.value(0.5)  # Less opacity for those not featured
            ),
            tooltip=[
                alt.Tooltip('name:N', title='Region'),
                alt.Tooltip('age:N', title='Age Range'),
                alt.Tooltip('OBS_VALUE:Q', title='% of Women Victims', format='.2f')
            ]
        ).properties(
            width=550,
            height=500,
            title= alt.TitleParams(
            text='% of Women Victims of Violence by Age Group and Region',
            align='center',
            anchor='middle'
            )
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.altair_chart(base, use_container_width=True)

        with col2:
            st.altair_chart(final_chart, use_container_width=True)

        st.markdown("")
        st.markdown("")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.altair_chart(chart_global, use_container_width=True)
        with col2:
            st.altair_chart(chart, use_container_width=True)
            st.markdown("<p style='text-align: center; font-weight: normal;'>The age group with the highest percentage for each region is highlighted with greater opacity.</p>", unsafe_allow_html=True)

    with tab2:
        # Get the mappings
        st.session_state.color_mapping = create_color_mapping()
        abbreviation_mapping = create_abbreviation_mapping()

        with st.expander('## Legend of Region Colors'):
            cols = st.columns(4)  # Adjust the number of columns as needed
            color_items = list(st.session_state.color_mapping.items())
            num_items = len(color_items)
            items_per_col = 10  # Distribute items across columns

            for idx, (region, color) in enumerate(color_items):
                col_idx = idx // items_per_col
                with cols[col_idx]:
                    abbreviation = abbreviation_mapping.get(region, "")
                    st.markdown(f'{region} ({abbreviation}): ![{color}](https://via.placeholder.com/15/{color.strip("#")}/000000?text=+)')

        df_erate = load_data("data/employment_rate_noms.txt")

        df_teritary = load_data("data/teritary_educational.txt")

        df_childhood = load_data("data/childhood_noms.txt")

        df_healthy_life = load_data("data/healthy_life_noms.txt")

        df_health_good = load_data("data/health_good_noms.txt")

        df_labour_force = load_data("data/labour_force_noms.txt")

        df_leavers = load_data("data/leavers_education_noms.txt")

        df_medical = load_data("data/medical_examination_noms.txt")

        df_adult = load_data("data/adult_participation_noms.txt")

        df_unemployment = load_data("data/unemployment_noms.txt")

        df_digital = load_data("data/basic_digital_noms.txt")

        datasets = {
            "Medical Examination": df_medical,
            "Early Childhood Education": df_childhood,
            "Early Leavers": df_leavers,
            "Adult Participation": df_adult,
            "Percived Good Health": df_health_good,
            "Tertiary Education": df_teritary,
            "Employed Rate": df_erate,
            "Unemployment": df_unemployment,
            "Basic Digital Skills": df_digital,
            "Caring Responsibilities":df_labour_force,
            "Healthy Life Years": df_healthy_life
        }

        combined_data = pd.merge(df_medical, df_labour_force, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_medical', '_care'))
        combined_data = pd.merge(df_health_good, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_good', ''))
        combined_data = pd.merge(df_adult, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_adult', ''))
        combined_data = pd.merge(df_leavers, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_leavers', ''))
        combined_data = pd.merge(df_teritary, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_edu', ''))
        combined_data = pd.merge(df_erate, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_emp', ''))
        combined_data = pd.merge(df_unemployment, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_unemp', ''))
        combined_data = pd.merge(df_childhood, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_childhood', ''))
        combined_data = pd.merge(df_healthy_life, combined_data, on=['sex', 'name', 'geo', 'TIME_PERIOD'], suffixes=('_life', ''))

        combined_data['TIME_PERIOD'] = pd.to_numeric(combined_data['TIME_PERIOD'], errors='coerce')

        combined_data.rename(columns={
            'OBS_VALUE_edu': 'Tertiary Studies',
            'OBS_VALUE_childhood': 'Childhood Education',
            'OBS_VALUE_leavers': 'Education/Training Dropouts',
            'OBS_VALUE_emp': 'Employed Rate',
            'OBS_VALUE': 'Percived\nGood Health',
            'OBS_VALUE_medical': 'Unmet Need\nfor Medical Care',
            'OBS_VALUE_adult': 'Adult Learning',
            'OBS_VALUE_unemp': 'Unemployment',
            'OBS_VALUE_care': 'Labour Force',
            'OBS_VALUE_life': 'Healthy Life Years',
        }, inplace=True)

        cols = st.columns([3, 2.5])
        min_year = combined_data['TIME_PERIOD'].min()
        max_year = combined_data['TIME_PERIOD'].max()
        with cols[0]:
            selected_year = st.slider('Select a year:', min_value=min_year, max_value=max_year, value=min_year, step=1)
        st.markdown(f"### Comprehensive Analysis of Women's Status Across Key Metrics in {selected_year}")
        st.markdown("")
        # Multisector for the regions
        col1, col2 = st.columns([3.3, 2])
        with col1:
            st.markdown(
                "<h6 style='text-align: center; color: #333333;'>Comparison of Women Rates Metrics</h6>",
                unsafe_allow_html=True
            )
            regions = np.intersect1d(df_erate['name'].unique(), df_teritary['name'].unique())
            regions = np.intersect1d(df_childhood['name'].unique(), regions)
            regions = np.intersect1d(df_healthy_life['name'].unique(), regions)
            regions = np.intersect1d(df_health_good['name'].unique(), regions)
            regions = np.intersect1d(df_labour_force['name'].unique(), regions)
            regions = np.intersect1d(df_leavers['name'].unique(), regions)
            regions = np.intersect1d(df_medical['name'].unique(), regions)
            regions = np.intersect1d(df_adult['name'].unique(), regions)

            selected_regions = st.multiselect('Select regions:', options=regions, default=['Netherlands'])

            # Filter the data by the selected year and regions, and only for 'F' (female)
            filtered_data = combined_data[
                (combined_data['TIME_PERIOD'] == selected_year) &
                (combined_data['sex'] == 'F') &
                (combined_data['name'].isin(selected_regions))
            ]

            # Normalize the data
            normalized_data = combined_data.copy()
            min_max_values = {}
            for col in normalized_data.columns[4:]:
                normalized_data[col] = pd.to_numeric(normalized_data[col], errors='coerce')
                min_max_values[col] = (normalized_data[col].min(), normalized_data[col].max())
                min_value, max_value = min_max_values[col]
                normalized_data[col] = (normalized_data[col] - min_value) / (max_value - min_value)

            filtered_data = normalized_data[
                (normalized_data['TIME_PERIOD'] == selected_year) &
                (normalized_data['sex'] == 'F') &
                (normalized_data['name'].isin(selected_regions))
            ]

            # Plotting
            labels = filtered_data.columns[4:]  # Adjust according to your data
            num_vars = len(labels)

            fig = go.Figure()

            for _, row in filtered_data.iterrows():
                data = row[4:].tolist()
                data += data[:1]  # Make the plot circular
                fig.add_trace(go.Scatterpolar(
                    r=data,
                    theta=labels.tolist() + [labels[0]],
                    fill='toself',
                    name=row['name'],
                    line_color=st.session_state.color_mapping[row['name']],
                    hovertemplate='<b>Region:</b> %{text}<br><b>Metric:</b> %{theta}<br><b>Rate:</b> %{r:.2f}',
                    text=[row['name']] * (num_vars + 1)
                ))

            fig.update_layout(
                width=600,  # Increase width
                height=600,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['0.2', '0.4', '0.6', '0.8', '1.0']
                    ),
                ),
                showlegend=True,
                title=''
            )

            st.plotly_chart(fig, use_container_width=True)


        ############################################
        ###### PARALLEL COORDINATE CHART ###########
        ############################################
        with col2:
            for column in combined_data.columns[4:]:
                combined_data[column] = pd.to_numeric(combined_data[column], errors='coerce')

            df_last_year = combined_data[combined_data["TIME_PERIOD"] == selected_year]
            # Group by region, name and time period
            grouped = df_last_year.groupby(['geo', 'name', 'TIME_PERIOD'])

            # Calculating the difference between men and women for each metric
            def calculate_differences(group):
                if 'F' in group['sex'].values and 'M' in group['sex'].values:
                    female_values = group.loc[group['sex'] == 'F'].iloc[0, 4:]
                    male_values = group.loc[group['sex'] == 'M'].iloc[0, 4:]
                    return male_values - female_values
                else:
                    return pd.Series([None] * (len(group.columns) - 4), index=group.columns[4:])

            differences = grouped.apply(calculate_differences)

            # Resetting the index to make it easier to work with the resulting data
            differences = differences.reset_index()

            # Rename columns to reflect that they are differences
            differences.columns = ['geo', 'name', 'TIME_PERIOD'] + [f'{col}_Difference' for col in combined_data.columns[4:]]

            # Calculate an overall difference measure (sum of the absolute differences)
            differences['Total_Difference'] = differences.iloc[:, 3:].sum(axis=1)

            # Order countries by overall measure of difference
            sorted_differences = differences.sort_values(by='Total_Difference', ascending=False)

            top_3_differences = sorted_differences.head(3)
            bottom_3_differences = sorted_differences.tail(3)

            # Concatenate results for visualisation
            top_bottom_differences = pd.concat([top_3_differences, bottom_3_differences])

            top_bottom_differences['Difference'] = top_bottom_differences['Total_Difference'].apply(lambda x: 'Male > Female' if x > 0 else 'Female > Male')


            # Create the bar chart
            bar_chart = alt.Chart(top_bottom_differences).mark_bar().encode(
                x=alt.X('geo:N', title='Region', sort='-y', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Total_Difference:Q', title='Percentage Difference'),
                color=alt.Color('Difference:N', scale=alt.Scale(domain=['Male > Female', 'Female > Male'], range=['#7fbf7b', '#af8dc3']), legend=alt.Legend(title="Difference")),
                tooltip=[
                alt.Tooltip('name:N', title='Region'),
                alt.Tooltip('Total_Difference:Q', title='Difference'), ]
            ).properties(
                width=150,
                height=690,
                title='Top - Bottom Regions Gender Differences (M-F)'
            )

            # Show the graph in Streamlit
            st.altair_chart(bar_chart, use_container_width=True)



        df_last_year = combined_data[combined_data["TIME_PERIOD"] == selected_year]
        # Calculate the mean for each gender category of interest
        numeric_columns = df_last_year.select_dtypes(include='number').columns
        df_numeric = df_last_year[['sex'] + list(numeric_columns)]

        # Calculate mean only for numerical columns by gender
        df_mean = df_numeric.groupby('sex').mean().reset_index()

        columns_of_interest = [
            'Employed Rate',
            'Tertiary Studies',
            'Percived\nGood Health',
            'Adult Learning',
            'Education/Training Dropouts',
            'Childhood Education',
            'Unemployment',
            'Labour Force',
            'Healthy Life Years',
        ]
        simplified_columns = {
            'Employed Rate': 'Employment',
            'Tertiary Studies': 'Tertiary Studies',
            'Percived\nGood Health': 'Good Health',
            'Adult Learning': 'Adult Learning',
            'Education/Training Dropouts': 'Early Leavers',
            'Childhood Education': 'Early Education',
            'Unemployment': 'Unemployment',
            'Labour Force': 'Labour Force',
            'Healthy Life Years': 'Life Years',
        }
        df_mean = df_mean[df_mean['sex'] != 'T']
        df_parallel = df_mean[['sex'] + columns_of_interest]
        df_parallel_melted = df_parallel.melt(id_vars=['sex'], var_name='Metric', value_name='Value')
        # Create the parallel coordinate chart
        df_parallel_melted['sex'] = df_parallel_melted['sex'].replace({'M': 'Male', 'F': 'Female'})
        df_parallel_melted['Simp_Domain'] = df_parallel_melted['Metric'].map(simplified_columns)

        color_sex = alt.Scale(domain=['Female', 'Male'], range=['#af8dc3', '#7fbf7b'])

        # Create a base chart
        base = alt.Chart(df_parallel_melted).transform_fold(
            columns_of_interest
        ).transform_joinaggregate(
            min="min(Value)",
            max="max(Value)",
            groupby=["Simp_Domain"]
        ).transform_calculate(
            norm_val="(datum.Value - datum.min) / (datum.max - datum.min)"
        ).properties(
            width=800,  # Adjusting the width to take up more space
            height=400
        )

        # Create the lines of the chart
        lines = base.mark_line(point=True).encode(
            x=alt.X('Simp_Domain:N', title='Metric'),  # Show X-axis title
            y=alt.Y('norm_val:Q', axis=alt.Axis(title='Percentage', labels=False, ticks=False, domain=False)),  # Mostrar el título y las etiquetas del eje Y
            color=alt.Color('sex:N', scale=color_sex, legend=alt.Legend(title='Sex')),
            detail='sex:N',
            tooltip=[
            alt.Tooltip('Metric:N'),
            alt.Tooltip('Value:Q', title='Percentage', format=".2f"),
            alt.Tooltip('sex:N', title='Sex')]
        )

        rules = base.mark_rule(color="#ccc", tooltip=None).encode(
            x="Simp_Domain:N",
            detail="count():Q"
        )

        # Create and configure the complete chart
        parallel_coordinates_plot = alt.layer(
            lines, rules, ytick(0, "max", 10, base), ytick(300, "min", -100, base)
        ).configure_axisX(
            domain=True,  # Show the X-axis domain
            tickColor="#ccc",
        ).configure_view(
            stroke=None
        ).properties(
            title=f'Gender Metrics Averages for All Regions in {selected_year}'
        )

        st.altair_chart(parallel_coordinates_plot, use_container_width=True)


if __name__ == '__main__':
    main()
