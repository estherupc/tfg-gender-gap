"""
This script generates the help guide page for the gender gap visualization tool using Streamlit.
The help page provides users with detailed explanations of the various chart types available in the tool,
instructions on how to interpret them, and how to navigate through the tool effectively.

Esther Fanyanàs I Ropero
"""

# ------- IMPORTS -------
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import streamlit_antd_components as sac

# -------- FUNCTIONS ---------
st.set_page_config(layout="wide")

st.title("Help Guide")
st.write("-----------")

with st.expander("How the Tool Works"):
    st.write("""
    Welcome to our tool! This guide will help you understand how to use the tool and interpret the various charts and graphs.
    """)
    st.write("""
    The tool is designed to provide insightful gender gap visualizations. You can explorer the data, select regions, years across different metrics.
    """)
    st.image("images/1.gif")
    st.write("""
    You will see that there are different boxes: sliders to select years, multiple select boxes where you can pick and quit the regions you want or single select boxes. Hovering the mouse over the points you will see the exactly information of that value.
    """)
    st.image("images/2.gif")
    st.write("""
    There are charts that are more interactive and even you have to click over them to see another chart, try and explore to discover information ;)
    """)

with st.expander("Understanding the Charts"):
    st.write("""
    We provide several types of charts to help you visualize your data:
    """)

    # Define the items for the vertical tabs
    menu_items = [
        sac.TabsItem(label='Line Chart'),
        sac.TabsItem(label='Bar Chart'),
        sac.TabsItem(label='Lollipop Chart'),
        sac.TabsItem(label='Donut Chart'),
        sac.TabsItem(label='Scatter Plot'),
        sac.TabsItem(label='Radar Chart'),
        sac.TabsItem(label='Bubble Chart'),
        sac.TabsItem(label='Violin Chart'),
        sac.TabsItem(label='Area Chart'),
        sac.TabsItem(label='Heatmap Chart'),
        sac.TabsItem(label='Choropleth Map'),
        sac.TabsItem(label='Parallel Coordinate Chart'),
        sac.TabsItem(label='Pictogram Chart')
    ]

    # Create a two-column layout
    col1, col2 = st.columns([1, 3])

    # Display the tabs in the left column
    with col1:
        selected_tab = sac.tabs(
            items=menu_items,
            position="left",
            size=12
        )

    # Display the content for each tab in the right column
    with col2:
        if selected_tab == 'Line Chart':
            st.markdown("### Line Chart")
            st.write("""
            A Line Chart is used to display quantitative values over a continuous interval or time period, it is most frequently used to show trends and analyse how the data has changed over time. Line Charts are drawn by first plotting data points on a Cartesian coordinate grid, and then connecting a line between all of these points. Typically, the y-axis has a quantitative value, while the x-axis is a timescale or a sequence of intervals. Negative values can be displayed below the x-axis.
            """)
            # Sample Data
            data = pd.DataFrame({
                'X': [1, 2, 3, 4, 5],
                'Y': [10, 20, 15, 25, 30]
            })

            # Create the line chart
            line_chart = alt.Chart(data).mark_line(point=True).encode(
                x=alt.X('X', title='X-axis'),
                y=alt.Y('Y', title='Y-axis'),
                tooltip=['X', 'Y']
            ).properties(
                width=600,
                height=400,
                title='Line Chart Example'
            )

            st.altair_chart(line_chart, use_container_width=True)

        if selected_tab == 'Bar Chart':
            st.markdown("### Bar Chart")
            st.write("""
            A Bar Chart uses either horizontal or vertical bars (column chart) to show discrete, numerical comparisons across categories. One axis of the chart shows the specific categories being compared and the other axis represents a discrete value scale.
            """)
            data = pd.DataFrame({
                'Category': ['Category A', 'Category B', 'Category C', 'Category D'],
                'Value': [10, 24, 36, 48]
            })

            # Creating the bar chart
            bar_chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Category', axis=alt.Axis(labelAngle=0)),
                y='Value',
                color='Category'
            ).properties(
                title='Bar Chart Example'
            )
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("#### Normalized Stacked Bar Chart")

            st.markdown("A normalized stacked bar chart is a bar chart where each bar represents multiple variables stacked on top of each other, but the total height of each bar is scaled to 100%, allowing for easy comparison of relative proportions within categories.")

            # Creating the bar chart
            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D', 'E'] * 3,
                'Subcategory': ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
                'Value': [10, 15, 20, 25, 30, 20, 10, 15, 20, 15, 25, 35, 20, 30, 25]
            })

            # Create the stacked normalized bar chart
            stacked_normalized_chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Category:N', title='Category', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Value:Q', stack='normalize', title='Proportion'),
                color=alt.Color('Subcategory:N', legend=alt.Legend(title="Subcategory")),
                tooltip=['Category:N', 'Subcategory:N', 'Value:Q']
            ).properties(
                width=600,
                height=400,
                title='Stacked Normalized Bar Chart Example'
            )

            st.altair_chart(stacked_normalized_chart, use_container_width=True)

        if selected_tab == 'Lollipop Chart':
            st.markdown("### Lollipop Chart")
            st.write("""
            Lollipop charts are a variation of a bar chart that combines the simplicity of bar charts with the clarity of scatter plots. It is used to compare categorical data points in a visually appealing and straightforward manner. Each data point is represented by a "lollipop" consisting of a line (stick) topped with a dot (candy) at the end, indicating the value.
            """)

            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D', 'E'],
                'Value': [10, 24, 36, 48, 60]
            })

            # Create the base chart with lines
            base = alt.Chart(data).mark_rule().encode(
                x=alt.X('Category:N', title='Category', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Value:Q', title='Value'),
                color=alt.value('blue')
            )

            # Add points to the end of the lines
            points = base.mark_point(size=100, filled=True).encode(
                color=alt.value('blue')  # Color of the points
            )

            # Combine the base and points
            lollipop_chart = base + points

            # Set the properties of the chart
            lollipop_chart = lollipop_chart.properties(
                width=600,
                height=400,
                title='Lollipop Chart Example'
            )

            st.altair_chart(lollipop_chart, use_container_width=True)

        if selected_tab == 'Donut Chart':

            st.markdown("### Donut Chart")
            st.write("""
            A Donut Chart is essentially a Pie Chart (help show proportions and percentages between categories, by dividing a circle into proportional segments. Each arc length represents a proportion of each category, while the full circle represents the total sum of all the data, equal to 100%), but with the area of the centre cut out.
            """)

            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D', 'E'],
                'Value': [20, 30, 25, 15, 10]
            })

            # Calculate the percentage of each category
            data['Percentage'] = data['Value'] / data['Value'].sum() * 100

            # Create the donut chart
            donut_chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Value", type="quantitative"),
                color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Category")),
                tooltip=['Category', 'Value', 'Percentage']
            ).properties(
                width=400,
                height=400,
                title='Donut Chart Example'
            )

            st.altair_chart(donut_chart, use_container_width=True)

        if selected_tab == 'Scatter Plot':

            st.markdown("### Scatter Plot")
            st.write("""
            A Scatter Plot places points on a Cartesian Coordinates system to display all the values between two variables. By having an axis for each variable, you can detect if a relationship or correlation between the two exists.""")

            data = pd.DataFrame({
                'X': [5, 7, 8, 7, 2, 17, 2, 9, 4, 11],
                'Y': [99, 86, 87, 88, 100, 86, 103, 87, 94, 78],
                'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']  # Additional category for color
            })

            # Create the scatter plot
            scatter_plot = alt.Chart(data).mark_point(size=160, filled=True).encode(
                x=alt.X('X', title='X-axis'),
                y=alt.Y('Y', title='Y-axis'),
                color=alt.Color('Category', legend=alt.Legend(title="Category")),
                tooltip=['X', 'Y', 'Category']
            ).properties(
                width=600,
                height=400,
                title='Scatter Plot Example'
            )

            st.altair_chart(scatter_plot, use_container_width=True)

        if selected_tab == 'Radar Chart':

            st.markdown("### Radar Chart")
            st.write("""
            A Radar Chart is a way of comparing multiple quantitative variables. This makes them useful for seeing which variables have similar values or if there are any outliers amongst each variable. Radar Charts are also useful for seeing which variables are scoring high or low within a dataset, making them suited for displaying performance.
            """)

            data = pd.DataFrame({
                'Variable': ['A', 'B', 'C', 'D', 'E'],
                'Value1': [4, 3, 2, 5, 4],
                'Value2': [5, 4, 3, 2, 3]
            })

            # Create the radar chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=data['Value1'],
                theta=data['Variable'],
                fill='toself',
                name='Group 1'
            ))

            fig.add_trace(go.Scatterpolar(
                r=data['Value2'],
                theta=data['Variable'],
                fill='toself',
                name='Group 2'
            ))

            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=True,
                title='Radar Chart Example'
            )

            st.plotly_chart(fig)

        if selected_tab == 'Bubble Chart':

            st.markdown("### Bubble Chart")
            st.write("""
            Bubble Charts use a Cartesian coordinate system to plot points along a grid where the X and Y axis are separate variables. However, each point is assigned a label or category (either displayed alongside or on a legend). Each plotted point then represents a third variable by the area of its circle. Colours can also be used to distinguish between categories or used to represent an additional data variable.
            """)
            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D', 'E'],
                'X Value': [10, 15, 13, 17, 10],
                'Y Value': [22, 25, 30, 35, 40],
                'Bubble Size': [100, 150, 200, 250, 300]
            })

            # Bubble chart
            bubble_chart = alt.Chart(data).mark_circle().encode(
                x=alt.X('X Value:Q', title='X Axis Label'),
                y=alt.Y('Y Value:Q', title='Y Axis Label'),
                size=alt.Size('Bubble Size:Q', title='Bubble Size'),
                color=alt.Color('Category:N', title='Category'),
                tooltip=[alt.Tooltip('Category:N', title='Category'),
                         alt.Tooltip('X Value:Q', title='X Value'),
                         alt.Tooltip('Y Value:Q', title='Y Value'),
                         alt.Tooltip('Bubble Size:Q', title='Bubble Size')]
            ).properties(
                title='Example Bubble Chart',
                width=600,
                height=400
            )

            # Display the chart in Streamlit
            st.altair_chart(bubble_chart, use_container_width=True)

        if selected_tab == 'Violin Chart':

            st.markdown("### Violin Chart")
            st.write("""
            A Violin Plot is used to visualise the distribution of the data and its probability density, that is rotated and placed on each side (to show the distribution shape of the data). The thin line extending from it represents the upper (max) and lower (min) adjacent values in the data.
            """)
            # Sample data
            data = pd.DataFrame({
             'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
             'Value': [10, 15, 13, 17, 19, 20, 10, 13, 14]
            })

            # Create violin plot
            fig = go.Figure()

            categories = data['Category'].unique()
            for category in categories:
             fig.add_trace(go.Violin(
                 x=data['Category'][data['Category'] == category],
                 y=data['Value'][data['Category'] == category],
                 name=category,
                 box_visible=True,
                 meanline_visible=True
             ))

            # Update layout
            fig.update_layout(
             title='Example Violin Chart',
             yaxis_title='Value',
             xaxis_title='Category',
             width=800,
             height=600
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        if selected_tab == 'Area Chart':

            st.markdown("### Area Chart")
            st.write("""
            Area Charts are Line Charts but with the area below the line filled in with a certain colour or texture. Area Graphs are drawn by first plotting data points on a Cartesian coordinate grid, joining a line between the points and finally filling in the space below the completed line.
            """)

            # Sample data
            data = pd.DataFrame({
                'Year': [2015, 2016, 2017, 2018, 2019, 2020],
                'Category': ['A', 'A', 'A', 'A', 'A', 'A'],
                'Value': [10, 15, 13, 17, 10, 23]
            })

            # Area chart
            area_chart = alt.Chart(data).mark_area(opacity=0.7).encode(
                x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Value:Q', title='Value'),
                color=alt.Color('Category:N', title='Category'),
                tooltip=[alt.Tooltip('Year:O', title='Year'),
                         alt.Tooltip('Value:Q', title='Value')]
            ).properties(
                title='Example Area Chart',
                width=600,
                height=400
            )

            # Display the chart in Streamlit
            st.altair_chart(area_chart, use_container_width=True)

        if selected_tab == 'Heatmap Chart':

            st.markdown("### Heatmap Chart")
            st.write("""
            Heatmaps visualise data through variations in colouring. When applied to a tabular format, Heatmaps are useful for cross-examining multivariate data, through placing variables in the rows and columns and colouring the cells within the table. Heatmaps are good for showing variance across multiple variables, revealing any patterns, displaying whether any variables are similar to each other, and for detecting if any correlations exist.
            """)
            # Sample data
            data = pd.DataFrame({
                'X': np.tile(np.arange(1, 11), 10),
                'Y': np.repeat(np.arange(1, 11), 10),
                'Value': np.random.randn(100)
            })

            # Heatmap
            heatmap = alt.Chart(data).mark_rect().encode(
                x=alt.X('X:O', title='X Axis', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Y:O', title='Y Axis'),
                color=alt.Color('Value:Q', scale=alt.Scale(scheme='viridis'), title='Value'),
                tooltip=[alt.Tooltip('X:O', title='X'),
                         alt.Tooltip('Y:O', title='Y'),
                         alt.Tooltip('Value:Q', title='Value', format='.2f')]
            ).properties(
                title='Example Heatmap',
                width=600,
                height=400
            )

            # Display the chart in Streamlit
            st.altair_chart(heatmap, use_container_width=True)

        if selected_tab == 'Choropleth Map':

            st.markdown("### Choropleth Map")
            st.markdown("Choropleth Maps display divided geographical areas or regions that are coloured, shaded or patterned in relation to a data variable. This provides a way to visualise values over a geographical area, which can show variation or patterns across the displayed location. The data variable uses colour progression to represent itself in each region of the map. Typically, this can be a blending from one colour to another, a single hue progression, transparent to opaque, light to dark or an entire colour spectrum.")

            # URL for the GeoJSON data
            url_geojson = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson"
            data_url_geojson = alt.Data(url=url_geojson, format=alt.DataFormat(property="features", type="json"))

            # Sample data for European countries
            df = pd.DataFrame({
                'country': ['Germany', 'France', 'United Kingdom', 'Italy', 'Sweden', 'Spain', 'Portugal'],  # Germany, France, UK, Italy, Sweden, Spain, Portugal (ISO 3166-1 alpha-3)
                'value': [300, 200, 100, 150, 250, 180, 220]
            })

            # Create a Choropleth map
            choropleth_map = alt.Chart(data_url_geojson).mark_geoshape().encode(
                color=alt.Color('value:Q', scale=alt.Scale(scheme='blues'), title='Value'),
                tooltip=[
                    alt.Tooltip('properties.NAME:N', title='Country'),
                    alt.Tooltip('value:Q', title='Value')
                ]
            ).transform_lookup(
                lookup='properties.NAME',
                from_=alt.LookupData(df, 'country', ['value'])
            ).properties(
                width=800,
                height=600,
                title='Example Choropleth Map of Europe'
            ).project(
                type='mercator'
            )

            # Display the chart in Streamlit
            st.altair_chart(choropleth_map, use_container_width=True)

        if selected_tab == 'Parallel Coordinate Chart':

            st.markdown("### Parallel Coordinate Chart")
            st.write("""
            Parallel coordinate charts are a type of data visualization used to plot multivariate numerical data. It allows for the comparison of many variables simultaneously, making it particularly useful for understanding complex datasets. In this chart, each variable is given its own axis, and all the axes are parallel to each other. Each data point is represented as a line that intersects each axis at the appropriate value for that variable.
            """)
            # Sample data
            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D', 'E'],
                'Metric 1': [10, 30, 30, 40, 50],
                'Metric 2': [15, 25, 45, 35, 55],
                'Metric 3': [20, 20, 60, 50, 40],
                'Metric 4': [25, 35, 35, 55, 65]
            })

            # Melt the DataFrame to long format
            data_melted = data.melt(id_vars=['Category'], var_name='Metric', value_name='Value')

            # Define columns of interest
            columns_of_interest = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4']

            # Transform and normalize data
            base = alt.Chart(data_melted).transform_fold(
                columns_of_interest
            ).transform_joinaggregate(
                min="min(Value)",
                max="max(Value)",
                groupby=["Metric"]
            ).transform_calculate(
                norm_val="(datum.Value - datum.min) / (datum.max - datum.min)"
            ).properties(
                width=800,
                height=400
            )

            # Create the lines of the chart
            lines = base.mark_line(point=True).encode(
                x=alt.X('Metric:N', title='Metric', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('norm_val:Q', axis=alt.Axis(title='Normalized Value', labels=True, ticks=True, domain=True)),
                color=alt.Color('Category:N', title='Category'),
                detail='Category:N',
                tooltip=[
                    alt.Tooltip('Category:N', title='Category'),
                    alt.Tooltip('Value:Q', title='Value', format=".2f"),
                    alt.Tooltip('Metric:N', title='Metric')
                ]
            )

            rules = base.mark_rule(color="#ccc", tooltip=None).encode(
                x="Metric:N",
                detail="count():Q"
            )

            # Create and configure the complete chart
            parallel_coordinates_plot = alt.layer(
                lines, rules
            ).configure_axisX(
                domain=True,
                tickColor="#ccc",
            ).configure_view(
                stroke=None
            ).properties(
                title='Example Parallel Coordinate Chart'
            )

            # Display the chart in Streamlit
            st.altair_chart(parallel_coordinates_plot, use_container_width=True)

        if selected_tab == 'Pictogram Chart':

            st.markdown("### Pictogram Chart")
            st.write("""Pictogram Charts use icons to give a more engaging overall view of small sets of discrete data. Typically, the icons represent the data’s subject or category, for example, data on population would use icons of people. Each icon can represent one unit or any number of units (e.g. each icon represents 10).""")
            # Sample data
            data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D'],
                'Count': [5, 3, 8, 6]
            })

            # Create a grid for pictogram
            data['dummy'] = 1
            data = data.loc[data.index.repeat(data['Count'])]

            data['X'] = [i % 10 for i in range(len(data))]
            data['Y'] = [i // 10 for i in range(len(data))]

            # Plotting
            pictogram = alt.Chart(data).mark_text(
                align='center',
                baseline='middle',
                fontSize=30,  # Adjust the size of the pictograms
                text='■'  # Square as a simple pictogram
            ).encode(
                x=alt.X('X:O', axis=None),
                y=alt.Y('Y:O', axis=None),
                color=alt.Color('Category:N', legend=alt.Legend(title='Category')),
                tooltip=['Category:N', 'Count:Q']
            ).properties(
                width=400,
                height=200,
                title='Pictogram Chart'
            )

            # Display the chart in Streamlit
            st.altair_chart(pictogram, use_container_width=True)

with st.expander("Understanding the Data"):

        menu_datasets = [
            'Physical and sexual violence to women by age group in 2012',
            'Population aged 25-34 who have successfully completed tertiary studies',
            'Population aged 18 to 24 not involved in education or training.',
            'People aged 16 to 74 who have at least basic digital skills',
            'Participation in early childhood education',
            'Seats held by women in national parliaments and governments',
            'Positions held by women in senior management positions',
            'Persons outside the labour force due to caring responsibilities',
            'Gender pay gap in unadjusted form',
            'Gender employment gap by type of employment',
            'Long-term unemployment',
            'Employment rate',
            'Healthy life years at birth',
            'Perceived good health',
            'Smoking prevalence',
            'Fatal accidents at work per 100 000 workers',
            'Self-reported unmet need for medical examination and care',
            'Standardised death rate due to homicide'
        ]

        selected_dataset = st.selectbox('Select a dataset:', menu_datasets)

        if selected_dataset == 'Physical and sexual violence to women by age group in 2012':

            st.markdown("##### Physical and sexual violence to women by age group in 2012 ")
            st.write("The indicator measures the share of women from the age of 15 who answered \"yes\" when they were asked whether they have experienced physical and/or sexual violence by a partner or non-partner in the 12 months prior to the interview. The results of the survey on violence against women are based on face-to-face interviews with 42 000 women in all 28 EU Member States, with on average 1 500 interviews per Member State. Women were asked to provide information about their personal experience of various forms of violence. Partners include persons with whom women are or have been married, living together without being married or involved in a relationship without living together. Non-partners include all perpetrators other than women’s current or previous partner. The indicator is based on a survey of the European Union Agency for Fundamental Rights (FRA).")

        if selected_dataset == 'Population aged 25-34 who have successfully completed tertiary studies':
            st.markdown("##### Population aged 25-34 who have successfully completed tertiary studies")
            st.write("The indicator measures the share of the population aged 25-34 who have successfully completed tertiary studies (e.g. university, higher technical institution, etc.). This educational attainment refers to ISCED (International Standard Classification of Education) 2011 level 5-8 for data from 2014 onwards and to ISCED 1997 level 5-6 for data up to 2013. The indicator is based on the EU Labour Force Survey (EU-LFS).")

        if selected_dataset == 'Population aged 18 to 24 not involved in education or training.':
            st.markdown("##### Population aged 18 to 24 not involved in education or training.")
            st.write("The indicator measures the share of the population aged 18 to 24 with at most lower secondary education who were not involved in any education or training during the four weeks preceding the survey. Lower secondary education refers to ISCED (International Standard Classification of Education) 2011 level 0-2 for data from 2014 onwards and to ISCED 1997 level 0-3C short for data up to 2013. Data stem from the EU Labour Force Survey (EU-LFS).")

        if selected_dataset == 'People aged 16 to 74 who have at least basic digital skills':
            st.markdown("##### People aged 16 to 74 who have at least basic digital skills")
            st.write("This indicator measures the share of people aged 16 to 74 who have at least basic digital skills. It is a composite indicator based on selected activities performed by individuals on the internet in specific areas: information and data literacy, communication and collaboration, digital content creation, safety and problem solving. The indicator assesses digital skills classified into six levels, of which the two highest constitute the basic or above basic level of digital skills. The indicator is based on data from the EU survey on the use of ICT in households and by individuals.")

        if selected_dataset == 'Participation in early childhood education':
            st.markdown("##### Participation in early childhood education")
            st.write("The indicator measures the share of the children between the age of three and the starting age of compulsory primary education who participated in early childhood education and care (ECEC) which can be classified as ISCED level 0 according to the International Standard Classification for Education (ISCED 2011). In order for ECEC programmes to be classified as ISCED level 0, they must be intentionally designed to support a child’s cognitive, physical and socio-emotional development. The starting age of compulsory primary education varies across countries. The participation of children in programmes which are not intentionally designed to support child development (such as childcare-only programmes) is not included in this indicator.")

        if selected_dataset == 'Seats held by women in national parliaments and governments':
            st.markdown("##### Seats held by women in national parliaments and governments")
            st.write("The indicator measures the proportion of women in national parliaments and national governments. The national parliament is the national legislative assembly and the indicator refers to both chambers (lower house and an upper house, where relevant). The count of members of a parliament includes the president/speaker/leader of the parliament. The national government is the executive body with authority to govern a country or a state. Members of government include both senior ministers (having a seat in the cabinet or council of ministers, including the prime minister) and junior ministers (not having a seat in the cabinet). In some countries state-secretaries (or the national equivalent) are considered as junior ministers within the government (with no seat in the cabinet) but in other countries they are not considered as members of the government. The data stem from the Gender Statistics Database of the European Institute for Gender Equality (EIGE).")

        if selected_dataset == 'Positions held by women in senior management positions':
            st.markdown("##### Positions held by women in senior management positions")
            st.write("The indicator measures the share of female board members and executives in the largest publicly listed companies. Publicly listed means that the shares of the company are traded on the stock exchange. The ‘largest’ companies are taken to be the members (max. 50) of the primary blue-chip index, which is an index maintained by the stock exchange and covers the largest companies by market capitalisation and/or market trades. Only companies which are registered in the country concerned are counted. Board members cover all members of the highest decision-making body in each company (i.e. chairperson, non-executive directors, senior executives and employee representatives, where present). Executives refer to senior executives in the two highest decision-making bodies of the largest (max. 50) nationally registered companies listed on the national stock exchange. The two highest decision-making bodies are usually referred to as the supervisory board and the management board (in case of a two-tier governance system) and the board of directors and executive/management committee (in a unitary system). The data comes from the Gender Statistics Database of the European Institute for Gender Equality (EIGE).﻿")

        if selected_dataset == 'Persons outside the labour force due to caring responsibilities':
            st.markdown("##### Persons outside the labour force due to caring responsibilities")
            st.write("The population outside the labour force comprises individuals who are not employed and are either not actively seeking work or not available to work (even if they have found a job that will start in the future). Therefore, they are neither employed nor unemployed. This definition used in the EU Labour Force Survey (EU-LFS) is based on the resolutions of the International Conference of Labour Statisticians (ICLS) organised by the International Labour Organization (ILO). The reason for being outside the labour force covered by this indicator includes ‘care of adults with disabilities or children’. Only people who express willingness to work, despite being outside the labour force, are considered.")

        if selected_dataset == 'Gender pay gap in unadjusted form':
            st.markdown("##### Gender pay gap in unadjusted form")
            st.write("The indicator measures the difference between average gross hourly earnings of male paid employees and of female paid employees as a percentage of average gross hourly earnings of male paid employees. The indicator has been defined as unadjusted, because it gives an overall picture of gender inequalities in terms of pay and measures a concept which is broader than the concept of equal pay for equal work. All employees working in firms with ten or more employees, without restrictions for age and hours worked, are included.")

        if selected_dataset == 'Gender employment gap by type of employment':
            st.markdown("##### Gender employment gap by type of employment")
            st.write("The indicator measures the difference between the employment rates of men and women aged 20 to 64. The employment rate is calculated by dividing the number of persons aged 20 to 64 in employment by the total population of the same age group. The indicator shows activity and employment status for four groups of persons: employed persons working full time, employed persons working part time, employed persons with temporary contract and underemployed persons working part time. The indicator is based on the EU Labour Force Survey.")

        if selected_dataset == 'Long-term unemployment':
            st.markdown("##### Long-term unemployment")
            st.write("The indicator measures the share of the economically active population aged 15 to 74 who has been unemployed for 12 months or more. Unemployed persons are defined as all persons who were without work during the reference week, were currently available for work and were either actively seeking work in the last four weeks or had already found a job to start within the next three months. The unemployment period is defined as the duration of a job search, or as the length of time since the last job was held (if shorter than the time spent on a job search). The economically active population comprises employed and unemployed persons. The indicator is part of the adjusted, break-corrected main indicators series and should not be compared with the annual and quarterly non-adjusted series, which have slightly different results.")

        if selected_dataset == 'Employment rate':
            st.markdown("##### Employment rate")
            st.write("The indicator measures the share of the population aged 20 to 64 which is employed. Employed persons are defined as all persons who, during a reference week, worked at least one hour for pay or profit or were temporarily absent from such work. The indicator is part of the adjusted, break-corrected main indicators series and should not be compared with the annual and quarterly non-adjusted series, which have slightly different results.")

        if selected_dataset == 'Healthy life years at birth':
            st.markdown("##### Healthy life years at birth")
            st.write("The indicator of healthy life years (HLY) measures the number of remaining years that a person of specific age is expected to live without any severe or moderate health problems. The notion of health problem for Eurostat's HLY is reflecting a disability dimension and is based on a self-perceived question which aims to measure the extent of any limitations, for at least six months, because of a health problem that may have affected respondents as regards activities they usually do (the so-called GALI - Global Activity Limitation Instrument foreseen in the annual EU-SILC survey). The indicator is therefore also called disability-free life expectancy (DFLE). So, HLY is a composite indicator that combines mortality data with health status data. HLY also monitor health as a productive or economic factor. An increase in healthy life years is one of the main goals for European health policy. And it would not only improve the situation of individuals but also result in lower levels of public health care expenditure. If healthy life years are increasing more rapidly than life expectancy, it means that people are living more years in better health. Please note that a revision took place in March 2012 and the whole series 2004-2010 were recalculated taking into account: i. the use of the age at interview for the GALI prevalences instead of the age of the income period (as it is traditionally done for many income and living indicators); differences with the previous calculations on outcomes and trends are minimal ii. the latest versions of the EU-SILC and Mortality data")

        if selected_dataset == 'Perceived good health':
            st.markdown("##### Perceived good health")
            st.write("The indicator is a subjective measure on how people judge their health in general on a scale from \"very good\" to \"very bad\". It is expressed as the share of the population aged 16 or over perceiving itself to be in \"good\" or \"very good\" health. The data stem from the EU Statistics on Income and Living Conditions (EU SILC). Indicators of perceived general health have been found to be a good predictor of people’s future health care use and mortality.")

        if selected_dataset == 'Smoking prevalence':
            st.markdown("##### Smoking prevalence")
            st.write("The indicator measures the share of the population aged 15 years and over who report that they currently smoke boxed cigarettes, cigars, cigarillos or a pipe. The data does not include use of other tobacco products such as electronic cigarettes and snuff. The data are collected through a Eurobarometer survey and are based on self-reports during face-to-face interviews in people’s homes.")

        if selected_dataset == 'Fatal accidents at work per 100 000 workers':
            st.markdown("##### Fatal accidents at work per 100 000 workers")
            st.write("The indicator measures the number of fatal accidents that occur during the course of work and lead to the death of the victim within one year of the accident. The incidence rate refers to the number of fatal accidents per 100 000 persons in employment. An accident at work is 'a discrete occurrence in the course of work which leads to physical or mental harm'. This includes all accidents in the course of work, whether they happen inside or outside the premises of the employer, accidents in public places or different means of transport during a journey in the course of the work (commuting accidents are excluded) and at home (such as during teleworking). It also includes cases of acute poisoning and wilful acts of other persons, if these happened during the course of the work.")

        if selected_dataset == 'Self-reported unmet need for medical examination and care':
            st.markdown("##### Self-reported unmet need for medical examination and care")
            st.write("The indicator measures the share of the population aged 16 and over reporting unmet needs for medical care due to one of the following reasons: ‘Financial reasons’, ‘Waiting list’ and ‘Too far to travel’ (all three categories are cumulated). Self-reported unmet needs concern a person’s own assessment of whether he or she needed medical examination or treatment (dental care excluded), but did not have it or did not seek it. The data stem from the EU Statistics on Income and Living Conditions (EU SILC). Note on the interpretation: The indicator is derived from self-reported data so it is, to a certain extent, affected by respondents’ subjective perception as well as by their social and cultural background. Another factor playing a role is the different organisation of health care services, be that nationally or locally. All these factors should be taken into account when analysing the data and interpreting the results.")

        if selected_dataset == 'Standardised death rate due to homicide':
            st.markdown("##### Standardised death rate due to homicide")
            st.write("The indicator measures the standardised death rate of homicide and injuries inflicted by another person with the intent to injure or kill by any means, including ‘late effects’ from assault (International Classification of Diseases (ICD) codes X85 to Y09 and Y87.1). It does not include deaths due to legal interventions or war (ICD codes Y35 and Y36). The rate is calculated by dividing the number of people dying due to homicide or assault by the total population. Data on causes of death (COD) refer to the underlying cause which - according to the World Health Organisation (WHO) - is \"the disease or injury which initiated the train of morbid events leading directly to death, or the circumstances of the accident or violence which produced the fatal injury\". COD data are derived from death certificates. The medical certification of death is an obligation in all Member States. The data are presented as standardised death rates, meaning they are adjusted to a standard age distribution in order to measure death rates independently of different age structures of populations. This approach improves comparability over time and between countries. The standardised death rates used here are calculated on the basis of the standard European population referring to the residents of the countries.")


with st.expander("Frequently Asked Questions"):
    st.write("Find answers to common questions below:")

    st.markdown("#### ")
    st.write("")

    st.markdown("#### ")
    st.write("")

    st.markdown("#### ")
    st.write("")

with st.expander("Need More Help?"):
    st.write("""
    If you need further assistance, please contact our support team at esther.fanyanas.i@estudiantat.upc.edu.
    """)