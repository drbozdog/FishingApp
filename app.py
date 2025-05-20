import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter
import re
import utils

# Set page configuration
st.set_page_config(page_title="ANPA Fishing Habitats Explorer", page_icon="🎣", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎣 ANPA Fishing Habitats Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Explore Romanian fishing habitats contracted for 2025</p>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/ANPA_habitats_contractate_2025_full.csv')
    
    # Clean data and handle multiline entries
    # Fix the 'Association' column that contains newlines
    df['Association'] = df['Association'].str.replace('\n', ' ')
    
    # Extract the association name from the "cu sediul în..." pattern
    df['Association_Name'] = df['Association'].apply(lambda x: x.split('cu sediul în')[0].strip() if isinstance(x, str) and 'cu sediul în' in x else x)
    
    # Extract county
    df['County'] = df['County'].str.strip()
    
    # Convert numeric values where applicable
    # Extract numeric values from Length_surface column using regex
    df['Length_km'] = df['Length_surface'].astype(str).str.extract(r'(\d+(?:\.\d+)?)\s*[Kk]m')
    df['Length_km'] = pd.to_numeric(df['Length_km'], errors='coerce')
    
    df['Area_ha'] = df['Length_surface'].astype(str).str.extract(r'(\d+(?:\.\d+)?)\s*[Hh]a')
    df['Area_ha'] = pd.to_numeric(df['Area_ha'], errors='coerce')
    
    # Categorize by type using the utils function
    df['Habitat_Type'] = df['Habitat'].apply(utils.categorize_habitat)
    
    # Extract start and end locations, types, and rivers from the Limits column using the utils function
    df[['start_location', 'end_location', 'start_type', 'end_type', 'start_from_spring', 'end_river']] = df.apply(
        lambda row: pd.Series(utils.extract_locations_and_types(row['Limits'])), 
        axis=1
    )
    
    # Check if ends in another river (based on presence of end_river)
    df['end_to_river'] = df['end_river'].notna()
    
    # Add geocoding information using the County column from the dataset
    df = utils.geocode_locations(df)
    
    # Ensure cache is saved before returning data
    utils.force_save_cache()
    
    return df

# Load the data
data = load_data()

# Sidebar
st.sidebar.header("Filters")

# County filter
counties = sorted(['All Counties'] + list(data['County'].dropna().unique()))
selected_county = st.sidebar.selectbox("Select County", counties)

# Habitat type filter
habitat_types = sorted(['All Types'] + list(data['Habitat_Type'].dropna().unique()))
selected_habitat_type = st.sidebar.selectbox("Select Habitat Type", habitat_types)

# Additional filters for location types
st.sidebar.header("Location Filters")
start_from_spring = st.sidebar.checkbox("Starts from spring")
ends_in_river = st.sidebar.checkbox("Ends in another river")

# Add location type filters
if 'start_type' in data.columns:
    start_types = sorted(['All Start Types'] + list(data['start_type'].dropna().unique()))
    selected_start_type = st.sidebar.selectbox("Start Location Type", start_types)

if 'end_type' in data.columns:
    end_types = sorted(['All End Types'] + list(data['end_type'].dropna().unique()))
    selected_end_type = st.sidebar.selectbox("End Location Type", end_types)

# Filter data based on selections
filtered_data = data.copy()
if selected_county != 'All Counties':
    filtered_data = filtered_data[filtered_data['County'] == selected_county]
if selected_habitat_type != 'All Types':
    filtered_data = filtered_data[filtered_data['Habitat_Type'] == selected_habitat_type]
if start_from_spring:
    filtered_data = filtered_data[filtered_data['start_from_spring'] == True]
if ends_in_river:
    filtered_data = filtered_data[filtered_data['end_to_river'] == True]
if 'start_type' in data.columns and selected_start_type != 'All Start Types':
    filtered_data = filtered_data[filtered_data['start_type'] == selected_start_type]
if 'end_type' in data.columns and selected_end_type != 'All End Types':
    filtered_data = filtered_data[filtered_data['end_type'] == selected_end_type]

# Display dataset overview
st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Habitats", len(data))
col2.metric("Counties", data['County'].nunique())
col3.metric("Habitat Types", data['Habitat_Type'].nunique())
col4.metric("Associations", data['Association_Name'].nunique())

# Display statistics for the new columns
st.markdown("<h2 class='sub-header'>Location Statistics</h2>", unsafe_allow_html=True)

# Create rows of metrics
row1_1, row1_2, row1_3, row1_4 = st.columns(4)
row1_1.metric("With Start Location", data['start_location'].notna().sum())
row1_2.metric("With End Location", data['end_location'].notna().sum())
row1_3.metric("With End River", data['end_river'].notna().sum())
row1_4.metric("Missing Both Locations", ((data['start_location'].isna()) & (data['end_location'].isna())).sum())

row2_1, row2_2, row2_3, row2_4 = st.columns(4)
row2_1.metric("Missing Start Only", ((data['start_location'].isna()) & (data['end_location'].notna())).sum())
row2_2.metric("Missing End Only", ((data['start_location'].notna()) & (data['end_location'].isna())).sum())
row2_3.metric("Starting from Spring", data['start_from_spring'].sum())
row2_4.metric("Ending in Another River", data['end_to_river'].sum())

# Location type statistics if available
if 'start_type' in data.columns and 'end_type' in data.columns:
    st.markdown("<h3 class='sub-header'>Location Types</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Start location types
        start_type_counts = data['start_type'].value_counts(dropna=False).reset_index()
        start_type_counts.columns = ['Type', 'Count']
        start_type_counts['Type'] = start_type_counts['Type'].fillna('Unspecified')
        
        fig = px.pie(
            start_type_counts,
            values='Count',
            names='Type',
            title='Start Location Types',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # End location types
        end_type_counts = data['end_type'].value_counts(dropna=False).reset_index()
        end_type_counts.columns = ['Type', 'Count']
        end_type_counts['Type'] = end_type_counts['Type'].fillna('Unspecified')
        
        fig = px.pie(
            end_type_counts,
            values='Count',
            names='Type',
            title='End Location Types',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# Display filtered data
st.markdown("<h2 class='sub-header'>Filtered Data</h2>", unsafe_allow_html=True)
st.dataframe(filtered_data)

# Visualizations
st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Habitats by County", "Habitat Types", "Area & Length Analysis", "Association Analysis", "Start/End Analysis"])

with tab1:
    # Habitats by County
    county_counts = pd.DataFrame(data['County'].value_counts()).reset_index()
    county_counts.columns = ['County', 'Count']
    
    fig = px.bar(
        county_counts.sort_values('Count', ascending=False).head(15),
        x='County',
        y='Count',
        title='Top 15 Counties by Number of Fishing Habitats',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Habitat Types Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        habitat_type_counts = pd.DataFrame(data['Habitat_Type'].value_counts()).reset_index()
        habitat_type_counts.columns = ['Habitat Type', 'Count']
        
        fig = px.pie(
            habitat_type_counts,
            values='Count',
            names='Habitat Type',
            title='Distribution of Habitat Types',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Habitat types by county
        if selected_county != 'All Counties':
            county_data = data[data['County'] == selected_county]
            habitat_type_by_county = pd.DataFrame(county_data['Habitat_Type'].value_counts()).reset_index()
            habitat_type_by_county.columns = ['Habitat Type', 'Count']
            
            fig = px.bar(
                habitat_type_by_county,
                x='Habitat Type',
                y='Count',
                title=f'Habitat Types in {selected_county} County',
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a specific county to see habitat type distribution in that county.")

with tab3:
    # Area & Length Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary statistics for area (Ha)
        st.markdown("<h3>Area Statistics (Ha)</h3>", unsafe_allow_html=True)
        area_stats = data['Area_ha'].describe().reset_index()
        area_stats.columns = ['Statistic', 'Value']
        st.dataframe(area_stats)
        
        # Top 10 largest areas
        top_areas = data.dropna(subset=['Area_ha']).sort_values('Area_ha', ascending=False).head(10)[['County', 'Habitat', 'Area_ha']]
        st.markdown("<h4>Top 10 Largest Areas (Ha)</h4>", unsafe_allow_html=True)
        st.dataframe(top_areas)
    
    with col2:
        # Summary statistics for length (Km)
        st.markdown("<h3>Length Statistics (Km)</h3>", unsafe_allow_html=True)
        length_stats = data['Length_km'].describe().reset_index()
        length_stats.columns = ['Statistic', 'Value']
        st.dataframe(length_stats)
        
        # Top 10 longest habitats
        top_lengths = data.dropna(subset=['Length_km']).sort_values('Length_km', ascending=False).head(10)[['County', 'Habitat', 'Length_km']]
        st.markdown("<h4>Top 10 Longest Habitats (Km)</h4>", unsafe_allow_html=True)
        st.dataframe(top_lengths)

with tab4:
    # Association Analysis
    associations_count = pd.DataFrame(data['Association_Name'].value_counts()).reset_index()
    associations_count.columns = ['Association', 'Number of Habitats Managed']
    
    fig = px.bar(
        associations_count.head(15),
        x='Association',
        y='Number of Habitats Managed',
        title='Top 15 Associations by Number of Managed Habitats',
        color='Number of Habitats Managed',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    # Start/End Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Start locations
        st.markdown("<h3>Start Locations Analysis</h3>", unsafe_allow_html=True)
        
        # Missing values stats
        start_missing = data['start_location'].isna().sum()
        start_present = len(data) - start_missing
        
        missing_df = pd.DataFrame({
            'Status': ['Present', 'Missing'],
            'Count': [start_present, start_missing]
        })
        
        fig = px.pie(
            missing_df,
            values='Count',
            names='Status',
            title='Start Location Availability',
            color_discrete_sequence=['#1E88E5', '#90CAF9']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart showing proportion of habitats starting from spring
        if start_present > 0:
            spring_counts = data['start_from_spring'].value_counts().reset_index()
            spring_counts.columns = ['Starts from Spring', 'Count']
            spring_counts['Starts from Spring'] = spring_counts['Starts from Spring'].map({True: 'Yes', False: 'No'})
            
            fig = px.pie(
                spring_counts,
                values='Count',
                names='Starts from Spring',
                title='Proportion of Habitats Starting from Spring',
                color_discrete_sequence=['#1E88E5', '#90CAF9']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # End locations/rivers
        st.markdown("<h3>End Locations & Rivers Analysis</h3>", unsafe_allow_html=True)
        
        # Missing values stats for end data
        end_missing = data['end_location'].isna().sum()
        end_present = len(data) - end_missing
        
        missing_df = pd.DataFrame({
            'Status': ['Present', 'Missing'],
            'Count': [end_present, end_missing]
        })
        
        fig = px.pie(
            missing_df,
            values='Count',
            names='Status',
            title='End Location Availability',
            color_discrete_sequence=['#1E88E5', '#90CAF9']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis of rivers at endpoints
        if data['end_river'].notna().sum() > 0:
            # Top rivers where habitats end
            top_rivers = data['end_river'].value_counts().reset_index().head(10)
            top_rivers.columns = ['River', 'Count']
            
            fig = px.bar(
                top_rivers,
                x='River',
                y='Count',
                title='Top 10 Rivers Where Habitats End',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Map visualization (if coordinates are available)
st.markdown("<h2 class='sub-header'>Explore by Habitat Name</h2>", unsafe_allow_html=True)

# Search by habitat name
habitat_search = st.text_input("Search for a specific habitat (e.g., 'Mureș', 'Dunăre', 'Siret')")

if habitat_search:
    search_results = data[data['Habitat'].str.contains(habitat_search, case=False, na=False)]
    
    if not search_results.empty:
        st.success(f"Found {len(search_results)} habitats matching your search.")
        st.dataframe(search_results)
        
        # Simple stats about the search results
        if len(search_results) > 1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"**Summary for habitats containing '{habitat_search}':**")
            
            total_length = search_results['Length_km'].sum()
            total_area = search_results['Area_ha'].sum()
            counties_count = search_results['County'].nunique()
            
            st.markdown(f"- Present in {counties_count} counties")
            if not pd.isna(total_length) and total_length > 0:
                st.markdown(f"- Total length: {total_length:.2f} km")
            if not pd.isna(total_area) and total_area > 0:
                st.markdown(f"- Total area: {total_area:.2f} ha")
                
            # Additional stats for new fields
            springs_count = search_results['start_from_spring'].sum()
            river_ends_count = search_results['end_to_river'].sum()
            
            if springs_count > 0:
                st.markdown(f"- {springs_count} start from springs ({springs_count/len(search_results)*100:.1f}%)")
            if river_ends_count > 0:
                st.markdown(f"- {river_ends_count} end in another river ({river_ends_count/len(search_results)*100:.1f}%)")
                
            # If there are entries ending in rivers, show distribution
            if river_ends_count > 0 and search_results['end_river'].notna().sum() > 0:
                rivers_count = search_results['end_river'].value_counts()
                if len(rivers_count) > 0:
                    st.markdown("**Most common endpoint rivers:**")
                    for river, count in rivers_count.head(5).items():
                        st.markdown(f"- {river}: {count} habitats")
                
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"No habitats found containing '{habitat_search}'.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Data source: National Agency for Fishing and Aquaculture (ANPA) Romania</p>", unsafe_allow_html=True) 