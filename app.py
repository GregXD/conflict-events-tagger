import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import cohere
import os
from dotenv import load_dotenv
import re
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap

# Load environment variables
load_dotenv()

# Set up Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    st.error("COHERE_API_KEY environment variable is not set.")
    st.stop()

co = cohere.Client(cohere_api_key)

# Set up SQLite database
conn = sqlite3.connect('conflict_events.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS events
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              url TEXT,
              event_type TEXT,
              confidence REAL,
              country TEXT,
              news_source TEXT,
              fatalities TEXT,
              summary TEXT,
              timestamp DATETIME)''')
conn.commit()

def classify_event(text):
    response = co.classify(
        model='2fcfb5aa-5d0c-4758-ace5-ce80d13034fd-ft',
        inputs=[text]
    )
    return response.classifications[0]

def get_country(text):
    prompt = f"""Based on the following news article, determine the country where the event occurred. Provide only the name of the country.

News article:
{text}

Country:"""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=20,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.strip()

def get_news_source(text):
    prompt = f"""Based on the following news article, determine the news source that published this article. Provide only the name of the news source.

News article:
{text}

News Source:"""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=20,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.strip()

def get_fatalities(text):
    prompt = f"""Based on the following news article, determine the number of recorded fatalities from the event described. Provide only the number as an integer. If the number is not specified or unclear, respond with "Unknown".

News article:
{text}

Number of fatalities:"""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=20,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    
    result = response.generations[0].text.strip()
    
    # Try to extract an integer from the result
    match = re.search(r'\d+', result)
    if match:
        return int(match.group())
    else:
        return "Unknown"

def get_summary(text):
    prompt = f"""Summarize the following news article about a conflict event in exactly two sentences. Focus on the key details of the event.

News article:
{text}

Two-sentence summary:"""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=100,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.strip()

def tag_news_source(url):
    # Load the text content from the URL
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])

    # Classify the event
    classification = classify_event(text)

    # Get the country
    country = get_country(text)

    # Get the news source
    news_source = get_news_source(text)

    # Get the number of fatalities
    fatalities = get_fatalities(text)

    # Get the summary
    summary = get_summary(text)

    # Insert the results into the database
    c.execute('''INSERT INTO events (url, event_type, confidence, country, news_source, fatalities, summary, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (url, classification.prediction, classification.confidence, country, news_source, str(fatalities), summary, datetime.now()))
    conn.commit()

    # Create a markdown table with the results
    result = f"""
| Field | Value |
|-------|-------|
| Event Type | {classification.prediction} |
| Confidence | {classification.confidence:.2f} |
| Country | {country} |
| News Source | {news_source} |
| Fatalities | {fatalities} |
| Summary | {summary} |
"""
    return result

# Define a Gapminder-inspired color palette
gapminder_colors = px.colors.qualitative.Set2

def create_event_type_chart():
    c.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=['Event Type', 'Count'])
    
    fig = px.bar(df, x='Event Type', y='Count', text='Count',
                 title='Distribution of Event Types',
                 labels={'Count': 'Number of Events'},
                 template='ggplot2',
                 color='Event Type',
                 color_discrete_sequence=gapminder_colors)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(showlegend=False)  # Hide legend as colors are self-explanatory
    
    return fig

def create_fatalities_by_country_chart():
    c.execute("SELECT country, SUM(CASE WHEN fatalities != 'Unknown' THEN CAST(fatalities AS INTEGER) ELSE 0 END) as total_fatalities FROM events GROUP BY country")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=['Country', 'Total Fatalities'])
    df = df.sort_values('Total Fatalities', ascending=False).head(10)  # Top 10 countries
    
    fig = px.bar(df, x='Country', y='Total Fatalities', text='Total Fatalities',
                 title='Top 10 Countries by Fatalities',
                 labels={'Total Fatalities': 'Number of Fatalities'},
                 template='ggplot2',
                 color='Country',
                 color_discrete_sequence=gapminder_colors)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(showlegend=False)  # Hide legend as colors are self-explanatory
    
    return fig

def create_map():
    # Fetch data from the database
    c.execute("SELECT country, fatalities, event_type FROM events WHERE country != ''")
    data = c.fetchall()
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Country', 'Fatalities', 'Event Type'])
    
    # Convert fatalities to numeric, replacing 'Unknown' with 0
    df['Fatalities'] = pd.to_numeric(df['Fatalities'].replace('Unknown', 0))
    
    # Group by country and sum fatalities
    df_grouped = df.groupby('Country').agg({'Fatalities': 'sum', 'Event Type': 'count'}).reset_index()
    
    # Create a map centered on the world
    m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron')
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Create a color map for fatalities
    colormap = LinearColormap(colors=['green', 'yellow', 'red'], vmin=df_grouped['Fatalities'].min(), vmax=df_grouped['Fatalities'].max())
    
    # Geocode countries and add markers
    geolocator = Nominatim(user_agent="conflict_event_app")
    
    for _, row in df_grouped.iterrows():
        try:
            location = geolocator.geocode(row['Country'])
            if location:
                folium.CircleMarker(
                    location=[location.latitude, location.longitude],
                    radius=min(row['Fatalities'], 30),  # Cap the radius at 30
                    popup=f"Country: {row['Country']}<br>Fatalities: {row['Fatalities']}<br>Events: {row['Event Type']}",
                    color=colormap(row['Fatalities']),
                    fill=True,
                    fill_color=colormap(row['Fatalities']),
                    fill_opacity=0.7
                ).add_to(marker_cluster)
        except Exception as e:
            st.warning(f"Could not locate {row['Country']}: {str(e)}")
    
    # Add color map to the map
    colormap.add_to(m)
    colormap.caption = 'Fatalities'
    
    return m

def data_entry_page():
    st.title("Conflict Event Classifier")

    url = st.text_input("Enter a news article URL:")

    if st.button("Analyze"):
        if url:
            with st.spinner("Analyzing the news source..."):
                try:
                    result = tag_news_source(url)
                    st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a URL.")

    # Add a section to display recent analyses as a styled table
    st.subheader("Recent Analyses")
    c.execute("SELECT url, event_type, confidence, country, timestamp FROM events ORDER BY timestamp DESC LIMIT 5")
    recent_analyses = c.fetchall()
    
    if recent_analyses:
        df = pd.DataFrame(recent_analyses, columns=['URL', 'Event Type', 'Confidence', 'Country', 'Timestamp'])
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2f}")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Truncate long URLs
        df['URL'] = df['URL'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        # Use Streamlit's built-in styling
        st.dataframe(
            df,
            column_config={
                "URL": st.column_config.TextColumn(
                    "URL",
                    help="The source URL of the news article",
                    max_chars=50,
                ),
                "Event Type": st.column_config.TextColumn(
                    "Event Type",
                    help="Classified type of the conflict event",
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    help="Confidence score of the classification",
                    format="%.2f",
                ),
                "Country": st.column_config.TextColumn(
                    "Country",
                    help="Country where the event occurred",
                ),
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    help="Date and time of the analysis",
                    format="YYYY-MM-DD HH:mm:ss",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recent analyses available.")

def dashboard_page():
    st.title("Conflict Event Dashboard")

    # Add bar charts
    col1, col2 = st.columns(2)

    with col1:
        event_type_chart = create_event_type_chart()
        st.plotly_chart(event_type_chart, use_container_width=True)

    with col2:
        fatalities_chart = create_fatalities_by_country_chart()
        st.plotly_chart(fatalities_chart, use_container_width=True)

    # Add map visualization
    st.subheader("Event Locations")
    map = create_map()
    
    # Display the map
    folium_static(map, width=1000, height=600)

# Main app logic
def main():
    st.sidebar.title("Navigation")
    
    # Create a dictionary of pages with their corresponding icons
    pages = {
        "Data Entry": "📝",
        "Dashboard": "📊"
    }
    
    # Create a list of options for the selectbox
    options = [f"{icon} {page}" for page, icon in pages.items()]
    
    # Create the selectbox
    selection = st.sidebar.selectbox("Go to", options)
    
    # Extract the page name from the selection
    page = selection.split(" ", 1)[1]

    if page == "Data Entry":
        data_entry_page()
    elif page == "Dashboard":
        dashboard_page()

    # Add some information about the app
    st.sidebar.header("About")
    st.sidebar.info(
        "This app analyzes news articles about conflict events, classifies them into one of 5 event types, "
        "determines the country where the event occurred, identifies the news source, "
        "estimates the number of fatalities, and provides a brief summary. "
        "Enter a URL of a news article to get started."
    )

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Exchange.Design")

if __name__ == "__main__":
    main()

# Close the database connection when the app is closed
conn.close()