import streamlit as st
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
from langchain_community.document_loaders import WebBaseLoader
import logging
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from queue import Queue
from datetime import datetime
from newsapi import NewsApiClient
from mapbox import Geocoder


st.set_page_config(page_title="Conflict Event Tagger", page_icon="xd_logo.png", layout="wide")

# Constants and Configuration
DATABASE_NAME = os.path.join(os.path.dirname(__file__), 'conflict_events.db')
COHERE_MODEL_ID = os.getenv("EVENT_CLASSIFICATION_MODEL_ID")
GAPMINDER_COLORS = px.colors.qualitative.Set2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and set up APIs
load_dotenv()
mapbox_api_key = os.getenv("MAPBOX_TOKEN")
if not mapbox_api_key:
    st.error("MAPBOX_TOKEN environment variable is not set.")
    st.stop()
geocoder = Geocoder(access_token=mapbox_api_key)

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    st.error("COHERE_API_KEY environment variable is not set.")
    st.stop()
co = cohere.Client(cohere_api_key)

# Load NewsAPI key from environment variables
newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    st.error("NEWSAPI_KEY environment variable is not set.")
    st.stop()
newsapi = NewsApiClient(api_key=newsapi_key)

# Database functions with caching
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        conn.close()

@st.cache_data(ttl=60)  # Cache for 1 minute
def execute_db_query_cached(query, params=None):
    """Cached version of database queries for read-only operations"""
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            if params:
                c.execute(query, params)
            else:
                c.execute(query)
            result = c.fetchall()
            return result
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

def execute_db_query(query, params=None, fetch=True):
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            if params:
                c.execute(query, params)
            else:
                c.execute(query)
            if fetch:
                result = c.fetchall()
                return result
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_dashboard_stats():
    """Get dashboard statistics with caching"""
    total_events = execute_db_query_cached("SELECT COUNT(*) FROM events")[0][0]
    total_countries = execute_db_query_cached("SELECT COUNT(DISTINCT country) FROM events")[0][0]
    total_fatalities = execute_db_query_cached("SELECT SUM(CAST(fatalities AS INTEGER)) FROM events WHERE fatalities != 'Unknown'")[0][0]
    return total_events, total_countries, total_fatalities or 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_analyses_cached(refresh_key=None):
    """Cached version of fetch_analyses"""
    analyses = execute_db_query_cached("SELECT id, url, event_type, confidence, country, news_source, fatalities, summary, event_date, key_actors FROM events ORDER BY event_date DESC")
    return [(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], 
             datetime.strptime(row[8], '%Y-%m-%d').date() if row[8] else None,
             row[9]) 
            for row in analyses]

@st.cache_data(ttl=600)  # Cache for 10 minutes
def geocode_country_cached(country):
    """Cached geocoding to avoid repeated API calls"""
    try:
        response = geocoder.forward(country).json()
        if response['features']:
            location = response['features'][0]['geometry']['coordinates']
            return location[1], location[0]  # Return latitude, longitude
    except Exception as e:
        logger.error(f"Geocoding error for {country}: {e}")
    return None, None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_news_cached(query="conflict", language="en"):
    """Cached version of fetch_live_news"""
    try:
        articles = newsapi.get_everything(q=query, language=language, sort_by="publishedAt")
        return articles['articles']
    except Exception as e:
        logger.error(f"Failed to fetch news: {str(e)}")
        return []

def update_database_schema():
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            c.execute('''CREATE TABLE IF NOT EXISTS events
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          url TEXT, event_type TEXT, confidence REAL,
                          country TEXT, news_source TEXT, fatalities TEXT,
                          summary TEXT, timestamp DATETIME, event_date DATE,
                          key_actors TEXT)''')
            conn.commit()
            logger.info("Events table created or already exists.")
        except sqlite3.Error as e:
            logger.error(f"An error occurred while updating the database schema: {e}")
            conn.rollback()


# Cohere API functions
def classify_event(text):
    response = co.classify(model=COHERE_MODEL_ID, inputs=[text])
    return response.classifications[0]

def get_cohere_response(prompt):
    response = co.generate(model='command', prompt=prompt, max_tokens=100, temperature=0.3, k=0, stop_sequences=[], return_likelihoods='NONE')
    return response.generations[0].text.strip()

# Event analysis functions
def get_country(text):
    prompt = f"Based on the following news article, determine the country where the event occurred. Provide only the name of the country.\n\nNews article:\n{text}\n\nCountry:"
    return get_cohere_response(prompt)

def get_news_source(text):
    prompt = f"Based on the following news article, determine the news source that published this article. Provide only the name of the news source.\n\nNews article:\n{text}\n\nNews Source:"
    return get_cohere_response(prompt)

def get_fatalities(text):
    prompt = f"Based on the following news article, determine the number of recorded fatalities from the event described. Provide only the number as an integer. If the number is not specified or unclear, respond with 'Unknown'.\n\nNews article:\n{text}\n\nNumber of fatalities:"
    result = get_cohere_response(prompt)
    match = re.search(r'\d+', result)
    return int(match.group()) if match else "Unknown"

def get_summary(text):
    prompt = f"Summarize the following news article about a conflict event in exactly two sentences. Focus on the key details of the event.\n\nNews article:\n{text}\n\nTwo-sentence summary:"
    return get_cohere_response(prompt)

def get_key_actors(text):
    prompt = f"Based on the following news article, identify the key actors (individuals, groups, or organizations) involved in the event. Provide a comma-separated list of the most important 2-3 actors.\n\nNews article:\n{text}\n\nKey actors:"
    return get_cohere_response(prompt)

def delete_event(event_id):
    try:
        execute_db_query("DELETE FROM events WHERE id = ?", (event_id,), fetch=False)
        logger.info(f"Deleted event with ID: {event_id}")
    except sqlite3.Error as e:
        logger.error(f"Error deleting event: {e}")
        raise

def delete_event_callback(event_id):
    try:
        delete_event(event_id)
        st.session_state.delete_success = f"Event {event_id} deleted successfully."
    except Exception as e:
        st.session_state.delete_error = f"Error deleting event {event_id}: {e}"

def fetch_analyses():
    """Non-cached version for real-time updates"""
    analyses = execute_db_query("SELECT id, url, event_type, confidence, country, news_source, fatalities, summary, event_date, key_actors FROM events ORDER BY event_date DESC")
    return [(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], 
             datetime.strptime(row[8], '%Y-%m-%d').date() if row[8] else None,
             row[9]) 
            for row in analyses]

def geocode_country(country):
    """Non-cached version for real-time updates"""
    try:
        response = geocoder.forward(country).json()
        if response['features']:
            location = response['features'][0]['geometry']['coordinates']
            return location[1], location[0]  # Return latitude, longitude
    except Exception as e:
        logger.error(f"Geocoding error for {country}: {e}")
    return None, None

def get_event_date(text):
    prompt = f"""Based on the following news article, determine the date when the event occurred. Provide the date in YYYY-MM-DD format. If the exact date is not specified, provide the most likely date based on the context. If no date can be determined, respond with 'Unknown'.

News article:
{text}

Event Date (YYYY-MM-DD):"""

    result = get_cohere_response(prompt)
    logger.info(f"Raw event date from Cohere: {result}")
    
    if result.lower() == 'unknown':
        logger.info("Event date is unknown")
        return None
    
    # Try parsing the date in different formats
    date_formats = ["%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]
    
    for date_format in date_formats:
        try:
            date_obj = datetime.strptime(result, date_format).date()
            logger.info(f"Parsed event date: {date_obj}")
            return date_obj
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {result}")
    # Try to extract a date from the text if Cohere didn't return a valid date
    date_pattern = r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
    match = re.search(date_pattern, text, re.IGNORECASE)
    if match:
        extracted_date = match.group(1)
        for date_format in date_formats:
            try:
                date_obj = datetime.strptime(extracted_date, date_format).date()
                logger.info(f"Extracted date from text: {date_obj}")
                return date_obj
            except ValueError:
                continue
        logger.warning(f"Extracted date is invalid: {extracted_date}")
    return None

def tag_news_source(url):
    max_retries = 3
    retry_delay = 2
    timeout = 60  # 60 seconds timeout

    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    result_queue = Queue()

    def analyze_url():
        try:
            result_queue.put(("progress", "Starting analysis..."))
            logger.info(f"Starting analysis for URL: {url}")
            
            result_queue.put(("progress", "Loading text from URL..."))
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                text = docs[0].page_content
            except Exception as e:
                logger.error(f"Error fetching URL: {e}")
                raise Exception(f"Failed to fetch the URL: {e}")

            # Limit text length to avoid exceeding API limits
            max_text_length = 10000
            if len(text) > max_text_length:
                text = text[:max_text_length]
                logger.warning(f"Text truncated to {max_text_length} characters")

            result_queue.put(("progress", "Text loaded from URL."))
            logger.info(f"Text loaded. Length: {len(text)} characters")

            result_queue.put(("progress", "Classifying event..."))
            classification = classify_event(text)
            result_queue.put(("result", f"This event appears to be a **{classification.prediction}**."))
            logger.info(f"Event classification: {classification.prediction} (Confidence: {classification.confidence:.2f})")

            result_queue.put(("progress", "Identifying country..."))
            country = get_country(text)
            result_queue.put(("result", f"This event appears to be a **{classification.prediction}** that occurred in **{country}**."))
            logger.info(f"Identified country: {country}")

            result_queue.put(("progress", "Identifying news source..."))
            news_source = get_news_source(text)
            result_queue.put(("result", f"This **{classification.prediction}** event occurred in **{country}**, as reported by **{news_source}**."))
            logger.info(f"Identified news source: {news_source}")

            result_queue.put(("progress", "Estimating fatalities..."))
            fatalities = get_fatalities(text)
            result_queue.put(("result", f"This **{classification.prediction}** event in **{country}**, reported by **{news_source}**, resulted in **{fatalities}** fatalities."))
            logger.info(f"Estimated fatalities: {fatalities}")

            result_queue.put(("progress", "Generating summary..."))
            summary = get_summary(text)
            result_queue.put(("result", f"""
            Event Type: **{classification.prediction}**
            Country: **{country}**
            News Source: **{news_source}**
            Fatalities: **{fatalities}**
            
            Summary: {summary}
            """))
            logger.info(f"Generated summary: {summary}")

            result_queue.put(("progress", "Determining event date..."))
            event_date = get_event_date(text)
            logger.info(f"Raw event_date: {event_date}")
            event_date_str = event_date.strftime("%Y-%m-%d") if event_date else None
            logger.info(f"Formatted event_date_str: {event_date_str}")
            result_queue.put(("result", f"This **{classification.prediction}** event occurred on **{event_date_str or 'Unknown date'}** in **{country}**, as reported by **{news_source}**."))
            logger.info(f"Determined event date: {event_date_str or 'Unknown'}")

            result_queue.put(("progress", "Identifying key actors..."))
            key_actors = get_key_actors(text)
            result_queue.put(("result", f"This **{classification.prediction}** event in **{country}**, reported by **{news_source}**, involved key actors: **{key_actors}**."))
            logger.info(f"Identified key actors: {key_actors}")

            result_queue.put(("progress", "Inserting results into the database..."))
            query = '''INSERT INTO events (url, event_type, confidence, country, news_source, fatalities, summary, timestamp, event_date, key_actors)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            params = (url, classification.prediction, classification.confidence, country, news_source, str(fatalities), summary, datetime.now(), event_date, key_actors)
            logger.info(f"SQL Query: {query}")
            logger.info(f"SQL Params: {params}")
            execute_db_query(query, params, fetch=False)
            
            # Get the ID of the newly inserted event
            new_id_result = execute_db_query("SELECT last_insert_rowid()")
            new_event_id = new_id_result[0][0] if new_id_result else None
            logger.info(f"New event ID: {new_event_id}")
            
            result_queue.put(("progress", "Analysis completed successfully."))
            result_queue.put(("success", new_event_id))
        except Exception as e:
            logger.exception(f"Error during URL analysis: {str(e)}")
            result_queue.put(("error", f"Error during analysis: {str(e)}"))

    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(analyze_url)
                while True:
                    try:
                        msg_type, msg_content = result_queue.get(timeout=timeout)
                        if msg_type == "progress":
                            progress_placeholder.text(msg_content)
                        elif msg_type == "result":
                            result_placeholder.markdown(msg_content)
                        elif msg_type == "error":
                            progress_placeholder.text(msg_content)
                            raise Exception(msg_content)
                        elif msg_type == "success":
                            # Analysis completed successfully, set session state for UI feedback
                            st.session_state.analysis_success = True
                            st.session_state.latest_analysis_id = msg_content
                            st.session_state.refresh_timestamp = time.time()  # Update to refresh data
                            progress_placeholder.text("✅ Analysis completed! Redirecting to Recent Analyses...")
                            return result_placeholder
                    except TimeoutError:
                        progress_placeholder.text("Analysis timed out. Please try again.")
                        logger.error("Analysis timed out")
                        raise

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                progress_placeholder.text(f"An error occurred. Retrying in {retry_delay} seconds...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All attempts failed")
                progress_placeholder.text("Analysis failed after multiple attempts.")
                return None

    return None


# Chart creation functions with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_events_by_country_chart():
    data = execute_db_query_cached("SELECT country, COUNT(*) as event_count FROM events GROUP BY country ORDER BY event_count DESC LIMIT 10")
    if not data:
        return None
    df = pd.DataFrame(data, columns=['Country', 'Event Count'])
    
    fig = px.bar(df, y='Country', x='Event Count', text='Event Count',
                 title='Top 10 Countries by Number of Events',
                 labels={'Event Count': 'Number of Events'},
                 template='ggplot2',
                 orientation='h',
                 color_discrete_sequence=['#3366cc'])  # Single blue color
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
    fig.update_yaxes(categoryorder='total ascending')  # Order from lowest to highest (bottom to top)
    return fig

@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_event_type_chart():
    data = execute_db_query_cached("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
    if not data:
        return None
    df = pd.DataFrame(data, columns=['Event Type', 'Count'])
    fig = px.bar(df, x='Event Type', y='Count', text='Count',
                 title='Distribution of Event Types',
                 labels={'Count': 'Number of Events'},
                 template='ggplot2', color='Event Type',
                 color_discrete_sequence=GAPMINDER_COLORS)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
    return fig

@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_fatalities_by_country_chart():
    data = execute_db_query_cached("SELECT country, SUM(CASE WHEN fatalities != 'Unknown' THEN CAST(fatalities AS INTEGER) ELSE 0 END) as total_fatalities FROM events GROUP BY country")
    if not data:
        return None
    df = pd.DataFrame(data, columns=['Country', 'Total Fatalities'])
    df = df.sort_values('Total Fatalities', ascending=False).head(10)
    fig = px.bar(df, x='Country', y='Total Fatalities', text='Total Fatalities',
                 title='Top 10 Countries by Fatalities',
                 labels={'Total Fatalities': 'Number of Fatalities'},
                 template='ggplot2', color='Country',
                 color_discrete_sequence=GAPMINDER_COLORS)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
    return fig

@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_events_over_time_chart():
    data = execute_db_query_cached("SELECT event_date, COUNT(*) as event_count FROM events GROUP BY event_date ORDER BY event_date")
    if not data:
        return None
    df = pd.DataFrame(data, columns=['Date', 'Event Count'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    if df.empty:
        return None
    
    fig = px.bar(df, x='Date', y='Event Count',
                 title='Events Over Time',
                 labels={'Event Count': 'Number of Events'},
                 template='ggplot2')
    
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Events")
    return fig

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_map_data():
    """Get and process map data with caching"""
    data = execute_db_query_cached("SELECT country, fatalities, event_type FROM events WHERE country != ''")
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=['Country', 'Fatalities', 'Event Type'])
    
    # Convert 'Unknown' to 0 and ensure all values are numeric
    df['Fatalities'] = pd.to_numeric(df['Fatalities'].replace({'Unknown': '0', '': '0'}), errors='coerce').fillna(0)
    
    # Aggregate data by country
    df_agg = df.groupby('Country').size().reset_index(name='Event Count')
    
    # Geocode countries to get latitude and longitude
    df_agg[['lat', 'lon']] = df_agg.apply(lambda row: pd.Series(geocode_country_cached(row['Country'])), axis=1)
    
    # Filter out rows with missing coordinates
    df_agg = df_agg.dropna(subset=['lat', 'lon'])
    
    if not df_agg.empty:
        # Normalize the event counts to a suitable range for dot sizes
        min_size = 10000  # Minimum dot size
        max_size = 50000  # Maximum dot size
        df_agg['Size'] = ((df_agg['Event Count'] - df_agg['Event Count'].min()) / 
                          (df_agg['Event Count'].max() - df_agg['Event Count'].min()) * 
                          (max_size - min_size) + min_size)
    
    return df_agg

def create_map():
    """Optimized Folium map creation with caching"""
    with st.spinner("Loading map data..."):
        df_agg = get_map_data()
    
    if df_agg.empty:
        st.warning("No data available to display the map.")
        return None
    
    # Create a Folium map centered on the mean coordinates
    m = folium.Map(
        location=[df_agg['lat'].mean(), df_agg['lon'].mean()],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Add marker cluster for better performance with many markers
    marker_cluster = MarkerCluster().add_to(m)
    
    # Create a colormap for event counts
    min_count = df_agg['Event Count'].min()
    max_count = df_agg['Event Count'].max()
    colormap = LinearColormap(
        colors=['yellow', 'orange', 'red'],
        vmin=min_count,
        vmax=max_count
    )
    
    # Add markers for each country
    for idx, row in df_agg.iterrows():
        # Calculate marker size based on event count
        radius = max(5, min(25, row['Event Count'] * 2))
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            popup=f"<b>{row['Country']}</b><br>Events: {row['Event Count']}",
            tooltip=f"{row['Country']}: {row['Event Count']} events",
            color='darkred',
            fill=True,
            fillColor=colormap(row['Event Count']),
            fillOpacity=0.7,
            weight=2
        ).add_to(marker_cluster)
    
    return m

def data_entry_page():
    st.title("Conflict Event Classifier")
    
    # Initialize session state for tab management
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'analysis_success' not in st.session_state:
        st.session_state.analysis_success = False
    if 'latest_analysis_id' not in st.session_state:
        st.session_state.latest_analysis_id = None
    if 'refresh_timestamp' not in st.session_state:
        st.session_state.refresh_timestamp = time.time()
    
    # Create tabs for News API and URL input
    tab1, tab2, tab3 = st.tabs(["Recent Analyses","Manual URL Input","Search for News"])
    
    with tab1:
        st.subheader("Recent Analyses")
        
        # Show success message if analysis was just completed
        if st.session_state.analysis_success:
            st.success("✓ New event successfully analyzed and added to the database!")
            if st.session_state.latest_analysis_id:
                st.info(f"Event ID: {st.session_state.latest_analysis_id} has been added to the table below.")
            st.session_state.analysis_success = False  # Reset flag
        
        # Add refresh button and filters
        col1, col2, col3 = st.columns([3, 2, 1])
        
        # Use cached data for better performance
        with st.spinner("Loading analyses..."):
            analyses = fetch_analyses_cached(refresh_key=st.session_state.refresh_timestamp)

        if analyses:
            with col1:
                # Event type filter
                event_types = ['All'] + sorted(set(row[2] for row in analyses))
                selected_event_type = st.selectbox("Filter by Event Type", event_types, key="data_entry_event_type")

            with col2:
                # Search bar
                search_term = st.text_input("Search in all fields")
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Add space to align with other inputs
                if st.button("↻ Refresh", key="refresh_analyses", help="Refresh the data from database"):
                    # Update refresh timestamp to invalidate cache
                    st.session_state.refresh_timestamp = time.time()
                    st.success("Data refreshed!")

            # Filter the analyses
            filtered_analyses = analyses
            if selected_event_type != 'All':
                filtered_analyses = [row for row in filtered_analyses if row[2] == selected_event_type]
            
            if search_term:
                filtered_analyses = [row for row in filtered_analyses if any(search_term.lower() in str(cell).lower() for cell in row)]

            # Pagination
            items_per_page = 10
            total_items = len(filtered_analyses)
            total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
            
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    page = st.selectbox("Page", range(1, total_pages + 1), key="analysis_page")
                    st.write(f"Showing {min((page-1)*items_per_page + 1, total_items)} - {min(page*items_per_page, total_items)} of {total_items} results")
            else:
                page = 1

            # Get items for current page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_analyses = filtered_analyses[start_idx:end_idx]

            # Display the table with headers
            if page_analyses:
                # Table headers
                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 2, 1, 2, 2, 1, 4, 2, 1])
                col1.write("**ID**")
                col2.write("**URL**")
                col3.write("**Event Type**")
                col4.write("**Confidence**")
                col5.write("**Country**")
                col6.write("**News Source**")
                col7.write("**Fatalities**")
                col8.write("**Summary**")
                col9.write("**Event Date**")
                col10.write("**Action**")

                # Table rows
                for row in page_analyses:
                    # Highlight newly added row
                    is_new_entry = (st.session_state.latest_analysis_id == row[0])
                    if is_new_entry:
                        st.markdown("""
                        <div style="background-color: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745; margin-bottom: 10px;">
                            <strong>● Newly Added Event</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 2, 1, 2, 2, 1, 4, 2, 1])
                    with col1:
                        if is_new_entry:
                            st.markdown(f"**{row[0]}** ●")
                        else:
                            st.write(row[0])  # ID
                    with col2:
                        st.write(row[1][:50] + '...' if len(row[1]) > 50 else row[1])  # URL
                    with col3:
                        st.write(row[2])  # Event Type
                    with col4:
                        st.write(f"{row[3]:.2f}")  # Confidence
                    with col5:
                        st.write(row[4])  # Country
                    with col6:
                        st.write(row[5])  # News Source
                    with col7:
                        st.write(row[6])  # Fatalities
                    with col8:
                        st.write(row[7])  # Summary
                    with col9:
                        # Handle both string and date object types
                        if isinstance(row[8], str):
                            st.write(row[8])
                        elif row[8]:
                            st.write(row[8].strftime('%Y-%m-%d'))
                        else:
                            st.write('')  # Event Date
                    with col10:
                        if st.button('×', key=f"delete_{row[0]}", help="Delete this event"):
                            try:
                                delete_event(row[0])
                                st.success(f"Event {row[0]} deleted successfully.")
                                # Clear all caches to refresh data
                                fetch_analyses_cached.clear()
                                get_dashboard_stats.clear()
                                get_map_data.clear()
                                create_events_by_country_chart.clear()
                                create_event_type_chart.clear()
                                create_fatalities_by_country_chart.clear()
                                create_events_over_time_chart.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting event {row[0]}: {e}")
            else:
                st.info("No analyses match your filters.")
        else:
            st.info("No analyses available.")
    
    with tab2:
        st.subheader("Manual URL Input")
        url = st.text_input("Enter a news article URL:")
        if st.button("Analyze", key="analyze_manual"):
            if url:
                try:
                    result = tag_news_source(url)
                    # Clear all caches after successful analysis
                    fetch_analyses_cached.clear()
                    get_dashboard_stats.clear()
                    get_map_data.clear()
                    create_events_by_country_chart.clear()
                    create_event_type_chart.clear()
                    create_fatalities_by_country_chart.clear()
                    create_events_over_time_chart.clear()
                    logger.info("Analysis completed and displayed to user")
                    
                except sqlite3.Error as e:
                    logger.exception(f"Database error: {str(e)}")
                    st.error(f"A database error occurred: {str(e)}")
                    st.error("Please check your database configuration and try again.")
                except Exception as e:
                    logger.exception(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again later or with a different URL.")
            else:
                st.warning("Please enter a URL.")
                logger.warning("User attempted to analyze without entering a URL")
        
        # Handle post-analysis navigation outside the try/except block
        if st.session_state.analysis_success:
            st.success("✓ Analysis completed! Check the Recent Analyses tab to see your new event.")
            if st.button("→ Go to Recent Analyses", key="goto_recent_manual"):
                st.session_state.analysis_success = False
                st.info("Please click on the 'Recent Analyses' tab to see your new event highlighted.")
    
    with tab3:
        st.subheader("Live News Feed")
        query = st.text_input("Search for news articles", value="conflict", key="news_query")
        
        with st.spinner("Fetching news articles..."):
            articles = fetch_live_news_cached(query=query)
        
        if articles:
            for i, article in enumerate(articles[:5]):  # Display top 5 articles
                with st.expander(f"• {article['title'][:100]}..."):
                    st.markdown(f"**Source**: {article['source']['name']}")
                    st.markdown(f"**Published**: {article['publishedAt']}")
                    st.markdown(f"**Description**: {article['description'] or 'No description available'}")
                    st.markdown(f"**[Read full article]({article['url']})**")
                    
                    if st.button(f"Analyze This Article", key=f"analyze_{i}"):
                        try:
                            result = tag_news_source(article['url'])
                            # Clear all caches after successful analysis
                            fetch_analyses_cached.clear()
                            get_dashboard_stats.clear()
                            get_map_data.clear()
                            create_events_by_country_chart.clear()
                            create_event_type_chart.clear()
                            create_fatalities_by_country_chart.clear()
                            create_events_over_time_chart.clear()
                            logger.info("Analysis completed and displayed to user")
                            
                        except sqlite3.Error as e:
                            logger.exception(f"Database error: {str(e)}")
                            st.error(f"A database error occurred: {str(e)}")
                            st.error("Please check your database configuration and try again.")
                        except Exception as e:
                            logger.exception(f"Error during analysis: {str(e)}")
                            st.error(f"An error occurred: {str(e)}")
                            st.error("Please try again later or with a different URL.")
        else:
            st.write("No articles found for this query.")
        
        # Handle post-analysis navigation for news feed
        if st.session_state.analysis_success:
            st.success("✓ Analysis completed! Check the Recent Analyses tab to see your new event.")
            if st.button("→ Go to Recent Analyses", key="goto_recent_news"):
                st.session_state.analysis_success = False
                st.info("Please click on the 'Recent Analyses' tab to see your new event highlighted.")

def text_classifier_page():
    st.title("Text Classifier")
    st.markdown("**Quickly classify text content by conflict event type using AI**")
    
    # Import spaCy classifier functions
    try:
        from spacy_classifier import classify_with_spacy, is_spacy_available, get_spacy_classifier
        spacy_imported = True
    except ImportError:
        spacy_imported = False
    
    # Create classifier selection
    st.markdown("### Classification Method")
    
    # Check what classifiers are available
    cohere_available = True  # Assuming Cohere is configured
    spacy_available = spacy_imported and is_spacy_available() if spacy_imported else False
    
    classifier_options = []
    if cohere_available:
        classifier_options.append("Cohere API")
    if spacy_available:
        classifier_options.append("spaCy (Local)")
    
    if not classifier_options:
        st.error("No classifiers available. Please configure Cohere API or train a spaCy model.")
        return
    
    # Let user choose classifier
    if len(classifier_options) > 1:
        selected_classifier = st.radio(
            "Choose classification method:",
            classifier_options,
            horizontal=True,
            help="Cohere API provides cloud-based classification, spaCy runs locally"
        )
    else:
        selected_classifier = classifier_options[0]
        st.info(f"Using {selected_classifier} classifier")
    
    # Show classifier info
    if selected_classifier == "spaCy (Local)" and spacy_available:
        st.success("✓ spaCy model loaded and ready")
    elif selected_classifier == "Cohere API":
        st.info("ℹ️ Using Cohere cloud API")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload Text File"],
            horizontal=True
        )
        
        user_text = ""
        
        if input_method == "Type/Paste Text":
            user_text = st.text_area(
                "Enter text to classify:",
                height=300,
                placeholder="Paste news article text, social media posts, reports, or any text describing conflict events here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt', 'md'],
                help="Upload a .txt or .md file containing the text you want to classify"
            )
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=user_text, height=200, disabled=True)
        
        # Character count
        if user_text:
            char_count = len(user_text)
            st.caption(f"Character count: {char_count:,}")
            if char_count > 10000:
                st.warning("⚠️ Text is quite long. Consider using shorter excerpts for better accuracy.")
    
    with col2:
        st.markdown("### Classification")
        
        # Classification button
        if st.button("🔍 Classify Text", type="primary", disabled=not user_text.strip()):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    try:
                        # Use the selected classifier
                        if selected_classifier == "spaCy (Local)" and spacy_available:
                            classification = classify_with_spacy(user_text)
                            if classification is None:
                                st.error("spaCy classification failed. Please try Cohere API.")
                                return
                        else:
                            # Use Cohere API (existing function)
                            classification = classify_event(user_text)
                        
                        # Display results
                        st.success("✓ Classification Complete")
                        
                        # Event type with confidence
                        confidence_color = "green" if classification.confidence > 0.7 else "orange" if classification.confidence > 0.5 else "red"
                        
                        st.markdown("#### Results")
                        st.markdown(f"""
                        **Event Type:** `{classification.prediction}`
                        
                        **Confidence:** <span style="color: {confidence_color}; font-weight: bold;">{classification.confidence:.1%}</span>
                        
                        **Classifier:** `{selected_classifier}`
                        """, unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if classification.confidence > 0.8:
                            st.info("🎯 **High confidence** - Very likely classification")
                        elif classification.confidence > 0.6:
                            st.info("📊 **Medium confidence** - Reasonably likely classification")
                        else:
                            st.warning("⚠️ **Low confidence** - Classification uncertain")
                        
                        # Show raw confidence score
                        st.progress(classification.confidence)
                        
                        # Show detailed predictions for spaCy
                        if selected_classifier == "spaCy (Local)" and spacy_available:
                            classifier = get_spacy_classifier()
                            all_predictions = classifier.get_all_predictions(user_text)
                            if all_predictions:
                                with st.expander("📊 View All Predictions"):
                                    st.markdown("**All category scores:**")
                                    for label, score in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                                        st.markdown(f"- **{label}**: {score:.1%}")
                        
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")
                        logger.error(f"Text classification error: {e}")
            else:
                st.warning("Please enter some text to classify.")
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Information section
    with st.expander("ℹ️ About Text Classification"):
        st.markdown("""
        ### How It Works
        
        This tool provides **two classification options**:
        
        **Cohere API (Cloud):**
        - Uses the same AI model that powers the full article analysis
        - Requires internet connection and API credits
        - Generally very accurate and handles diverse text styles
        
        **spaCy (Local):**
        - Runs entirely on your machine (no internet required)
        - Uses a custom-trained model on conflict event data
        - Faster response time and no API costs
        - Available when the local model is trained and loaded
        
        **What it classifies:**
        - Armed conflicts and battles
        - Explosions and remote violence
        - Violence against civilians
        - Riots and civil unrest
        - Protests and demonstrations
        - Strategic developments
        
        **Tips for better results:**
        - Use clear, descriptive text about the event
        - Include context about what happened
        - Shorter, focused text often works better than very long passages
        - English text typically produces the most accurate results
        
        **Note:** This is a classification-only tool. For full analysis including location, fatalities, and other details, use the main **Data Entry** page.
        """)
    
    # Sample texts for testing
    

def dashboard_page():
    st.title("Conflict Event Dashboard")

    # Key Statistics at the top with caching
    st.markdown("### Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Loading dashboard statistics..."):
        total_events, total_countries, total_fatalities = get_dashboard_stats()

    with col1:
        st.metric(label="Total Events", value=total_events)

    with col2:
        st.metric(label="Countries Affected", value=total_countries)

    with col3:
        st.metric(label="Total Fatalities", value=total_fatalities)

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the map
    st.subheader("Event Locations")
    map_obj = create_map()
    if map_obj:
        folium_static(map_obj, width=1200, height=500)

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Events over time chart
    with st.spinner("Loading events over time chart..."):
        events_over_time = create_events_over_time_chart()
    if events_over_time:
        st.plotly_chart(events_over_time, use_container_width=True)
    else:
        st.info("No temporal data available to display events over time.")

    # Create tabs for different chart views
    tab1, tab2 = st.tabs(["Events Overview", "Fatalities Overview"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Loading event type chart..."):
                event_type_chart = create_event_type_chart()
            if event_type_chart:
                st.plotly_chart(event_type_chart, use_container_width=True)
            else:
                st.info("No event type data available.")
        with col2:
            with st.spinner("Loading events by country chart..."):
                events_by_country_chart = create_events_by_country_chart()
            if events_by_country_chart:
                st.plotly_chart(events_by_country_chart, use_container_width=True)
            else:
                st.info("No country-specific data available.")

    with tab2:
        with st.spinner("Loading fatalities chart..."):
            fatalities_chart = create_fatalities_by_country_chart()
        if fatalities_chart:
            st.plotly_chart(fatalities_chart, use_container_width=True)
        else:
            st.info("No fatality data available for visualization.")

def about_page():
    st.title("About Conflict Event Tagger")
    
    # Introduction
    st.markdown("""
    ## What is Conflict Event Tagger?
    
    Conflict Event Tagger is an AI-powered application designed to analyze news articles about conflict events worldwide. 
    It automatically extracts key information from news sources and provides comprehensive analytics and visualizations 
    to help researchers, analysts, and policymakers understand global conflict patterns.
    """)
    
    # How it works section
    st.markdown("## How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📰 Data Collection
        - **Manual URL Input**: Analyze individual news articles by URL
        - **Live News Feed**: Automatically fetch recent conflict-related articles via NewsAPI
        - **Web Scraping**: Extract full article content using advanced web loaders
        """)
        
        st.markdown("""
        ### 🤖 AI Analysis Pipeline
        - **Event Classification**: Categorize conflict types using Cohere's custom model
        - **Information Extraction**: Extract key details using natural language processing
        - **Confidence Scoring**: Provide reliability metrics for each analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data Processing
        - **Location Identification**: Determine the country where events occurred
        - **Fatality Estimation**: Extract casualty information when available
        - **Actor Recognition**: Identify key participants in conflicts
        - **Date Extraction**: Determine when events took place
        """)
        
        st.markdown("""
        ### 💾 Storage & Visualization
        - **SQLite Database**: Store all analyzed events for historical tracking
        - **Interactive Dashboard**: Real-time charts and statistics
        - **World Map**: Geographic visualization of conflict hotspots
        """)
    
    # Technical details
    st.markdown("## Technical Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **🤖 AI & NLP**
        - Cohere API for text classification
        - Custom conflict event model
        - Natural language processing
        - Confidence scoring algorithms
        """)
    
    with tech_col2:
        st.markdown("""
        **🌐 Data Sources**
        - NewsAPI for live feeds
        - Web scraping capabilities
        - Mapbox for geocoding
        - Multiple news outlets
        """)
    
    with tech_col3:
        st.markdown("""
        **📊 Visualization**
        - Streamlit interface
        - Plotly charts
        - Folium maps
        - Real-time updates
        """)
    
    # Features section
    st.markdown("## Key Features")
    
    feature_tabs = st.tabs(["Analysis", "Dashboard", "Mapping", "Performance"])
    
    with feature_tabs[0]:
        st.markdown("""
        ### Intelligent Event Analysis
        
        **🎯 What We Extract:**
        - **Event Type**: Conflict classification (e.g., armed conflict, terrorism, protests)
        - **Location**: Country and region identification
        - **Fatalities**: Casualty counts when reported
        - **Key Actors**: Main participants in the conflict
        - **Timeline**: When the event occurred
        - **News Source**: Original reporting outlet
        - **Summary**: AI-generated 2-sentence summary
        
        **🔬 Analysis Process:**
        1. Article content extraction from URL
        2. Text preprocessing and cleaning
        3. AI model inference for classification
        4. Information extraction using NLP
        5. Confidence scoring and validation
        6. Database storage with metadata
        """)
    
    with feature_tabs[1]:
        st.markdown("""
        ### Comprehensive Dashboard
        
        **📈 Visualizations Include:**
        - **Timeline Analysis**: Events over time with interactive controls
        - **Geographic Distribution**: Countries with most events
        - **Event Type Breakdown**: Conflict category analysis
        - **Fatality Statistics**: Casualty data by region
        - **Key Metrics**: Total events, countries affected, fatalities
        
        **🔄 Real-time Updates:**
        - Automatic chart refreshing after new analyses
        - Cached data for improved performance
        - Manual refresh controls
        - Responsive design for all screen sizes
        """)
    
    with feature_tabs[2]:
        st.markdown("""
        ### Interactive World Map
        
        **🌍 Map Features:**
        - **Event Clustering**: Groups nearby events for clarity
        - **Size-based Visualization**: Marker size reflects event frequency
        - **Color Coding**: Intensity-based color mapping
        - **Interactive Tooltips**: Detailed information on hover
        - **Popup Details**: Click for country-specific data
        
        **📍 Geocoding:**
        - Mapbox API integration for accurate coordinates
        - Country-level positioning
        - Automatic location detection from text
        - Fallback mechanisms for unclear locations
        """)
    
    with feature_tabs[3]:
        st.markdown("""
        ### Performance Optimizations
        
        **⚡ Speed Enhancements:**
        - **Smart Caching**: 1-10 minute cache durations based on data type
        - **Lazy Loading**: Charts load progressively with spinners
        - **Pagination**: Large datasets split into manageable pages
        - **Async Processing**: Background analysis with progress indicators
        
        **🔄 Cache Management:**
        - Automatic cache invalidation on data changes
        - Manual refresh controls for users
        - Optimized database queries
        - Minimal API calls through intelligent caching
        """)
    
    # Usage guide
    st.markdown("## How to Use")
    
    usage_steps = st.expander("Step-by-Step Guide", expanded=False)
    with usage_steps:
        st.markdown("""
        ### 🚀 Getting Started
        
        **1. Analyze News Articles**
        - Go to the **Data Entry** page
        - Choose from three input methods:
          - **Manual URL**: Paste a news article URL
          - **Live News**: Browse recent conflict-related articles
          - **Recent Analyses**: View previously analyzed events
        
        **2. Review Analysis Results**
        - Watch the real-time analysis progress
        - Review extracted information and confidence scores
        - Check the **Recent Analyses** tab for your new event
        
        **3. Explore the Dashboard**
        - Navigate to the **Dashboard** page
        - Examine key statistics and trends
        - Interact with charts and the world map
        - Use filters and controls to explore data
        
        **4. Manage Your Data**
        - Use search and filters in Recent Analyses
        - Delete incorrect or duplicate entries
        - Refresh data manually when needed
        - Export or analyze patterns over time
        """)
    
    # Limitations and considerations
    st.markdown("## Limitations & Considerations")
    
    st.warning("""
    **Important Notes:**
    - **AI Accuracy**: While our models are trained on conflict data, no AI system is 100% accurate
    - **Language Support**: Currently optimized for English-language news sources
    - **Real-time Data**: Analysis depends on news article availability and API limits
    - **Geographic Precision**: Location detection is at country level, not city/region specific
    - **Bias Considerations**: Results may reflect biases present in training data and news sources
    """)
    
    # Data sources and credits
    st.markdown("## Data Sources & Credits")
    
    credits_col1, credits_col2 = st.columns(2)
    
    with credits_col1:
        st.markdown("""
        **🤖 AI & APIs:**
        - [Cohere](https://cohere.ai) - Natural Language Processing
        - [NewsAPI](https://newsapi.org) - News article feeds
        - [Mapbox](https://mapbox.com) - Geocoding services
        """)
    
    with credits_col2:
        st.markdown("""
        **🛠️ Technology:**
        - [Streamlit](https://streamlit.io) - Web application framework
        - [Plotly](https://plotly.com) - Interactive charts
        - [Folium](https://folium.readthedocs.io) - Map visualizations
        """)
    

def main():
    try:
        # Add custom CSS for better performance and styling
        st.markdown("""
        <style>
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .stSelectbox > div > div > div {
            background-color: white;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        logger.info(f"Using database: {os.path.abspath(DATABASE_NAME)}")
        update_database_schema()
        
        st.sidebar.title("Navigation")
        # Navigation with clean, professional styling
        st.sidebar.markdown("""
        <style>
        .nav-item {
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .nav-item:hover {
            background-color: #f0f0f0;
        }
        .nav-item.active {
            background-color: #e8f4f8;
            border-left: 3px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        pages = ["Data Entry", "Text Classifier", "Dashboard", "About"]
        selection = st.sidebar.selectbox(
            "Go to", 
            pages,
            key="main_navigation_selectbox"
        )
        page = selection

        if page == "Data Entry":
            data_entry_page()
        elif page == "Text Classifier":
            text_classifier_page()
        elif page == "Dashboard":
            dashboard_page()
        elif page == "About":
            about_page()

        st.sidebar.header("About")
        st.sidebar.info(
            "This app analyzes news articles about conflict events, classifies them into event types, "
            "determines the country where the event occurred, identifies the news source, "
            "estimates the number of fatalities, and provides a brief summary. "
            "Enter a URL of a news article to get started."
        )



        # Add Font Awesome CSS
        st.sidebar.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
            """,
            unsafe_allow_html=True
        )

        # Add GitHub and Company Homepage links with icons side by side
        st.sidebar.markdown(
            """
            <div style="display: flex; justify-content: space-around;">
                <a href="https://github.com/GregXD/conflict-events-tagger" target="_blank" style="text-decoration: none; color: white;">
                    <i class="fab fa-github" style="font-size: 24px;"></i>
                </a>
                <a href="https://www.exchange.design" target="_blank" style="text-decoration: none; color: white;">
                    <i class="fas fa-home" style="font-size: 24px;"></i>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.exception("An unexpected error occurred")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the logs for more information.")


if __name__ == "__main__":
    update_database_schema()
    main()