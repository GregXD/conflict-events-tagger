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

st.set_page_config(page_title="Conflict Event Tagger", page_icon="xd_logo.png", layout="wide")

# Constants and Configuration
DATABASE_NAME = os.path.join(os.path.dirname(__file__), 'conflict_events.db')
COHERE_MODEL_ID = os.getenv("EVENT_CLASSIFICATION_MODEL_ID")
GAPMINDER_COLORS = px.colors.qualitative.Set2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and set up Cohere API
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    st.error("COHERE_API_KEY environment variable is not set.")
    st.stop()
co = cohere.Client(cohere_api_key)

# Database functions
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        conn.close()

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
        conn = sqlite3.connect('conflict_events.db')  # Replace with your actual database name
        c = conn.cursor()
        c.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
        logger.info(f"Deleted event with ID: {event_id}")
    except sqlite3.Error as e:
        logger.error(f"Error deleting event: {e}")
        raise
    finally:
        conn.close()

def delete_event_callback(event_id):
    try:
        delete_event(event_id)
        st.session_state.delete_success = f"Event {event_id} deleted successfully."
    except Exception as e:
        st.session_state.delete_error = f"Error deleting event {event_id}: {e}"



def fetch_analyses():
    analyses = execute_db_query("SELECT id, url, event_type, confidence, country, news_source, fatalities, summary, event_date, key_actors FROM events ORDER BY event_date DESC")
    return [(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], 
             datetime.strptime(row[8], '%Y-%m-%d').date() if row[8] else None,
             row[9]) 
            for row in analyses]

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
            logger.info("Results inserted into the database")

            result_queue.put(("progress", "Analysis completed successfully."))
            result_queue.put(("done", None))
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
                        elif msg_type == "done":
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


# Chart creation functions
def create_events_by_country_chart():
    data = execute_db_query("SELECT country, COUNT(*) as event_count FROM events GROUP BY country ORDER BY event_count DESC LIMIT 10")
    df = pd.DataFrame(data, columns=['Country', 'Event Count'])
    
    fig = px.bar(df, x='Country', y='Event Count', text='Event Count',
                 title='Top 10 Countries by Number of Events',
                 labels={'Event Count': 'Number of Events'},
                 template='ggplot2', color='Country',
                 color_discrete_sequence=GAPMINDER_COLORS)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
    return fig

def create_event_type_chart():
    data = execute_db_query("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
    df = pd.DataFrame(data, columns=['Event Type', 'Count'])
    fig = px.bar(df, x='Event Type', y='Count', text='Count',
                 title='Distribution of Event Types',
                 labels={'Count': 'Number of Events'},
                 template='ggplot2', color='Event Type',
                 color_discrete_sequence=GAPMINDER_COLORS)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
    return fig

def create_fatalities_by_country_chart():
    data = execute_db_query("SELECT country, SUM(CASE WHEN fatalities != 'Unknown' THEN CAST(fatalities AS INTEGER) ELSE 0 END) as total_fatalities FROM events GROUP BY country")
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

def create_map():
    data = execute_db_query("SELECT country, fatalities, event_type FROM events WHERE country != ''")
    df = pd.DataFrame(data, columns=['Country', 'Fatalities', 'Event Type'])
    
    # Convert 'Unknown' to 0 and ensure all values are numeric
    df['Fatalities'] = pd.to_numeric(df['Fatalities'].replace({'Unknown': '0', '': '0'}), errors='coerce').fillna(0)
    
    df_grouped = df.groupby('Country').agg({'Fatalities': 'sum', 'Event Type': 'count'}).reset_index()
    
    # Check if there's any data
    if df_grouped.empty:
        return None

    # Ensure we have at least two distinct values for the colormap
    fatality_values = sorted(df_grouped['Fatalities'].unique())
    if len(fatality_values) < 2:
        fatality_values = [0, 1]  # Default range if there's only one unique value or no values
    
    m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(m)
    
    # Create the colormap with the sorted values
    colormap = LinearColormap(colors=['green', 'yellow', 'red'], vmin=min(fatality_values), vmax=max(fatality_values))
    
    geolocator = Nominatim(user_agent="conflict_event_app")
    for _, row in df_grouped.iterrows():
        try:
            location = geolocator.geocode(row['Country'])
            if location:
                folium.CircleMarker(
                    location=[location.latitude, location.longitude],
                    radius=min(row['Fatalities'], 30),
                    popup=f"Country: {row['Country']}<br>Fatalities: {row['Fatalities']}<br>Events: {row['Event Type']}",
                    color=colormap(row['Fatalities']),
                    fill=True,
                    fill_color=colormap(row['Fatalities']),
                    fill_opacity=0.7
                ).add_to(marker_cluster)
        except Exception as e:
            st.warning(f"Could not locate {row['Country']}: {str(e)}")
    
    colormap.add_to(m)
    colormap.caption = 'Fatalities'
    return m

def create_events_over_time_chart():
    data = execute_db_query("SELECT event_date, COUNT(*) as event_count FROM events GROUP BY event_date ORDER BY event_date")
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

def data_entry_page():
    st.title("Conflict Event Classifier")
    url = st.text_input("Enter a news article URL:")
    if st.button("Analyze"):
        if url:
            try:
                result = tag_news_source(url)
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

    st.subheader("Recent Analyses")

    # Fetch analyses from the database
    analyses = fetch_analyses()

    # Event type filter
    event_types = ['All'] + sorted(set(row[2] for row in analyses))
    selected_event_type = st.selectbox("Filter by Event Type", event_types, key="data_entry_event_type")

    # Search bar
    search_term = st.text_input("Search in all fields")

    # Filter the analyses
    filtered_analyses = analyses
    if selected_event_type != 'All':
        filtered_analyses = [row for row in filtered_analyses if row[2] == selected_event_type]
    
    if search_term:
        filtered_analyses = [row for row in filtered_analyses if any(search_term.lower() in str(cell).lower() for cell in row)]

    # Display the table with headers
    if filtered_analyses:
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
        for row in filtered_analyses:
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 2, 1, 2, 2, 1, 4, 2, 1])
            with col1:
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
                if st.button('🗑️', key=f"delete_{row[0]}"):
                    try:
                        delete_event(row[0])
                        st.success(f"Event {row[0]} deleted successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting event {row[0]}: {e}")
    else:
        st.info("No analyses available.")

def dashboard_page():
    st.title("Conflict Event Dashboard")

    # Key Statistics at the top
    st.markdown("### Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    total_events = execute_db_query("SELECT COUNT(*) FROM events")[0][0]
    total_countries = execute_db_query("SELECT COUNT(DISTINCT country) FROM events")[0][0]
    total_fatalities = execute_db_query("SELECT SUM(CAST(fatalities AS INTEGER)) FROM events WHERE fatalities != 'Unknown'")[0][0]

    with col1:
        st.markdown(
            """
            <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;">
                <h3 style="color: #3366cc;">Total Events</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
            """.format(total_events),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;">
                <h3 style="color: #dc3912;">Countries Affected</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
            """.format(total_countries),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;">
                <h3 style="color: #ff9900;">Total Fatalities</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
            """.format(total_fatalities or 0),
            unsafe_allow_html=True
        )

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the map
    st.subheader("Event Locations")
    map_obj = create_map()
    if map_obj:
        folium_static(map_obj, width=1000, height=400)
    else:
        st.warning("No data available to display the map.")

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Events over time chart
    events_over_time = create_events_over_time_chart()
    if events_over_time:
        st.plotly_chart(events_over_time, use_container_width=True)
    else:
        st.warning("No data available to display events over time.")

    # Create tabs for different chart views
    tab1, tab2 = st.tabs(["Events Overview", "Fatalities Overview"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            event_type_chart = create_event_type_chart()
            if event_type_chart:
                st.plotly_chart(event_type_chart, use_container_width=True)
            else:
                st.warning("No data available for event type distribution.")
        with col2:
            events_by_country_chart = create_events_by_country_chart()
            if events_by_country_chart:
                st.plotly_chart(events_by_country_chart, use_container_width=True)
            else:
                st.warning("No data available for events by country.")

    with tab2:
        fatalities_chart = create_fatalities_by_country_chart()
        if fatalities_chart:
            st.plotly_chart(fatalities_chart, use_container_width=True)
        else:
            st.warning("No data available for fatalities by country.")


def dashboard_page():
    st.title("Conflict Event Dashboard")

    # Create a centered container with max width
    container = st.container()
    with container:
        # Set the maximum width of the content
        st.markdown(
            """
            <style>
            .reportview-container .main .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Key Statistics at the top
        st.markdown("### Key Statistics")
        col1, col2, col3 = st.columns(3)
        
        total_events = execute_db_query("SELECT COUNT(*) FROM events")[0][0]
        total_countries = execute_db_query("SELECT COUNT(DISTINCT country) FROM events")[0][0]
        total_fatalities = execute_db_query("SELECT SUM(CAST(fatalities AS INTEGER)) FROM events WHERE fatalities != 'Unknown'")[0][0]

        with col1:
            st.metric(label="Total Events", value=total_events)

        with col2:
            st.metric(label="Countries Affected", value=total_countries)

        with col3:
            st.metric(label="Total Fatalities", value=total_fatalities or 0)

        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)

        # Display the map
        st.subheader("Event Locations")
        map_obj = create_map()
        if map_obj:
            # Use st.components.v1.html to make the map responsive
            map_html = folium.Figure().add_child(map_obj).render()
            st.components.v1.html(
                f"""
                <div style="width:100%; height:400px;">
                    {map_html}
                </div>
                """,
                height=400,
            )
        else:
            st.warning("No data available to display the map.")

        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)

        # Events over time chart
        events_over_time = create_events_over_time_chart()
        if events_over_time:
            st.plotly_chart(events_over_time, use_container_width=True)
        else:
            st.warning("No data available to display events over time.")

        # Create tabs for different chart views
        tab1, tab2 = st.tabs(["Events Overview", "Fatalities Overview"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                event_type_chart = create_event_type_chart()
                if event_type_chart:
                    st.plotly_chart(event_type_chart, use_container_width=True)
                else:
                    st.warning("No data available for event type distribution.")
            with col2:
                events_by_country_chart = create_events_by_country_chart()
                if events_by_country_chart:
                    st.plotly_chart(events_by_country_chart, use_container_width=True)
                else:
                    st.warning("No data available for events by country.")

        with tab2:
            fatalities_chart = create_fatalities_by_country_chart()
            if fatalities_chart:
                st.plotly_chart(fatalities_chart, use_container_width=True)
            else:
                st.warning("No data available for fatalities by country.")


def main():
    try:
        logger.info(f"Using database: {os.path.abspath(DATABASE_NAME)}")
        update_database_schema()
        st.sidebar.title("Navigation")
        pages = {
            "Data Entry": "📝",
            "Dashboard": "📊"
        }
        selection = st.sidebar.selectbox(
            "Go to", 
            [f"{icon} {page}" for page, icon in pages.items()],
            key="main_navigation_selectbox"
        )
        page = selection.split(" ", 1)[1]

        if page == "Data Entry":
            data_entry_page()
        elif page == "Dashboard":
            dashboard_page()

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