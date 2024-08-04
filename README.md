# Conflict Events Tagger

This Streamlit application tags and visualizes conflict events reported in the news using fine tunes AI models from Cohere. The application extracts content from given URLs, analyzes the text to identify conflict event details, and presents the information through an interactive dashboard.

## Features

- Extracts text content from provided URLs
- Analyzes text to identify and tag conflict event details using AI
- Stores event data in a SQLite database
- Provides an interactive dashboard for visualizing conflict event data
- Displays event locations on a world map
- Shows various charts and statistics about conflict events

## Project Structure

```
Conflict_Tagger/
│
├── app.py                 # Main Streamlit application file
├── requirements.txt       # Python dependencies
├── conflict_events.db     # SQLite database for storing event data
├── country_coordinates.py # Dictionary of country coordinates for mapping
│
├── templates/
│   └── index.html         # HTML template for the web interface
│
└── README.md              # Project documentation (this file)
```

## Main Components

1. **URL Input and Text Extraction**: Users can input news article URLs. The app extracts the main content from these URLs.

2. **AI Analysis**: 
   - A fine-tune version of Cohere's command model is used for additional text analysis and classification.

3. **Database**: Event data is stored in a SQLite database (`conflict_events.db`).

4. **Dashboard**:
   - Displays key statistics (total events, countries affected, total fatalities)
   - Shows a world map with event locations
   - Presents various charts:
     - Events over time
     - Event type distribution
     - Events by country
     - Fatalities by country

5. **Data Visualization**: Uses libraries like Folium for mapping and Plotly for interactive charts.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/conflict-events-tagger.git
   cd conflict-events-tagger
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     COHERE_API_KEY=your_cohere_api_key
     ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Open the app in your web browser (typically at `http://localhost:8501`).
2. Use the sidebar to navigate between the "Tag Events" and "Dashboard" pages.
3. On the "Tag Events" page, enter a URL of a news article about a conflict event.
4. The app will extract the text, analyze it, and store the event details.
5. View the analyzed events and statistics on the "Dashboard" page.

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- OpenAI for providing the GPT model
- Cohere for their NLP capabilities
- Streamlit for the web application framework
- Folium and Plotly for data visualization