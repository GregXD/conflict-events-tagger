```markdown
# Conflict Events Tagger

This Flask application tags conflict events reported in the news using an AI model from OpenAI. The application takes a URL as input, extracts the content, and returns structured information about the conflict event in a markdown table format.

## Features
- Extracts text content from a given URL.
- Analyzes the text to identify and tag conflict event details.
- Returns the tagged information as a structured markdown table.

## Requirements
- Python 3.7+
- Flask
- langchain_community
- OpenAI API key

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/GregXD/conflict-events-tagger.git
cd conflict-events-tagger
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up the OpenAI API key:**

Ensure you have an OpenAI API key and set it as an environment variable:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

On Windows, use:

```bash
set OPENAI_API_KEY='your_openai_api_key'
```

## Usage

1. **Run the Flask application:**

```bash
python app.py
```

2. **Open your web browser and go to:**

```
http://localhost:5000
```

3. **Enter a URL in the form and submit:**

The application will fetch the content from the URL, analyze it, and return the conflict event details in a markdown table format.

## Project Structure

- `app.py`: The main Flask application.
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: List of required Python packages.

## Example

To tag a news source, enter a URL in the form on the web interface. The output will be a markdown table with detailed information about the conflict event, including event ID, date, location, actors involved, and more.
