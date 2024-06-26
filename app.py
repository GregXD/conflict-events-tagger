# MIT License
#
# Copyright (c) 2024 Exchange.Design
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI 
import os
import logging
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = api_key

def tag_news_source(url):
    # Load the text content from the URL
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])

    # Define the prompt template
    prompt = PromptTemplate(
        template="""You are a bot designed to analyze conflict events as reported in the news. When given a news source, tag the news source by the following fields. If the source provided does not appear to be about a conflict event, tell the user. Provide the output as a markdown table:

Field	Description
EVENT_ID_CNTY	Unique identifier for each event, consisting of a number and country acronym.
EVENT_DATE	Date when the event occurred, recorded as Year-Month-Day (e.g., 2023-02-16).
YEAR	Year when the event occurred.
TIME_PRECISION	Numeric code (1-3) indicating the level of precision of the event date: 1 = exact day, 2 = within a week, 3 = within a month.
DISORDER_TYPE	Disorder category the event belongs to (e.g., Political violence, Demonstrations, Strategic developments).
EVENT_TYPE	Category of the event (e.g., Battles, Riots).
SUB_EVENT_TYPE	More specific categorization of the event (e.g., Armed clash, Peaceful protest).
ACTOR1	Primary actor involved in the event.
ASSOC_ACTOR_1	Associated actor(s) for the primary actor.
INTER1	Numeric code (0-8) indicating the type of the primary actor.
ACTOR2	Secondary actor involved in the event.
ASSOC_ACTOR_2	Associated actor(s) for the secondary actor.
INTER2	Numeric code (0-8) indicating the type of the secondary actor.
INTERACTION	Two-digit numeric code combining the interaction codes of both actors.
CIVILIAN_TARGETING	Indicates if civilians were targeted (e.g., Civilians targeted or blank).
ISO	ISO country code.
REGION	Geographical region of the event (e.g., Eastern Africa).
COUNTRY	Country where the event occurred (e.g., Ethiopia).
ADMIN1	First-level administrative division (e.g., state, province).
ADMIN2	Second-level administrative division (e.g., county, district).
ADMIN3	Third-level administrative division (e.g., sub-district).
LOCATION	Specific location where the event occurred (e.g., Abomsa).
LATITUDE	Latitude coordinate of the event location in four decimal degrees notation (e.g., 8.5907).
LONGITUDE	Longitude coordinate of the event location in four decimal degrees notation (e.g., 39.8588).
GEO_PRECISION	Numeric code (1-3) indicating the level of certainty of the location recorded for the event: 1 = exact location, 2 = approximate location, 3 = broad location.
SOURCE:	Sources used to record the event, separated by a semicolon.
SOURCE_SCALE: Geographic closeness of the sources to the event (e.g., Local partner, National).
NOTES: Two Sentence description of the event.
FATALITIES Number of reported fatalities resulting from the event.
TAGS Additional structured information about the event, separated by a semicolon.
TIMESTAMP Unix timestamp representing the exact date and time the event was uploaded to the ACLED API.
Text: {text}

Output:""",
        input_variables=["text"],
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=2048)
    chain = LLMChain(prompt=prompt, llm=llm)
 
    # Create the LLMChain
    chain = LLMChain(prompt=prompt, llm=llm)
        # Run the chain and return the output
    return chain.run(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        logging.debug(f"Received URL: {url}")
        result = tag_news_source(url)
        logging.debug(f"Result: {result}")
        return jsonify(result=result)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)