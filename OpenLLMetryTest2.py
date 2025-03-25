import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging  
 
from openai import OpenAI
from traceloop.sdk import Traceloop

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from traceloop.sdk.decorators import workflow, task

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Configure session with retry strategy
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

# Setup Traceloop to Dynatrace OTLP Endpoint and Auth Token 
os.environ["TRACELOOP_BASE_URL"]=r"https://yex81559.sprint.dynatracelabs.com"
os.environ["TRACELOOP_HEADERS"] = f"Authorization=Api-Token%20accesstokenhere" 

# Test connection to Traceloop Base URL
logger.info(f"Traceloop Base URL: {os.environ['TRACELOOP_BASE_URL']}")
try:
    response = session.get(os.environ["TRACELOOP_BASE_URL"])
    response.raise_for_status()
    logger.info("Successfully connected to Traceloop Base URL")
except requests.exceptions.RequestException as e:
    logger.error(f"Error connecting to Traceloop Base URL: {e}")

# Setup openai key
os.environ["OPENAI_API_KEY"] = "openai-api-key-here"

 
# Initialize Traceloop
try:    
    Traceloop.init(app_name="openai-obs", disable_batch=True)
    logger.info("Traceloop initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Traceloop: {e}")

# OpenAI Client
try:
    openai_client = OpenAI()
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")

@task(name="add_prompt_context")
def add_prompt_context():
    prompt = ChatPromptTemplate.from_template("Explain the business of company {company} in a max of {length} words")
    model = ChatOpenAI()
    chain = prompt | model
    return chain

@task(name="prep_prompt_chain")
def prep_prompt_chain():
    return add_prompt_context()

@workflow(name="ask_question")
def prompt_question():
    chain = prep_prompt_chain()
    return chain.invoke({"company": "Dynatrace", "length" : 5})

if  __name__ == "__main__":
    print(prompt_question())
