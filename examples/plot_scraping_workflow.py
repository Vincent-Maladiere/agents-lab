
"""
=====================
Web Scraping Workflow
=====================

In this notebook, we explore how `Pydantic-AI <https://ai.pydantic.dev/>`_ workflows
can help us extract structured information from HTML pages. This example can be run
entirely for free, and only requires signing up for
`Groq's free tier <https://groq.com/>`_.

Context
-------

Scraping content from HTML pages has become ubiquitous, with use-cases ranging from
growth marketing to dynamic pricing. A well-known challenge with web scraping is
the polymorphic and ever-changing HTML structure of websites, making scraper automation
difficult to maintain and scale. Developers often harcode the HTML paths of
elements to be fetched, along with the logic for fetching them, resulting in brittle
solutions.

Research in LLMs offers promising avenues for performing web scraping more efficiently
due to the following capabilities:

- longer context window, which can now contains large HTML DOMs
  (`up to 128k tokens with llama 3 <https://huggingface.co/blog/llama31#whats-new-with-llama-31>`_)
- structured output, adhering to predefined JSON schemas
- better reasoning to interpret raw text and extract higher-level information
- faster and cheaper inference, making LLM use economically viable

Strategy
--------

Workflows vs agents
~~~~~~~~~~~~~~~~~~~

In most cases, web scraping does not require user-feedback, web search or tool
invocation. Instead, it typically consists of sequential, acyclic steps.
This makes simple workflows preferable for robustness and predictability
compared to agent-based systems. For an in-depth discussion of the trade-offs between
workflows and agents, refer to Anthropic's blogpost on
`Building effective agents <https://www.anthropic.com/research/building-effective-agents>`_.

HTML DOMs vs Screenshots
~~~~~~~~~~~~~~~~~~~~~~~~

Recent work such as
`WebVoyager (He et al. 2024) <https://arxiv.org/abs/2401.13919>`_, has explored 
using screenshots to perform online actions (e.g. booking a flight) with agent-based
system. For a practical implementation, see this
`Langchain tutorial <https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager>`_.

Currently, the main limitations of the screenshot-based approach include:

- Low average accuracy (~60%), due to the varying complexity of the websites and the
  number of steps required to perform the task.
- Limited choice of visual LLM. Since high definition screenshots are needed to read
  text, only GPT4-V and GPT4-o are adapted to perform these benchmarks.
- Limited use of textual information and HTML DOMs, screenshots rely heavily on visual
  data, while textual information and HTML DOMs remain LLMs' primary mode of operation.

Our use-case is simpler than WebVoyager, as it does not require performing actions or
navigating accross multiple websites. Instead, we deal with a few web pages processed
sequentially.

Given this, our focus is on extracting HTML DOMs, while stripping away non-informative
content such as styles or scripts. Beyond this automatic stripping, we avoid additional
HTML transformation or lookups to keep the workflow as general and maintainable
as possible.

Workflow
~~~~~~~~

Our use-case involves fetching information about car dealerships from a popular French
e-commerce platform called "LeBonCoin" (LBC). To keep this notebook concise, we begin
with a list of dealership URLs, which were previously obtained using another scraping
system.

The objective is to extract information from each dealerships' LBC page and enrich it
with financial data sourced from another website, "pappers.fr".

Our workflow is the following:

.. mermaid::

   flowchart TD
      A(Webdriver) -->|Browse LBC company url| B(LBC Agent)
      B --> |Extract company info|C{Success}
      C --> |Yes|D[Webdriver]
      C -->|No| E[End]
      D --> |Browse Pappers listing using 'company name'|F[Pappers Agent]
      F --> |Find the company page from a list of companies, using name and city|G{Success}
      G --> |Yes|H[Pappers Agent]
      G --> |No|I[End]
      H --> |Extract company financial info|J[Finish]


Our webdriver uses a mix of ``requests`` for static pages (on LBC) and ``selenium`` where
Javascript need to be enabled to access pages (on Pappers).

We define our HTML fetching functions below:
"""

# %%
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from selenium import webdriver
import chromedriver_autoinstaller
from pyvirtualdisplay import Display

# Check if the current version of chromedriver exists and if it doesn't,
# download it automatically and add chromedriver to the path.
chromedriver_autoinstaller.install()


display = Display(visible=0, size=(800, 800))  
display.start()


# We monkey-patch `requests.get` because GitHub CI triggers LBC bot detection.
@dataclass
class Response:
    text: str
    status_code: int = 200


def get_lbc(url, headers=None):
    with open("../doc/_static/lbc_HTML_DOM.txt") as f:
        return Response(text=f.read())


requests.get = get_lbc


def fetch_html_content(url, static_page=True, text_only=False):

    # Headers to mimic a browser request
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    if static_page:
        soup = fetch_requests(url, user_agent)
    else:
        soup = fetch_selenium(url, user_agent)
    
    # Strip all tags, and only return the text content of the page.
    if text_only:
        return soup.get_text(separator=" ").strip()

    return soup


def fetch_requests(url, user_agent):
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        return parse_html(response.text)    
    else:
        raise ValueError(
            f"Failed to fetch the URL. Status code: {response.status_code}"
        )
       

def fetch_selenium(url, user_agent):
    
    chrome_options = webdriver.ChromeOptions()
    options = [
        f"--user-agent={user_agent}",
        "--window-size=1200,1200",
        "--ignore-certificate-errors",
    ]
    for option in options:
        chrome_options.add_argument(option)

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Necessary to give the page time to load.
    time.sleep(3)
    
    return parse_html(driver.page_source)


def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove the following tags
    for tag in soup(["style", "script", "svg", "header", "head"]):
        tag.decompose()

    # Remove the following attributes within tags
    for tag in soup():
        for attribute in ["class", "id", "name", "style"]:
            del tag[attribute]

    return soup


# %%
# To keep this notebook concise and fast to execute, we will use only a single URL.
# Since the HTML structure is not required for this step, we will extract and retain 
# only the text content from the dealership page.
#
# You can open the URL provided to review the outputs generated by the LLM, 
# which will be produced in the next cell.

def print_with_token_count(text):
    print(f"character numbers: {len(text):,}\n")
    print(text)


url_lbc = "https://www.leboncoin.fr/boutique/17050/garage_hamelin.htm"
text_content_lbc = fetch_html_content(url_lbc, static_page=True, text_only=True)
print_with_token_count(text_content_lbc)

# %%
# Next, we pass the raw text content to a LLM. For this example, we choose the
# following:
#
# - Groq: Used as our LLM endpoint, as it provides free access to the llama-3.3-70 model.
# - Pydantic-AI: Selected as our LLM client/framework due to its streamlined approach 
#   to structuring responses, requiring less boilerplate compared to alternatives
#   like LangChain.

from pprint import pprint
import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent


# Enable nested event loop in a notebook, so that we can run asynchrone coroutines in
# pydantic-ai.
nest_asyncio.apply()

# Load GROQ_API_KEY from a source file placed in root.
load_dotenv()

# Our desired structured output.
class CompanyInfoLBC(BaseModel):
    company_name: str
    description: str
    services_provided: str
    number_of_cars: int
    main_phone_number: str
    country: str
    city: str
    full_address: str

model_name = "groq:llama-3.3-70b-versatile"
scraper_system_prompt = """
You are a scrapping assistant, and your goal is to extract company information
from html text provided by the user.
"""

agent_lbc = Agent(
    model_name,
    system_prompt=scraper_system_prompt,
    result_type=CompanyInfoLBC,
)
result_lbc = agent_lbc.run_sync(user_prompt=text_content_lbc)
company_info = result_lbc.data.model_dump()
pprint(company_info)

# %%
# We see that all fields are extracted as desired! Let's also observe the messaging
# sequence of Pydantic-AI:
import json


pprint(
    json.loads(result_lbc.all_messages_json())
)
# %%
# Interestingly, we observe that the framework produced three messages (from top to
# bottom), but when looking at the Groq dev console, we notice that only a single API
# call was made to the LLM.
#
# Here is the Pydantic-AI workflow:
#
# 1. The first message is the request to the model, consisting of two parts: the system
#    prompt and the user prompt, which in this case is the HTML text.
#    Under the hood, 
#    `pydantic-ai adds a structured output tool to the Groq client <https://github.com/pydantic/pydantic-ai/blob/16325844995f18977174638e9c4effc51036704e/pydantic_ai_slim/pydantic_ai/models/groq.py#L125-L132>`_.
# 2. Using this tool, Groq returns a JSON object, which pydantic-ai parses into a
#    Pydantic model.
# 3. Finally, since the LLM indicates completion in step 2, pydantic-ai generates
#    a closing message and returns the result.
#
# To quench our curiosity, here is the structured result tool passed to Groq:

from pprint import pprint


pprint(agent_lbc._result_schema.tool_defs())
# %%
# The next step in our workflow is to enrich the information from LBC with financial
# data sourced from Pappers. This involves two LLM calls:
#
# 1. Generate a Pappers search URL using the company name and access the resulting page.
#    This leads to a list of companies with similar names. We ask the LLM to identify
#    the company that best matches our query, based on the provided name and city.
# 2. Generate a Pappers company URL to access the company's specific page, and then
#    prompt the LLM to extract the desired financial information.
#
# To illustrate the first step, here is an example of how the company list appears:
#
# .. image:: ../_static/pappers_list.png
#     
# Notice that the company we are searching for – located in Perpignan – is the third
# entry on the list!

from urllib.parse import urljoin

PAPPERS_BASE_URL = "https://www.pappers.fr/"


def make_pappers_search_url(company_name):
    query = "+".join(company_name.split())
    return urljoin(PAPPERS_BASE_URL, f"recherche?q={query}")


def make_pappers_company_url(company_href):
    return urljoin(PAPPERS_BASE_URL, company_href)


pappers_search_url = make_pappers_search_url(company_info["company_name"])
print(pappers_search_url)

soup = fetch_html_content(pappers_search_url, static_page=False, text_only=False)
print_with_token_count(str(soup))

# %%
# We retain the HTML tags because they allow us to fetch the href corresponding
# to the company we are looking for. Keeping the HTML list structure helps the LLM
# distinguish between the different items more effectively.
#
# The downside is that the user prompt becomes quite large, exceeding 15,000 characters.
# This increases inference costs and requires using an LLM with a large context window.

class CompanyHref(BaseModel):
    href: str

system_prompt_href = """
    You are a web scraping assistant. Your task is to find the item in a HTML list
    which matches best the query company information. Only returns the href matching
    this item.
"""

agent_pappers_href = Agent(
    model_name,
    system_prompt=system_prompt_href,
    result_type=CompanyHref,
)

query = {k: company_info[k] for k in ["city", "company_name"]}
user_prompt = f"""
    query: <{query}>

    html: <{soup}>
"""
result_href = agent_pappers_href.run_sync(user_prompt)
result_href.data

# %%
# This href corresponds to the Perpignan dealership we are searching for!
# Next, we complete the workflow by fetching financial information from the 
# company's Pappers page.

pappers_company_url = make_pappers_company_url(result_href.data.href)
print(pappers_company_url)

pappers_text_content = fetch_html_content(
    pappers_company_url, static_page=False, text_only=True
)

class FinancialInfo(BaseModel):
    owner_name: str
    owner_age: str
    turnover: int
    social_capital: int

agent_pappers_info = Agent(
    model_name,
    system_prompt=scraper_system_prompt,
    result_type=FinancialInfo,
)

user_prompt = f"html: <{pappers_text_content}>"
pappers_info = agent_pappers_info.run_sync(user_prompt)
financial_info = pappers_info.data.model_dump()
pprint(financial_info)

# %%
# Finally, we store the output in a database and synchronize the lead with our CRM.
company_info.update(financial_info)
pprint(company_info)
