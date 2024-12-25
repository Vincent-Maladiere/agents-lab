# %%
import os
from langchain_huggingface import HuggingFaceEndpoint


# The most powerful (12B), open weight model from Mistral, available for free on the
# HF Hub.
REPO_ID = "mistralai/Mistral-Nemo-Instruct-2407"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
user_msg = "Describe this job title in two sentences: Sr BDR"

llm = HuggingFaceEndpoint(repo_id=REPO_ID, huggingfacehub_api_token=HF_API_TOKEN)
llm.invoke(user_msg)

# %%
# Fetch a page from LeBonCoin and returns its raw content
import requests
from bs4 import BeautifulSoup
from selenium import webdriver


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
    
    options = webdriver.ChromeOptions(f"--user-agent={user_agent}")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    return parse_html(driver.page_source)


def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["style", "script", "svg"]):
        tag.decompose()
    return soup

# %%

url = "https://www.leboncoin.fr/boutique/17050/garage_hamelin.htm"
text_content = fetch_html_content(url, static_page=True, text_only=True)
print(text_content)

# %%
import json


query = f"""
    You are a scrapping assistant, and your goal is to extract the following fields
    from the raw text between <>. Only output a JSON format as follows:

    {{"Company name": "...",
    "Description": "...",
    "Services provided": "...",
    "Number of cars": "...",
    "Main phone number": "...",
    "Country": "...",
    "City": "...",
    "Full Address": "..."}}

    <{text_content}>
"""

answer = llm.invoke(query)
lbc_info = json.loads(answer.content)
lbc_info

# %%
# Fetch additional info on Pappers.fr
# 1. Create an url based on Company Name + City
# 2. Fetch the list
# 3. Select the most likely page and its url
# 4. - Access the page and get the required info
#    - Or skip and return an empty dict

from urllib.parse import urljoin

PAPPERS_BASE_URL = "https://www.pappers.fr/"


def make_pappers_list_url(company_name):
    query = "+".join(company_name.split())
    return urljoin(PAPPERS_BASE_URL, f"recherche?q={query}")


url = make_pappers_list_url(lbc_info["Company name"])
soup = fetch_html_content(url)

print(soup.prettify())

# %%

summary = {k: lbc_info[k] for k in ["City", "Company name"]}

query = f"""
    You are a web scraping assistant. Your task is to find the item in the HTML list
    between <> which matches best the query company information below:

    query: {summary}

    Only output a JSON format containing the href matching the item, like the following
    example:
    {{"href": "/entreprise/my_awesome_company-1234"}}

    If you can't find an item matching the query, only output the following JSON:
    {{"href": null}}.

    Don't output anything else than a JSON.

    <{soup}>
"""

answer = llm.invoke(query)
href = json.loads(answer.content.strip())
href

# %%

url = urljoin(PAPPERS_BASE_URL, href["href"])
text_content = fetch_html_content(url, static_page=False, text_only=True)

query = f"""
    You are a scrapping assistant, and your goal is to extract the following fields
    from the raw text between <>. Only output a JSON format as follows:

    {{"Owner name": "...",
    "Owner age": "...",
    "Turnover": "...",
    "Social capital": "..."}}

    <{text_content}>
"""

answer = llm.invoke(query)
pappers_info = json.loads(answer.content)
pappers_info

# %%