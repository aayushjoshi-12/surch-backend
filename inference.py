from bs4 import BeautifulSoup
import requests
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
cse_id = os.getenv("CSE_ID")
access_token = os.getenv("ACCESS_TOKEN")

template = """
system: you are a summarizer that finds the answer the user is looking for in the context provided by the google search results top links.
keep your answers factual and only generate one response.
give the answer in markdown format.
context: {context}
user: {prompt}
"""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=access_token,
)

llm_chain = prompt | llm


class Item(BaseModel):
    query: str
    api_key: str


def google_search(query, api_key, cse_id, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": api_key, "cx": cse_id, "num": num_results}
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    return results


def get_context(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = ""
    paragraphs = "".join([p.get_text() for p in soup.find_all("p")[:10]])
    return paragraphs


@app.post("/inference")
async def get_inference(Item: Item):
    top_url = [
        result["link"]
        for result in google_search(Item.query, Item.api_key, cse_id, num_results=3)
    ]
    context = r"".join([get_context(url) for url in top_url])
    answer = llm_chain.invoke({"context": context, "prompt": Item.query})
    return {"query": Item.query, "inference": answer}

