from openai import OpenAI
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
client = OpenAI()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load dataset
@st.cache_data
def load_news_data():
    return pd.read_csv("news_data.csv")

df = load_news_data()

# Search similar articles from dataset
def search_similar_articles(news_text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = df['News'].tolist() + [news_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = cosine_sim.argsort()[0, -top_n:][::-1]
    similar = df.iloc[top_indices][['News', 'Status']].to_dict(orient='records')
    return similar

# Google Custom Search API integration
def google_search(query, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": CSE_ID,
        "num": num_results
    }
    response = requests.get(search_url, params=params)
    items = response.json().get("items", [])
    return [
        {"title": item["title"], "link": item["link"], "snippet": item.get("snippet", "")}
        for item in items
    ]

# Define tools OpenAI can call
functions = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_articles",
            "description": "Finds the most relevant articles in the dataset for comparison",
            "parameters": {
                "type": "object",
                "properties": {
                    "news_text": {"type": "string", "description": "User provided news article"},
                    "top_n": {"type": "integer", "default": 5}
                },
                "required": ["news_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Searches the web for credible sources related to a news article",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text, usually the article headline or summary"},
                    "num_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

# Streamlit UI
st.title("Fake News Detector with GEN AI")
user_input = st.text_area("Enter the news article you want to verify:")

if st.button("Analyze"):
    with st.spinner("Analyzing the news article..."):

        # Step 1: Call GPT with tools available
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Is this news real or fake? {user_input}"}],
            tools=functions,
            tool_choice="auto"
        )

        messages = [{"role": "user", "content": f"Is this news real or fake? {user_input}"}]
        tool_calls = response.choices[0].message.tool_calls

        tool_responses = []

        for call in tool_calls:
            args = json.loads(call.function.arguments)
            if call.function.name == "search_similar_articles":
                result = search_similar_articles(**args)
            elif call.function.name == "google_search":
                result = google_search(**args)
            else:
                continue

            tool_responses.append({
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": json.dumps(result)
            })

        # Combine all tool messages
        final_messages = messages + [response.choices[0].message] + tool_responses

        # Final GPT response with all tool evidence
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=final_messages + [
                {
                    "role": "user",
                    "content": f"""
                                    Please determine whether the following news article is Real or Fake based on similar known articles and live web search.

                                    Tasks:
                                    1. Return a verdict: Real or Fake.
                                    2. Confidence score from 0 to 100.
                                    3. Justify your answer using the dataset evidence.
                                    4. Include credible links found from web search.

                                    News:
                                    {user_input}
                                """
                }
            ],
        )

        st.subheader("AI Verdict")
        st.info(final_response.choices[0].message.content)
