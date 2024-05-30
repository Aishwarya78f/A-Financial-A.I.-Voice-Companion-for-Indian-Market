import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import g4f  # Import g4f library for language model interactions

docs = []

# Function to fetch data from URLs
def fetch_data_from_urls(urls, separators, chunk_size, docs):
    data = []

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()  # Extract text from HTML
            data.append(text)
            print("Text fetched from URL:", url)  # Print debug information
            chunks = split_text_with_separators(text, separators, chunk_size)
            for chunk in chunks:
                docs.append(chunk)
                print("Length of docs after adding chunk:", len(docs))  # Print debug information
        else:
            print("Failed to fetch data from URL:", url)  # Print debug information
    return data

# Function to split text into overlapping chunks
def split_text_with_separators(text, separators, chunk_size):
    chunks = []
    chunk = ''
    for char in text:
        chunk += char
        if any(separator in chunk for separator in separators) or len(chunk) >= chunk_size:
            chunks.append(chunk.strip())  # Strip extra whitespace
            chunk = ''
    if chunk:
        chunks.append(chunk.strip())  # Strip extra whitespace
    print("Text split into chunks:", len(chunks))  # Print debug information
    return chunks

# Streamlit app
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "tfidf_index.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data from URLs
    separators = ['\n\n', '\n', '.', ',']
    chunk_size = 3000  # Increased chunk size
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = fetch_data_from_urls(urls, separators, chunk_size, docs)
    
    # Split data
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    
    for doc in data:
       chunks = split_text_with_separators(doc, separators, chunk_size)
       docs.extend(chunks)
    
    # Create TF-IDF vectors and save them to a pickle file
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    main_placeholder.text("TF-IDF Vectorization Started...✅✅✅")
    time.sleep(2)

    # Save the TF-IDF matrix and vectorizer to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump((tfidf_matrix, vectorizer), f)

# Load TF-IDF matrix and vectorizer (if file exists)
try:
    with open(file_path, "rb") as f:
        tfidf_matrix, vectorizer = pickle.load(f)
except FileNotFoundError:
    tfidf_matrix, vectorizer = None, None
    st.info("No pre-processed data found. TF-IDF functionality disabled.")

query = st.text_input("Question: ")

if query and vectorizer is not None:
    # Use the LLM from g4f to generate the answer
    out = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=[{"role": "user", "content": query}],
    )

    # Print the answer
    st.header("Best Answer")
    st.write(out)

elif query:
    st.warning("No pre-processed data found. Please process URLs first.")