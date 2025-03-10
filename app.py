from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import gzip
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import openai
import os
import re
import requests

# Initialize Flask app
app = Flask(__name__)

# ----- Utility functions for API key storage -----
CONFIG_FILE = "config.json"

def load_api_key():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("OPENAI_API_KEY")
    except FileNotFoundError:
        return None

def store_api_key(key):
    config = {"OPENAI_API_KEY": key}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    # Update OpenAI's client directly
    openai.api_key = key

# Set the OpenAI API key from storage if it exists
api_key = load_api_key()
if api_key:
    openai.api_key = api_key

# ----- Download Data Files if Not Present -----
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename

data_file = "AMAZON_FASHION.json.gz"
meta_file = "meta_AMAZON_FASHION.json.gz"

if not os.path.exists(data_file):
    print(f"Downloading {data_file} ...")
    download_file("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz", data_file)

if not os.path.exists(meta_file):
    print(f"Downloading {meta_file} ...")
    download_file("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_AMAZON_FASHION.json.gz", meta_file)

# ----- Data Loading -----
data = []
with gzip.open(data_file, 'rt', encoding='utf-8') as f:
    for l in f:
        data.append(json.loads(l.strip()))

metadata = []
with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
    for l in f:
        metadata.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)
df = df[df['reviewText'].notna()]
df_meta = pd.DataFrame.from_dict(metadata)

# ----- Helper functions -----
def display_topics(model, feature_names, no_top_words):
    topic_summaries = []
    for topic_idx, topic in enumerate(model.components_):
        topic_summaries.append(" ".join([feature_names[i]
                                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topic_summaries

def analyze_product_reviews(mode, reviews_data):
    if mode == 'positive':
        prompt = (
            "I have collected a set of positive customer reviews for a product. These reviews reflect what customers enjoy and appreciate about the product. "
            "Please analyze the content of these reviews to identify and summarize the key positive aspects and features that customers highlight. "
            "Focus on understanding what makes customers happy with this product, any specific features or qualities they frequently praise, and any recurring patterns of satisfaction you observe.\n\n"
            + str(reviews_data)
        )
    elif mode == 'negative':
        prompt = (
            "I have performed an LDA (Latent Dirichlet Allocation) analysis on a collection of product reviews and obtained the following topics. "
            "Each topic is represented by a list of its most significant words. Please analyze these topics to identify and summarize the main issues or problems customers have with the product.\n\n"
            + str(reviews_data)
        )
    else:
        return "Invalid mode selected."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

# ----- Routes -----
@app.route('/', methods=['GET'])
def index():
    # If API key is not set, redirect to key entry page
    if not openai.api_key:
        return redirect(url_for('set_api_key'))
    return render_template('index.html')

@app.route('/set_api_key', methods=['GET', 'POST'])
def set_api_key():
    if request.method == 'POST':
        key = request.form.get('openai_api_key')
        if key:
            store_api_key(key)
            return redirect(url_for('index'))
        else:
            return render_template('set_api_key.html', error="Please enter a valid API key.")
    return render_template('set_api_key.html')

@app.route('/product_info', methods=['POST'])
def product_info():
    asin = request.form['asin']
    product_df = df[df['asin'] == asin]

    if product_df.empty:
        return render_template('index.html', error="No product found with this ASIN.")

    ratings_count = product_df['overall'].value_counts().sort_index().to_dict()
    product_meta = df_meta[df_meta['asin'] == asin].iloc[0]
    product_title = product_meta.get('title', 'No Title')
    product_description = product_meta.get('description', 'No Description')

    return render_template('product_info.html', title=product_title, description=product_description, asin=asin, ratings_count=ratings_count)

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    asin = request.form['asin']
    review_type = request.form['review_type']

    if review_type == 'positive':
        filtered_df = df[(df['asin'] == asin) & (df['overall'] >= 4)]['reviewText']
    else:  # Negative
        filtered_df = df[(df['asin'] == asin) & (df['overall'] <= 3)]['reviewText']

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(2, 2))
    data_vectorized = vectorizer.fit_transform(filtered_df)

    lda = LDA(n_components=15, random_state=0)
    lda.fit(data_vectorized)

    tf_feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = display_topics(lda, tf_feature_names, 15)
    chatgpt_response = analyze_product_reviews(review_type, top_words_per_topic)

    return render_template('results.html', lda_topics=top_words_per_topic, chatgpt_response=chatgpt_response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=False)


