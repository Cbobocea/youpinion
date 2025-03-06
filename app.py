import os
import json
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request , send_from_directory , redirect , url_for , flash , request
import re
from collections import Counter
from openai.error import RateLimitError
from transformers import BertTokenizer, BertModel, pipeline
import nltk
import logging

#Set up logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

# Initialize NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Google YouTube API setup
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
DEVELOPER_KEY = 'AIzaSyCnuvkeTxoZdFQifWc2624JpN5NvQcYj4Q'

# Path to the service account key file in the Docker container
#credentials_path = r'C:\Users\munianio\Downloads\service-account-key.json' # path for testing
credentials_path = '/app/service-account-key.json'  # path for live
# Load credentials from the service account key file
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Stop words for text processing
stop_words = set(stopwords.words('english'))

# Create a summarization pipeline
summarizer = pipeline("summarization")

# Function to get YouTube comments
def get_youtube_comments(video_id, max_comments=1000):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    youtube = build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Preprocessing comments
def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        comment = comment.lower()
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        cleaned_comments.append(comment)
    return cleaned_comments

# Getting BERT embeddings
def get_bert_embeddings(comments):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []

    for comment in comments:
        # Limit the length of the comment to avoid excessive input size
        if len(comment) > 512:  # Adjust as appropriate
            comment = comment[:512]

        inputs = tokenizer(comment, return_tensors="pt", max_length=512, truncation=True, padding=True)

        try:
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            embeddings.append(np.zeros((768,)))  # Append a zero vector for failed cases

    return np.array(embeddings).squeeze()

# Clustering comments
def cluster_comments(embeddings, n_clusters=5):
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

# Save clustered comments
def save_clustered_comments(comments, clusters, filename="sorted_comments_by_cluster.txt"):
    clustered_comments = {i: [] for i in range(len(set(clusters)))}
    for comment, cluster in zip(comments, clusters):
        clustered_comments[cluster].append(comment)

    with open(filename, "w", encoding="utf-8") as f:
        for cluster, cluster_comments in clustered_comments.items():
            f.write(f"\nCluster {cluster}:\n")
            for comment in cluster_comments:
                f.write(f"{comment}\n")
            f.write("\n")

# Function to find the most common cluster
def find_most_common_cluster(clusters):
    cluster_counts = Counter(clusters)
    most_common_cluster, count = cluster_counts.most_common(1)[0]
    return most_common_cluster, count

# Main execution function
def main(video_id, n_clusters=5, max_comments=1000, batch_size=100):
    comments = get_youtube_comments(video_id, max_comments)
    cleaned_comments = preprocess_comments(comments)

    all_embeddings = []
    for i in range(0, len(cleaned_comments), batch_size):
        batch_comments = cleaned_comments[i:i + batch_size]
        embeddings = get_bert_embeddings(batch_comments)
        all_embeddings.extend(embeddings)

    kmeans = cluster_comments(all_embeddings, n_clusters=n_clusters)
    clusters = kmeans.labels_

    most_common_cluster, count = find_most_common_cluster(clusters)

    # Extract comments belonging to the most common cluster
    common_cluster_comments = [cleaned_comments[i] for i in range(len(clusters)) if clusters[i] == most_common_cluster]

    save_clustered_comments(cleaned_comments, clusters)

    return most_common_cluster, count, common_cluster_comments

# Function to extract video ID from the URL
def extract_video_id(url):
    if 'youtu.be' in url:
        # For short links (youtu.be)
        return url.split('/')[-1].split('?')[0]  # Split and remove any query parameters
    elif 'youtube.com' in url:
        # For full links (youtube.com)
        match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1)
        match = re.search(r'\/embed\/([a-zA-Z0-9_-]{11})|\/watch\?v=([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1) or match.group(2)
    return None

# Function to summarize comments
def summarize_comments(comments):
    summaries = []
    for comment in comments:
        text = comment[:512]  # Limit to relevant length
        max_len = min(150, len(text))  # set max length based on input length

        try:
            summary = summarizer(text, max_length=max_len, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            summaries.append("Error summarizing comment.")

    return summaries


# Flask routes and logic
from flask import redirect, url_for, flash, request


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        youtube_link = request.form['youtube_link']
        video_id = extract_video_id(youtube_link)

        if video_id:
            n_clusters = 5
            most_common_cluster, count, common_cluster_comments = main(video_id, n_clusters)

            # Call the summarization function
            summary = summarize_comments(common_cluster_comments)

            # Redirect to results page with necessary data passed as query parameters
            return redirect(url_for('results', cluster=most_common_cluster, count=count, summary=summary))

        else:
            flash("Invalid YouTube link. Please try again.")  # Use flash for user-friendly messages
            return redirect(url_for('home'))

    return render_template('index.html')


@app.route('/results')
def results():
    cluster = request.args.get('cluster')
    count = request.args.get('count')
    summary = request.args.get('summary')

    return render_template('results.html', cluster=cluster, count=count, summary=summary)



if __name__ == '__main__':
    app.run(debug=True)