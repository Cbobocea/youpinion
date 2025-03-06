import os
import json

import openai
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

# Configure OpenAI API key
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Path to the service account key file in the Docker container
credentials_path = r'C:\Users\munianio\Downloads\service-account-key.json' # path for testing
#credentials_path = '/app/service-account-key.json'  # path for live


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

    # Summarize comments instead of getting embeddings
    all_summaries = summarize_comments(cleaned_comments)

    # You can perform clustering here based on the summaries
    # For this, we need to obtain embeddings for the summaries
    all_embeddings = []  # Initialize as empty

    for i in range(0, len(all_summaries), batch_size):
        batch_summaries = all_summaries[i:i + batch_size]

        # Generate embeddings for the summary texts, you can use OpenAI embeddings here
        try:
            embeddings = get_openai_embeddings(batch_summaries)  # Create this function
            all_embeddings.extend(embeddings)
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            continue  # Skip this batch or handle as necessary

    # Proceed to cluster comments based on their embeddings
    kmeans = cluster_comments(all_embeddings, n_clusters=n_clusters)
    clusters = kmeans.labels_

    # Find most common cluster
    most_common_cluster, count = find_most_common_cluster(clusters)

    # Extract summaries belonging to the most common cluster
    common_cluster_comments = [all_summaries[i] for i in range(len(clusters)) if clusters[i] == most_common_cluster]

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

#Function to summarize comments ( Open AI embedded)
def summarize_comments(comments):
    summaries = []
    concatenated_comments = ' '.join(comments)  # Combine comments into one string
    text = concatenated_comments[:250]  # Limit total input length

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Provide a summary of the following comments: {text}"}
            ],
            max_tokens=100  # Adjust as needed
        )
        summary = response['choices'][0]['message']['content']

        # Simplify the response to remove specific attributions
        summary = summary.replace("The speaker ", "Here is the summary: ")

        summaries.append(summary or "Summary could not be generated.")
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        summaries.append("Error summarizing comments.")

    return summaries


def get_openai_embeddings(comments):
    embeddings = []
    for comment in comments:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-001", # or another embedding model you prefer
                input=comment
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            embeddings.append(np.zeros(1536)  # Assuming dimensionality of 1536 for this model
    return np.array(embeddings)

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