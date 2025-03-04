import os
import json
import nltk
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel, pipeline  # Ensure both are imported
from flask import Flask, render_template, request , send_from_directory
import re
from collections import Counter
from openai.error import RateLimitError
from transformers import BertTokenizer, BertModel, pipeline

app = Flask(__name__)

# Initialize NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Google YouTube API setup
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
DEVELOPER_KEY = 'your_developer_key_here'
credentials_path = r'C:\Users\munianio\Downloads\service-account-key.json'  # Change to your path

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
        inputs = tokenizer(comment, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
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
def main(video_id, n_clusters=5, max_comments=1000):
    comments = get_youtube_comments(video_id, max_comments)
    cleaned_comments = preprocess_comments(comments)
    embeddings = get_bert_embeddings(cleaned_comments)
    kmeans = cluster_comments(embeddings, n_clusters=n_clusters)
    clusters = kmeans.labels_

    most_common_cluster, count = find_most_common_cluster(clusters)

    # Extract comments belonging to the most common cluster
    common_cluster_comments = [cleaned_comments[i] for i in range(len(clusters)) if clusters[i] == most_common_cluster]

    save_clustered_comments(cleaned_comments, clusters)

    return most_common_cluster, count, common_cluster_comments  # Return comments as well

# Function to extract video ID from the URL
def extract_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', url)  # Ensure this line is complete and correct
        if match:
            return match.group(1)
        match = re.search(r'\/embed\/([a-zA-Z0-9_-]{11})|\/watch\?v=([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1) or match.group(2)
    return None

# Function to summarize comments
def summarize_comments(comments):
    text = "\n".join(comments)  # Combine the comments into a single string

    # Use the Hugging Face transformers pipeline to summarize comments
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)  # Adjust parameters as needed
    return summary[0]['summary_text']  # Extract the summary text

# Route for favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.getcwd(), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Flask routes and logic
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

            result = f"The most common cluster is {most_common_cluster} with {count} comments. Here’s a summary:\n\n{summary}"
        else:
            result = "Invalid YouTube link. Please try again."

        return render_template('index.html', result=result)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)