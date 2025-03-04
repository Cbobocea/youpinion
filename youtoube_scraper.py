import os
import json
import nltk
import openai
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import BertTokenizer, BertModel
import torch
import random
from collections import Counter

# Initialize NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Set your OpenAI API key here
openai.api_key = 'your_openai_api_key_here'

# Google YouTube API setup
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
DEVELOPER_KEY = 'your_developer_key_here'
credentials_path = r'C:\Users\munianio\Downloads\service-account-key.json'  # Change this to the path of your service account JSON key file


# Initialize YouTube API client
def get_youtube_comments(video_id, max_comments=1000):
    # Load credentials and build the YouTube API client
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    youtube = build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

    # Request for the comments
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,  # Maximum allowed per request
            pageToken=next_page_token  # Use the nextPageToken for pagination
        )
        response = request.execute()

        # Extract comments from response
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check if there is a next page of comments
        next_page_token = response.get("nextPageToken")

        # If there is no next page token, break the loop
        if not next_page_token:
            break

    return comments


# Preprocessing and cleaning the comments
def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        comment = comment.lower()  # Convert to lowercase
        comment = ' '.join([word for word in comment.split() if word not in stop_words])  # Remove stopwords
        cleaned_comments.append(comment)
    return cleaned_comments


# BERT Embeddings and Clustering
def get_bert_embeddings(comments):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Use mean of last hidden state
        embeddings.append(embedding)
    return np.array(embeddings).squeeze()


# Apply KMeans clustering
def cluster_comments(embeddings, n_clusters=5):
    # Standardize embeddings before clustering
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    return kmeans


# Save results to a file with UTF-8 encoding
def save_clustered_comments(comments, clusters, filename="sorted_comments_by_cluster.txt"):
    clustered_comments = {i: [] for i in range(len(set(clusters)))}
    for comment, cluster in zip(comments, clusters):
        clustered_comments[cluster].append(comment)

    # Write sorted comments by cluster to a file using UTF-8 encoding
    with open(filename, "w", encoding="utf-8") as f:  # specify encoding as utf-8
        for cluster, cluster_comments in clustered_comments.items():
            f.write(f"\nCluster {cluster}:\n")
            for comment in cluster_comments:
                f.write(f"{comment}\n")
            f.write("\n")


# Function to find the most common cluster
def find_most_common_cluster(clusters):
    # Count the frequency of each cluster
    cluster_counts = Counter(clusters)

    # Find the most common cluster
    most_common_cluster, count = cluster_counts.most_common(1)[0]
    return most_common_cluster, count


# Main Execution Function
def main(video_id, n_clusters=5, max_comments=1000):
    print("Extracting comments...")
    comments = get_youtube_comments(video_id, max_comments)
    print(f"Extracted {len(comments)} comments.")

    print("Preprocessing comments...")
    cleaned_comments = preprocess_comments(comments)

    print("Getting BERT embeddings...")
    embeddings = get_bert_embeddings(cleaned_comments)

    print(f"Clustering comments into {n_clusters} clusters...")
    kmeans = cluster_comments(embeddings, n_clusters=n_clusters)
    clusters = kmeans.labels_

    # Find the most common cluster
    most_common_cluster, count = find_most_common_cluster(clusters)
    print(f"The most common cluster is {most_common_cluster} with {count} comments.")

    print("Saving clustered comments to file...")
    save_clustered_comments(cleaned_comments, clusters)
    print("Clustering completed and results saved to 'sorted_comments_by_cluster.txt'.")


# Run the script
if __name__ == "__main__":
    video_id = input("Enter YouTube video ID: ")
    n_clusters = int(input("Enter number of clusters: "))
    main(video_id, n_clusters)
