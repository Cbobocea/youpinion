import os
import json
import openai
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for, flash
import re
from collections import Counter
from openai.error import RateLimitError
import nltk
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'deskjet13g3Ras11??'

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
#credentials_path = r'C:\Users\munianio\Downloads\service-account-key.json'  # path for testing
credentials_path = '/app/service-account-key.json'  # path for live

# Load credentials from the service account key file
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Stop words for text processing
stop_words = set(stopwords.words('english'))

# Function to get YouTube comments
def get_youtube_comments(video_id, max_comments=1000):
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

        for item in response.get("items", []):
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

# Function to summarize comments
def summarize_comments(comments):
    summaries = []
    concatenated_comments = ' '.join(comments)  # Combine comments into one string
    text = concatenated_comments[:4000]  # Limit total input length

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Provide a summary of the following comments: {text}"}
            ],
            max_tokens=100
        )
        summary = response['choices'][0]['message']['content']

        # Replace specific phrases for a more generic output
        summary = summary.replace("The commenter ", "Comments are referring to ")

        summaries.append(summary)
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        summaries.append("Error summarizing comments.")

    return summaries

# Function to get OpenAI embeddings (for clustering if needed)
def get_openai_embeddings(comments):
    embeddings = []
    for comment in comments:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-001",  # or adjust based on your use
                input=comment
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            embeddings.append(np.zeros(1536))  # Adjust based on dimensionality
    return np.array(embeddings)

# Clustering comments
def cluster_comments(embeddings, n_clusters=5):
    if len(embeddings) < n_clusters:
        logging.warning("Not enough samples for clustering.")
        return None  # Handle appropriately if you cannot cluster

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


# Function to find the most common cluster
def find_most_common_cluster(clusters):
    # Ensure you have clusters to process
    if len(clusters) == 0:
        return None, 0  # Handle appropriately if no clusters

    cluster_counts = Counter(clusters)
    most_common_cluster, count = cluster_counts.most_common(1)[0]
    return most_common_cluster, count


# Main execution function
def main(video_id, n_clusters=5, max_comments=1000, batch_size=100):
    try:
        comments = get_youtube_comments(video_id, max_comments)
        cleaned_comments = preprocess_comments(comments)

        # Summarize comments using OpenAI
        all_summaries = summarize_comments(cleaned_comments)

        all_embeddings = []

        for i in range(0, len(all_summaries), batch_size):
            batch_summaries = all_summaries[i:i + batch_size]

            # Get OpenAI embeddings for the summaries
            try:
                embeddings = get_openai_embeddings(batch_summaries)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logging.error(f"Error during embedding generation: {e}")
                flash("Model overloaded, please reload the page and try again with fewer comments.")
                return None, 0, []  # Return early

        # Ensure enough embeddings for clustering
        if len(all_embeddings) < n_clusters:
            logging.warning("Not enough samples for clustering.")
            flash("Not enough comments to cluster. Please try with a different video.")
            return None, 0, all_summaries

        kmeans = cluster_comments(all_embeddings, n_clusters=n_clusters)
        clusters = kmeans.labels_

        # Find most common cluster
        most_common_cluster, count = find_most_common_cluster(clusters)

        # Extract summaries belonging to the most common cluster
        common_cluster_comments = [all_summaries[i] for i in range(len(clusters)) if clusters[i] == most_common_cluster]

        return most_common_cluster, count, common_cluster_comments

    except MemoryError:
        logging.error("Memory overload encountered.")
        flash("Model overloaded, please reload the page and try again with fewer comments.")
        return None, 0, []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        flash("An error occurred while processing. Please try again.")
        return None, 0, []


# Function to extract video ID from the URL
def extract_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]  # For short links
    elif 'youtube.com' in url:
        match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1)
        match = re.search(r'\/embed\/([a-zA-Z0-9_-]{11})|\/watch\?v=([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1) or match.group(2)
    return None


# Flask routes and logic
@app.route('/', methods=['GET', 'POST'])
def home():
    current_year = datetime.now().year  # Get the current year
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
            flash("Invalid YouTube link. Please try again.")
            return redirect(url_for('home'))

    return render_template('index.html', year=current_year)  # Pass the year to the template

#results route
@app.route('/results')
def results():
    cluster = request.args.get('cluster')
    count = request.args.get('count')
    summary = request.args.get('summary')

    return render_template('results.html', cluster=cluster, count=count, summary=summary)

#Terms&Cond route
@app.route('/terms')
def terms():
    return render_template('terms.html')  # Make sure this template exists in your templates folder
#Privacy route
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')  # Ensure privacy.html is in your templates folder


# New route for the About page
@app.route('/about')
def about():
    return render_template('about.html')  # Ensure you have about.html in your templates folder

#Contact us route
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    # Here you would process the form data, e.g. save it to a database or send an email
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']

    # Process the contact message (e.g., send an email, save to a database)

    flash("Thank you for your message! We'll get back to you soon.")
    return redirect(url_for('home'))  # Redirect back to the home page or another page

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)