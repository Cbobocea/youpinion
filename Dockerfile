# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Command to run your app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

# Copy the service account key file to a secure location in your container
COPY service-account-key.json /app/service-account-key.json

