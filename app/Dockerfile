# Dockerfile
# Description: This file contains the instructions for building a Docker image
# for the VAE Anomaly Detection API.

# --- Stage 1: Base Image ---
# We start with an official, slim Python image. This provides a lightweight
# Linux environment with Python pre-installed. Using a specific version
# like 3.9-slim ensures our environment is reproducible.
FROM python:3.9-slim

# --- Stage 2: Set up the Environment ---
# Set the working directory inside the container. All subsequent commands
# will run from this directory.
WORKDIR /app

# Set an environment variable to tell Python not to write .pyc files to disk,
# which is a good practice for containers.
ENV PYTHONDONTWRITEBYTECODE 1
# Tell Python to run in unbuffered mode, which is better for logging in containers.
ENV PYTHONUNBUFFERED 1

# --- Stage 3: Install Dependencies ---
# Copy the requirements file into the container first.
COPY requirements.txt .

# Run the pip install command. By copying only the requirements file first,
# Docker can cache this layer. If we change our app code but not the requirements,
# Docker won't need to re-install all the libraries, making builds much faster.
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code ---
# Now, copy all the application code from our local 'app' directory
# into the container's '/app/app' directory.
COPY ./app /app/app

# Also copy the machine learning models into the container.
COPY ./models /app/models

# --- Stage 5: Expose Port and Run ---
# Tell Docker that the container will listen on port 8000.
# This doesn't actually open the port, but serves as documentation for the user.
EXPOSE 8000

# This is the command that will be run when the container starts.
# It uses 'uvicorn' to launch our FastAPI application, which is located in the
# 'app.main' module and is named 'app'.
# --host 0.0.0.0 is crucial to make the server accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
