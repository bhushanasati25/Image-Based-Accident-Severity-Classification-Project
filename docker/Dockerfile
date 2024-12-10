# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements.txt
COPY ../requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app
COPY ../streamlit_app /app/streamlit_app

# Copy models
COPY ../models /app/models

# Copy label encoder
COPY ../data/processed/label_encoder.json /app/data/processed/label_encoder.json

# Expose port
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
