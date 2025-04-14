# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip

RUN pip install notebook

# Clone and install Ultralytics YOLOv8
RUN pip install ultralytics opencv-python

# Copy your project files into the container
COPY . /app

# Run detection script by default
CMD ["python", "main.py"]


