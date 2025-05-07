#!/bin/bash

# Build the Docker image
docker build -t bdd_downloader .

# Run the Docker container and open bash inside
docker run -it --name bdd_container -v ~/bdd_data:/data bdd_downloader bash
