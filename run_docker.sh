#!/bin/bash

# Build the Docker image
docker build -t my-python-app .

# Run the Docker container and open bash inside
docker run -it --rm my-python-app bash