#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Set up directories
mkdir -p data/raw data/processed data/samples
mkdir -p models
