#!/usr/bin/env bash
# Build script for Render

# Install apt packages
apt-get update
apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
