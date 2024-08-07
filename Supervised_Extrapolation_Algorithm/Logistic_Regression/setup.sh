#!/bin/bash

# Update package list and install Java JDK 8
echo "Installing Java JDK 8..."
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk-headless

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create a directory for Weka if it doesn't exist
mkdir -p weka

# Download Weka if it doesn't already exist
if [ ! -f "weka/weka-3-8-5-azul-zulu-linux.zip" ]; then
    echo "Downloading Weka..."
    wget http://prdownloads.sourceforge.net/weka/weka-3-8-5-azul-zulu-linux.zip -O weka/weka-3-8-5-azul-zulu-linux.zip
fi

# Unzip Weka if it hasn't been unzipped
if [ ! -d "weka/weka-3-8-5" ]; then
    echo "Unzipping Weka..."
    unzip weka/weka-3-8-5-azul-zulu-linux.zip -d weka
fi

echo "Setup complete. You can now run the script with 'python3 src/test.py'."

