#!/bin/bash

# AIAP 21 Technical Assessment - Run Script
# This script runs the main solution for the assessment

echo "AIAP 21 Technical Assessment - Starting Solution"
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || venv\Scripts\activate 2>/dev/null

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: data directory not found"
    exit 1
fi

# Check if gas_monitoring.db exists
if [ ! -f "data/gas_monitoring.db" ]; then
    echo "Warning: gas_monitoring.db not found in data directory"
    echo "Please ensure the database file is placed in the data/ directory"
fi

# Run the main solution
echo "Running main solution..."
python src/main.py

echo "Solution completed!"
