#!/bin/bash

# Neural BCI Backend Startup Script
echo "Starting Neural BCI Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p logs

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Warning: Redis server is not running. Starting Redis..."
    redis-server --daemonize yes
fi

# Initialize database
echo "Initializing database..."
python3 -c "from database import create_tables, init_db; create_tables(); init_db()"

# Start the FastAPI server
echo "Starting FastAPI server..."
python3 main.py

# Deactivate virtual environment on exit
deactivate
