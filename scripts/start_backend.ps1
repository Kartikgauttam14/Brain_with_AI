# Neural BCI Backend Startup Script for Windows
Write-Host "Starting Neural BCI Backend..." -ForegroundColor Green

# Check if Python is installed
try {
    python --version
} catch {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path "..\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv ..\venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ..\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create necessary directories
if (-not (Test-Path -Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}
if (-not (Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Check if Redis is installed and running
# Note: Redis is not natively available on Windows, so we'll just warn the user
Write-Host "Warning: Redis server is required but may not be running on Windows." -ForegroundColor Yellow
Write-Host "Please ensure Redis is installed and running, or modify the code to work without Redis." -ForegroundColor Yellow

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Yellow
python -c "from database import create_tables, init_db; create_tables(); init_db()"

# Start the FastAPI server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
python main.py