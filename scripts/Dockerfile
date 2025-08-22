FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
redis-server --daemonize yes\n\
python main.py' > start.sh && chmod +x start.sh

# Start the application
CMD ["./start.sh"]
