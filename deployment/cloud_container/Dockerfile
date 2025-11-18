# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ingestion script
COPY ingestion.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=newllm369-478400

# Run the ingestion script
CMD ["python", "ingestion.py"]