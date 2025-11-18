# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

# Expose port for API
EXPOSE 8080

# Default command
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8080"]