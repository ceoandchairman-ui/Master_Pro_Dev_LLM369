# Training Container Dockerfile for Custom LLM
# Optimized for 4×L4 GPUs with DeepSpeed ZeRO-2
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional training dependencies
RUN pip install --no-cache-dir \
    deepspeed==0.12.3 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    datasets==2.15.0 \
    tensorboard==2.15.1 \
    google-cloud-storage==2.10.0 \
    google-cloud-aiplatform==1.38.0 \
    pyyaml==6.0.1 \
    ninja==1.11.1 \
    flash-attn==2.4.2 --no-build-isolation

# Copy project files
COPY . /app/

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Environment variables for optimization
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/tmp/huggingface
ENV TORCH_DISTRIBUTED_DEBUG=INFO
ENV NCCL_DEBUG=INFO
ENV CUDA_LAUNCH_BLOCKING=0

# Create necessary directories
RUN mkdir -p /tmp/checkpoints /tmp/logs /tmp/data

# Set entrypoint for DeepSpeed launcher
ENTRYPOINT ["deepspeed", "--num_gpus=4"]

# Default command
CMD ["src/training/train.py"]
