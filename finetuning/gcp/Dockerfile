# Dockerfile for Unsloth + LoRA fine-tuning on Vertex AI
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install system packages (including C/C++ compiler)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc

# Set working directory
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]