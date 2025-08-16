# Base image with CUDA 12.4 (matching PyTorch 2.6.0 + cu124)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python 3.11 and basic build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    curl \
    git \
    libgl1 \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1

# Environment tweaks to avoid TorchInductor and compilation surprises
ENV TORCHINDUCTOR_DISABLE=1
ENV TORCH_COMPILE=0
ENV CC=gcc

# Set working directory and copy app
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip

# Then install everything else (vLLM, unsloth, bitsandbytes will link to this torch/CUDA)
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8080

# Start the app
CMD ["python", "server.py"]