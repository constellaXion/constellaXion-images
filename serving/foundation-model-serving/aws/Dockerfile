FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Additional for testing
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Additional for testing
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install Python 3.11 and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Optionally, set Python 3.11 as the default python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1

# Additonal for testing
# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN ca-certificates \
    openjdk-8-jdk-headless \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py

# Additional for testing
# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                retrying

# Additional for testing
# Copy entrypoint script to the image
# COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
# RUN chmod +x /usr/local/bin/dockerd-entrypoint.py


WORKDIR /app
COPY . /app

# Additional for testing
RUN mkdir -p /home/model-server/

# Additional for testing
# Copy the default custom service file to handle incoming data and inference requests
COPY test_server.py /home/model-server/test_server.py


# # Copy entrypoint script
# COPY entrypoint.sh /app/entrypoint.sh
# RUN chmod +x /app/entrypoint.sh

RUN pip install --no-cache-dir -r requirements.txt

# Accept build-time Hugging Face token and set it as an environment variable
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Disable TorchInductor
ENV TORCHINDUCTOR_DISABLE=1
ENV TORCH_COMPILE=0

EXPOSE 8080
# Use the entrypoint script to remount /dev/shm before starting the server
# This larger shared memory area should prevent NCCL errors due to insufficient shared memory when running distributed serving (e.g., with pipeline parallelism)
# ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "test_server:app", "--host", "0.0.0.0", "--port", "8080"]
