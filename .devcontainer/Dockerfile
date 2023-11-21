# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt .

# Update the system and install necessary tools
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch with CUDA support
RUN pip3 install -r requirements.txt


