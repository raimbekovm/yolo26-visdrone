# YOLO26-VisDrone Docker Image
# Multi-stage build for efficient image size

# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder with dependencies
FROM base as builder

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Stage 3: Final runtime image
FROM base as runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories for data and outputs
RUN mkdir -p datasets runs results assets

# Set default command
CMD ["python", "-m", "src.training.trainer", "--help"]

# Expose port for Gradio demo
EXPOSE 7860

# Labels
LABEL maintainer="Murat Raimbekov <murat.raimbekov2004@gmail.com>"
LABEL version="0.1.0"
LABEL description="YOLO26 vs YOLO11 benchmark on VisDrone dataset"

# =====================================================
# Build and run instructions:
#
# Build image:
#   docker build -t yolo26-visdrone:latest -f docker/Dockerfile .
#
# Run training (with GPU):
#   docker run --gpus all -it --rm \
#     -v $(pwd)/datasets:/app/datasets \
#     -v $(pwd)/runs:/app/runs \
#     yolo26-visdrone:latest \
#     python -m src.training.trainer --model yolo26n.pt --epochs 100
#
# Run demo:
#   docker run --gpus all -it --rm \
#     -p 7860:7860 \
#     -v $(pwd)/runs:/app/runs \
#     yolo26-visdrone:latest \
#     python app.py
#
# Interactive shell:
#   docker run --gpus all -it --rm \
#     -v $(pwd):/app \
#     yolo26-visdrone:latest \
#     /bin/bash
# =====================================================
