#!/bin/bash

# Check if NVIDIA drivers are available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected! Using GPU acceleration."
    # Start with GPU configuration
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up "$@"
else
    echo "💻 No NVIDIA GPU detected. Running on CPU."
    # Start with base configuration (CPU only)
    docker-compose up "$@"
fi