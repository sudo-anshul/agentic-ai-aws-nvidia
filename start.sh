#!/bin/bash

# Quick Start Script for EDU Bot

echo "=========================================="
echo "🎓 Starting EDU Bot Voice Assistant"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create .env file with:"
    echo "  NVIDIA_API_KEY=your-api-key-here"
    echo "  NEMOTRON_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1"
    echo "  EMBEDDING_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1"
    echo ""
    exit 1
fi

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "❌ Error: src directory not found!"
    exit 1
fi

# Navigate to src and run
cd src
echo "🚀 Launching EDU Bot..."
echo ""
python3 voice_assistant_nvidia_nim.py
