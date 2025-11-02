#!/bin/bash

# EDU Bot Installation Script
# This script installs all dependencies for the voice assistant

echo "=========================================="
echo "🎓 EDU Bot Installation"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

echo ""
echo "🔧 Installing system dependencies..."
echo "Note: You may need to install these manually:"
echo "  - espeak-ng (for phonemizer/TTS)"
echo "  - ffmpeg (for audio processing)"
echo ""

# Detect OS and provide instructions
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected. Install with Homebrew:"
    echo "  brew install espeak-ng ffmpeg"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected. Install with:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y espeak-ng ffmpeg libportaudio2"
fi

echo ""
read -p "Press Enter when system dependencies are installed, or Ctrl+C to exit..."

echo ""
echo "📦 Installing Python packages..."
echo "This may take 15-20 minutes..."
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.4..."
pip3 install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure .env file is configured with NVIDIA_API_KEY"
echo "  2. Run: cd src && python3 voice_assistant_nvidia_nim.py"
echo "  3. Open browser to: http://localhost:7860"
echo ""
