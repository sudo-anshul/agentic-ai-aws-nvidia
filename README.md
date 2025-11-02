# EDU Bot - AI Assistant for UMass Dartmouth

[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76B900?logo=nvidia)](https://build.nvidia.com/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.49-orange)](https://gradio.app/)

> **AI-powered chatbot for UMass Dartmouth students using NVIDIA NIM microservices with Retrieval-Augmented Generation (RAG)**

---

**Agentic AI Unleashed: AWS & NVIDIA Hackathon 2025**

**Team**: [Your Team Name]  
**Demo Video**: [YouTube Link - To be uploaded]  
**GitHub**: https://github.com/[your-username]/edu-bot

---

## Quick Start

### Prerequisites
- Python 3.13+
- NVIDIA API Key ([Get it here](https://build.nvidia.com/))
- 4GB free disk space

### Installation (5 minutes)

```bash
# 1. Clone/navigate to project
cd "agentic-ai-aws-nvidia-main 2"

# 2. Create .env file with your API key
echo "NVIDIA_API_KEY=your-api-key-here" > .env
echo "LLM_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1" >> .env
echo "EMBEDDING_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1" >> .env

# 3. Install dependencies
pip3 install --no-cache-dir python-dotenv requests gradio numpy transformers torch scikit-learn

# 4. Run the assistant
python3 text_assistant_nvidia_nim.py
```

### Open in Browser
Navigate to: **http://localhost:7860**

---

## Project Overview

EDU Bot is an intelligent chatbot that helps UMass Dartmouth students access university information instantly. Powered by NVIDIA NIM microservices, it uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from a knowledge base of 50+ university Q&A pairs.

### Key Features

- 🤖 **NVIDIA NIM Integration**: Powered by Llama 3.1 Nemotron Nano 8B (8B parameter LLM)
- 🧠 **RAG Pipeline**: Semantic search with NV-EmbedQA-E5-V5 embeddings (1024-dim vectors)
- � **Interactive Chat**: Clean web UI built with Gradio
- 📚 **Knowledge Base**: 50+ Q&A pairs about courses, admissions, campus life
- ⚡ **Fast Responses**: Cached embeddings, typically 3-5 second response time
- � **Context-Aware**: Maintains conversation history for multi-turn dialogues

---

##  Architecture

```
┌─────────────────────────────────────────┐
│         User Input (Text)               │
│        Gradio Web Interface             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      RAG Knowledge Base (50 docs)       │
│  • Load data.json                       │
│  • Generate embeddings (passage mode)   │
│  • Cosine similarity search             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        NVIDIA NIM API Catalog           │
│  ┌────────────────────────────────────┐ │
│  │  nvidia/nv-embedqa-e5-v5           │ │
│  │  (1024-dim embeddings)             │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  nvidia/llama-3.1-nemotron-nano-8b │ │
│  │  (LLM for chat completions)        │ │
│  └────────────────────────────────────┘ │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│          Generated Response             │
│     (with retrieved context)            │
└─────────────────────────────────────────┘
```

### Technology Stack

**NVIDIA NIM Microservices:**
- **LLM**: `nvidia/llama-3.1-nemotron-nano-8b-v1` (8B parameter reasoning model)
- **Embeddings**: `nvidia/nv-embedqa-e5-v5` (1024-dim vectors for RAG)

**Infrastructure:**
- **API**: NVIDIA API Catalog (hosted NIM endpoints)
- **Interface**: Gradio 5.49 web UI
- **RAG**: Vector similarity search with cosine distance

### Agentic Capabilities

✅ **Autonomous Information Retrieval**: Automatically searches knowledge base  
✅ **Semantic Understanding**: Uses embeddings to find relevant context  
✅ **Adaptive Responses**: Adjusts answers based on retrieved information  
✅ **Conversation Memory**: Maintains context across multiple exchanges  
✅ **Real-time Learning**: Updates embeddings when knowledge base changes

---

## 📁 Project Structure

```
agentic-ai-aws-nvidia-main 2/
├── text_assistant_nvidia_nim.py  # Main application (text chat)
├── simple_voice_assistant.py     # CLI version (lightweight)
├── voice_assistant_nvidia_nim.py # Voice version (deprecated)
├── data.json                     # Knowledge base (50 Q&A pairs)
├── requirements.txt              # Python dependencies
├── .env                          # API keys (create this)
├── .env.example                  # Template
├── .gitignore                    # Git exclusions
├── install.sh                    # Installation script
├── start.sh                      # Quick start script
├── README.md                     # This file
└── deployment/
    └── nvidia_api_setup.py       # API testing script
```

---

##  Usage Examples

### Starting the Application

```bash
# Method 1: Direct run
python3 text_assistant_nvidia_nim.py

# Method 2: Using start script
./start.sh

# Method 3: CLI version (no UI)
python3 simple_voice_assistant.py
```

### Example Questions

Try asking:
- "What computer science courses are available?"
- "How do I register for classes?"
- "Tell me about campus housing options"
- "Who is my academic advisor?"
- "What are the graduation requirements?"
- "When do classes start?"

### RAG Toggle

- **Enabled (default)**: Uses knowledge base for context-aware answers with retrieved documents shown
- **Disabled**: Direct LLM responses without context retrieval

### Features in the Interface

- 💬 **Chat History**: See all previous messages
- 📄 **Retrieved Docs**: View relevant documents with similarity scores
- 🔄 **RAG Toggle**: Turn retrieval on/off
- 📊 **System Info**: Knowledge base stats, API endpoints
- ❓ **Example Questions**: Quick-start prompts

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# Required
NVIDIA_API_KEY=nvapi-YOUR-API-KEY-HERE

# Optional (defaults shown)
LLM_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
```

### Get Your NVIDIA API Key

1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Sign in with your NVIDIA account (or create one)
3. Navigate to API Catalog
4. Select "Generate API Key"
5. Copy to `.env` file

---

## Testing

### Test NVIDIA APIs

```bash
cd deployment
python3 nvidia_api_setup.py
```

Expected output:
```
✅ LLM API working (200 OK)
✅ Embedding API working (1024-dim vectors)
```

### Test Knowledge Base

```bash
python3 simple_voice_assistant.py
# Type questions from Examples section
# Type "quit" to exit
```

### Verify Installation

```bash
# Check Python version
python3 --version  # Should be 3.13+

# Check installed packages
pip3 list | grep -E "gradio|torch|transformers"

# Check disk space
df -h .
```

---

## 🛠️ Troubleshooting

### Port 7860 Already in Use

```bash
# Find and kill process
lsof -ti:7860 | xargs kill -9

# Restart application
python3 text_assistant_nvidia_nim.py
```

### No Module Named 'X'

```bash
# Reinstall dependencies
pip3 install --no-cache-dir -r requirements.txt
```

### Embedding API Errors (400 Bad Request)

Check your `.env` file:
- API key is correct (starts with `nvapi-`)
- No extra spaces or quotes around values
- Endpoints are correct

### Out of Disk Space

```bash
# Clear pip cache
pip3 cache purge

# Check available space
df -h .

# Remove unused packages
pip3 uninstall <package-name>
```

### Application Loading Slowly

- First-time embedding generation takes ~30 seconds (50 documents)
- Embeddings are cached for subsequent runs
- Check terminal for "Processing X/50..." progress

---

## Dependencies

### Core
- Python 3.13+
- python-dotenv 1.1.0
- requests 2.32+
- numpy 2.3+

### ML & AI
- torch 2.9.0 (CPU version)
- transformers 4.57+
- scikit-learn 1.7+

### Web Interface
- gradio 5.49+

### Optional (for voice features)
- openai-whisper
- librosa
- soundfile

---

## Acknowledgments

- **NVIDIA** for NIM microservices and API access
- **AWS** for cloud infrastructure support and hackathon organization
- **UMass Dartmouth** for the use case and knowledge base data
- **Gradio** for the intuitive web interface framework
- **Hugging Face** for Transformers library

---
Create a `.env` file:
