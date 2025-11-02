import os
import time
import tempfile
import threading
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import gradio as gr
import torch
from transformers import pipeline
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️  Kokoro TTS not available - TTS will be disabled")
import requests
import json

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# NVIDIA NIM endpoints (configure these based on your deployment)
NEMOTRON_NIM_ENDPOINT = os.getenv("NEMOTRON_NIM_ENDPOINT", "http://localhost:8000/v1")
EMBEDDING_NIM_ENDPOINT = os.getenv("EMBEDDING_NIM_ENDPOINT", "http://localhost:8001/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

print("Initializing Whisper pipeline...")
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device
)
print("Whisper pipeline initialized!")

if KOKORO_AVAILABLE:
    os.environ['ESPEAK_DATA_PATH'] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
    kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    print("Kokoro TTS initialized!")
else:
    kokoro_pipeline = None
    print("Running without TTS - audio output disabled")

response_cache = {}
tts_cache = {}
cache_lock = threading.Lock()

@dataclass
class AppState:
    conversation_history: list = field(default_factory=list)
    knowledge_base: list = field(default_factory=list)
    is_first_interaction: bool = True

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text: str, input_type: str = "query") -> List[float]:
    """Get embeddings from NVIDIA NIM Embedding microservice"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "model": "nvidia/nv-embedqa-e5-v5",
            "input_type": input_type,
            "encoding_format": "float"
        }
        response = requests.post(
            f"{EMBEDDING_NIM_ENDPOINT}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def retrieve_relevant_context(query: str, knowledge_base: List[Dict], top_k: int = 3) -> str:
    """Retrieve relevant context using embedding similarity"""
    if not knowledge_base:
        return ""
    
    query_embedding = get_embedding(query)
    if not query_embedding:
        return ""
    
    # Calculate cosine similarity
    similarities = []
    for item in knowledge_base:
        if "embedding" in item:
            similarity = np.dot(query_embedding, item["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
            )
            similarities.append((similarity, item["content"]))
    
    # Sort by similarity and get top_k
    similarities.sort(reverse=True, key=lambda x: x[0])
    relevant_contexts = [content for _, content in similarities[:top_k]]
    
    return "\n\n".join(relevant_contexts)

def call_nemotron_nim(messages: List[Dict], context: str = "") -> str:
    """Call llama-3.1-nemotron-nano-8B-v1 via NVIDIA NIM"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add context to system message if available
        system_message = {
            "role": "system",
            "content": f"""You are EDUBOT, an intelligent voice assistant for UMass Dartmouth students. 
You provide helpful, accurate, and friendly responses about university topics.

{f'Relevant Context: {context}' if context else ''}

Keep responses conversational and concise (2-3 sentences). Do not use line breaks, bullets, or colons."""
        }
        
        full_messages = [system_message] + messages
        
        payload = {
            "model": "meta/llama-3.1-nemotron-70b-instruct",  # or your deployed model
            "messages": full_messages,
            "temperature": 0.6,
            "max_tokens": 400,
            "top_p": 0.9
        }
        
        response = requests.post(
            f"{NEMOTRON_NIM_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Nemotron NIM: {e}")
        return f"I'm having trouble connecting right now. Please try again."

def load_knowledge_base(json_path: str = "data.json") -> List[Dict]:
    """Load and embed knowledge base"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        knowledge_base = []
        for item in data:
            conv = item.get("conversation", [])
            if len(conv) >= 2:
                question = conv[0].get("content", "")
                answer = conv[1].get("content", "")
                content = f"Q: {question}\nA: {answer}"
                
                embedding = get_embedding(content, input_type="passage")
                if embedding:
                    knowledge_base.append({
                        "content": content,
                        "embedding": embedding
                    })
        
        print(f"Loaded {len(knowledge_base)} items into knowledge base")
        return knowledge_base
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return []

def transcribe_audio_optimized(audio_data, sample_rate):
    if audio_data is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate, format="wav")
            temp_path = temp_file.name
        result = whisper_pipe(temp_path)
        text = result["text"].strip()
        os.unlink(temp_path)
        return text if text else None
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def generate_tts_audio_optimized(text):
    if not KOKORO_AVAILABLE or kokoro_pipeline is None:
        print("TTS not available - skipping audio generation")
        return None
    
    try:
        cached_audio = tts_cache.get(get_cache_key(text))
        if cached_audio and os.path.exists(cached_audio):
            return cached_audio
        
        generator = kokoro_pipeline(text, voice='af_heart')
        audio_path = f'/tmp/response_{hashlib.md5(text.encode()).hexdigest()}.wav'
        for _, _, audio in generator:
            sf.write(audio_path, audio, 24000)
            break
        
        with cache_lock:
            tts_cache[get_cache_key(text)] = audio_path
        return audio_path
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

def response_optimized(state: AppState, audio: tuple):
    if not audio:
        return state, None
    
    audio_data, sample_rate = audio[1], audio[0]
    transcription = transcribe_audio_optimized(audio_data, sample_rate)
    
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."
        
        state.conversation_history.append({"role": "user", "content": transcription})
        
        if state.is_first_interaction:
            assistant_message = "Welcome to EDU Bot, your UMass Dartmouth assistant—how can I help you today?"
            state.is_first_interaction = False
        else:
            # Retrieve relevant context using embeddings
            context = retrieve_relevant_context(transcription, state.knowledge_base)
            
            # Generate response using Nemotron NIM with context
            assistant_message = call_nemotron_nim(state.conversation_history, context)
        
        state.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        print(f"User: {transcription}")
        print(f"Assistant: {assistant_message}")
        
        audio_path = generate_tts_audio_optimized(assistant_message)
        return state, audio_path
    
    return state, None

def process_audio(audio: tuple, state: AppState):
    return audio, state

def start_recording_user(state: AppState):
    return None

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded");
    var record = document.querySelector('.record-button');
    record.textContent = "🎤 Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) stop.click();
      }
    });
    myvad.start();
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () => {
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "🎤 Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

# Initialize knowledge base at startup
initial_knowledge_base = load_knowledge_base()

with gr.Blocks(js=js) as demo:
    gr.Markdown("## 🎓 EDU Bot - UMass Dartmouth (Powered by NVIDIA NIM + AWS)")
    gr.Markdown("*Agentic AI Assistant using llama-3.1-nemotron-nano-8B-v1 and Retrieval Embeddings*")

    output_audio = gr.Audio(
        label="Voice Response",
        autoplay=True,
        show_label=True,
        container=True
    )

    input_audio = gr.Audio(
        label="🎤 Voice Input",
        sources=["microphone"],
        type="numpy",
        streaming=False,
        show_label=True,
        container=True
    )

    state = gr.State(value=AppState(knowledge_base=initial_knowledge_base))

    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )

    respond = input_audio.stop_recording(
        response_optimized,
        [state, input_audio],
        [state, output_audio]
    )

    restart = respond.then(
        start_recording_user,
        [state],
        [input_audio]
    ).then(
        lambda state: state,
        state,
        state,
        js=js_reset
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
