#!/usr/bin/env python3
"""
EDU Bot - Text-based AI Assistant
Powered by NVIDIA NIM with RAG
"""

import os
import json
import hashlib
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import gradio as gr
import requests

# Load environment variables
load_dotenv()

# Configuration
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
LLM_NIM_ENDPOINT = os.getenv('LLM_NIM_ENDPOINT', 'https://integrate.api.nvidia.com/v1')
EMBEDDING_NIM_ENDPOINT = os.getenv('EMBEDDING_NIM_ENDPOINT', 'https://integrate.api.nvidia.com/v1')

print("🤖 Initializing EDU Bot...")

# Cache for responses
response_cache = {}

def get_cache_key(text):
    """Generate cache key for responses"""
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text: str, input_type: str = "query") -> List[float]:
    """Get embeddings from NVIDIA NIM Embedding API"""
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
        print(f"❌ Error getting embedding: {e}")
        return None

def retrieve_relevant_context(query: str, knowledge_base: List[Dict], top_k: int = 3) -> tuple:
    """Retrieve relevant context using embedding similarity"""
    if not knowledge_base:
        return "", []
    
    query_embedding = get_embedding(query, input_type="query")
    if not query_embedding:
        return "", []
    
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
    top_contexts = similarities[:top_k]
    
    # Filter by threshold
    relevant_contexts = [(score, content) for score, content in top_contexts if score > 0.3]
    
    if not relevant_contexts:
        return "", []
    
    # Format context
    context_text = "\n\n".join([content for _, content in relevant_contexts])
    context_info = [{"score": float(score), "content": content} for score, content in relevant_contexts]
    
    return context_text, context_info

def generate_response(query: str, context: str = "", conversation_history: List[Dict] = None) -> str:
    """Generate response using NVIDIA NIM LLM"""
    try:
        # Check cache
        cache_key = get_cache_key(f"{query}{context}")
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Build messages
        system_prompt = """You are EDU Bot, a helpful AI assistant for UMass Dartmouth students. 
Answer questions clearly and concisely based on the provided context. 
If the context doesn't contain relevant information, say so politely and provide general guidance."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 exchanges)
        if conversation_history:
            messages.extend(conversation_history[-20:])
        
        # Add current query with context
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_message = query
        
        messages.append({"role": "user", "content": user_message})
        
        # Call NVIDIA NIM API
        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        
        response = requests.post(
            f"{LLM_NIM_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()["choices"][0]["message"]["content"]
        
        # Cache the response
        response_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I apologize, but I encountered an error: {str(e)}\n\nPlease try again or rephrase your question."

def load_knowledge_base():
    """Load knowledge base from data.json"""
    knowledge_base = []
    
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
        
        print(f"📚 Loading knowledge base...")
        for i, item in enumerate(data):
            conv = item.get("conversation", [])
            if len(conv) >= 2:
                question = conv[0].get("content", "")
                answer = conv[1].get("content", "")
                content = f"Q: {question}\nA: {answer}"
                
                print(f"  Processing {i+1}/{len(data)}...", end='\r')
                embedding = get_embedding(content, input_type="passage")
                if embedding:
                    knowledge_base.append({
                        "content": content,
                        "embedding": embedding
                    })
        
        print(f"\n✅ Loaded {len(knowledge_base)} items into knowledge base")
        return knowledge_base
        
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return []

# Load knowledge base at startup
print("📚 Loading knowledge base...")
initial_knowledge_base = load_knowledge_base()
print("✅ Ready!")

def chat_response(message, history, use_rag):
    """Process chat message and return response"""
    if not message or not message.strip():
        return "Please enter a question."
    
    # Convert Gradio history format to our format
    conversation_history = []
    if history:
        for user_msg, bot_msg in history:
            if user_msg:
                conversation_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                conversation_history.append({"role": "assistant", "content": bot_msg})
    
    # Retrieve context if RAG is enabled
    context = ""
    context_info = []
    if use_rag and initial_knowledge_base:
        context, context_info = retrieve_relevant_context(message, initial_knowledge_base)
        
        if context_info:
            print(f"🔍 Found {len(context_info)} relevant documents:")
            for i, ctx in enumerate(context_info):
                print(f"  {i+1}. Score: {ctx['score']:.3f}")
    
    # Generate response
    response = generate_response(message, context, conversation_history)
    
    # Add context information to response if available
    if context_info:
        context_display = "\n\n---\n**Retrieved Context:**\n"
        for i, ctx in enumerate(context_info):
            snippet = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
            context_display += f"\n{i+1}. (Relevance: {ctx['score']:.2f})\n{snippet}\n"
        response += context_display
    
    return response

# Create Gradio interface
with gr.Blocks(title="EDU Bot - UMass Dartmouth AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 EDU Bot - UMass Dartmouth AI Assistant
    ### Powered by NVIDIA NIM with RAG
    
    Ask questions about courses, registration, campus life, and more!
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="E.g., What computer science courses are available?",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Chat 🗑️")
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            use_rag = gr.Checkbox(
                label="Enable RAG",
                value=True,
                info="Use knowledge base for context"
            )
            
            gr.Markdown(f"""
            ### System Info
            - **Model**: Nemotron Nano 8B
            - **Embeddings**: NV-EmbedQA-E5-V5
            - **Knowledge Base**: {len(initial_knowledge_base)} documents
            - **Status**: ✅ Ready
            """)
            
            gr.Markdown("""
            ### Example Questions
            - What CS courses are available?
            - How do I register for classes?
            - Tell me about campus housing
            - Who is my academic advisor?
            """)
    
    # Event handlers
    def respond(message, chat_history, use_rag):
        bot_message = chat_response(message, chat_history, use_rag)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, use_rag], [msg, chatbot])
    submit.click(respond, [msg, chatbot, use_rag], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting EDU Bot on http://localhost:7860")
    print("="*60 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7860)
