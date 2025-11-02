#!/usr/bin/env python3
"""
Simple Voice Assistant - Lightweight Version
Uses NVIDIA NIM APIs with RAG, without heavy UI dependencies
"""

import os
import json
import requests
from dotenv import load_dotenv
import soundfile as sf
import numpy as np

# Load environment variables
load_dotenv()

class NVIDIANIMClient:
    """Simple client for NVIDIA NIM APIs"""
    
    def __init__(self):
        self.api_key = os.getenv('NVIDIA_API_KEY')
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages, temperature=0.5, max_tokens=1024):
        """Generate chat completion"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_embedding(self, text, input_type="query"):
        """Generate text embedding"""
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": text,
            "input_type": input_type,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            return None

class SimpleRAG:
    """Simple RAG implementation"""
    
    def __init__(self, client, knowledge_file="data.json"):
        self.client = client
        self.documents = []
        self.embeddings = []
        self.load_knowledge(knowledge_file)
    
    def load_knowledge(self, filename):
        """Load knowledge base"""
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            return
        
        with open(filename, 'r') as f:
            data = json.load(f)
            
            # Convert conversations to documents
            if isinstance(data, list):
                for item in data:
                    if 'conversation' in item:
                        conv = item['conversation']
                        # Combine user question and assistant answer
                        question = conv[0].get('content', '') if len(conv) > 0 else ''
                        answer = conv[1].get('content', '') if len(conv) > 1 else ''
                        self.documents.append({
                            'title': question[:50] + '...' if len(question) > 50 else question,
                            'content': f"Q: {question}\nA: {answer}"
                        })
            else:
                self.documents = data.get('documents', [])
        
        print(f"Loaded {len(self.documents)} documents")
        print("Generating embeddings... (this may take a minute)")
        
        for i, doc in enumerate(self.documents):
            text = doc.get('content', '')
            embedding = self.client.get_embedding(text, input_type="passage")
            if embedding:
                self.embeddings.append(embedding)
            print(f"  {i+1}/{len(self.documents)} embeddings generated", end='\r')
        
        print(f"\n✓ Knowledge base ready with {len(self.embeddings)} embeddings")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents"""
        if not self.embeddings:
            return []
        
        query_embedding = self.client.get_embedding(query, input_type="query")
        if not query_embedding:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            if score > 0.3:  # Threshold
                results.append({
                    'content': self.documents[idx].get('content', ''),
                    'title': self.documents[idx].get('title', 'Unknown'),
                    'score': float(score)
                })
        
        return results

def main():
    """Main function"""
    print("\n" + "="*60)
    print("  EDU Bot - Simple Voice Assistant")
    print("  Powered by NVIDIA NIM")
    print("="*60 + "\n")
    
    # Initialize
    print("Initializing NVIDIA NIM client...")
    client = NVIDIANIMClient()
    
    print("Loading RAG knowledge base...")
    rag = SimpleRAG(client)
    
    print("\n" + "="*60)
    print("  Ready! Type your questions or 'quit' to exit")
    print("="*60 + "\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("\n You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        print("\n🔍 Searching knowledge base...")
        
        # Retrieve relevant documents
        relevant_docs = rag.retrieve(user_input, top_k=3)
        
        if relevant_docs:
            print(f"   Found {len(relevant_docs)} relevant documents:")
            for i, doc in enumerate(relevant_docs):
                print(f"   {i+1}. {doc['title']} (relevance: {doc['score']:.2f})")
            
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
        else:
            print("   No relevant documents found")
            context = ""
        
        # Prepare messages
        system_prompt = """You are EDU Bot, a helpful AI assistant for UMass Dartmouth students.
Answer questions clearly and concisely based on the provided context.
If the context doesn't contain relevant information, say so politely."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 5 exchanges)
        for msg in conversation_history[-10:]:
            messages.append(msg)
        
        # Add current query with context
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {user_input}"
        else:
            user_message = user_input
        
        messages.append({"role": "user", "content": user_message})
        
        print("\n💭 Generating response...")
        
        # Generate response
        response = client.chat_completion(messages)
        
        print(f"\n🤖 EDU Bot: {response}\n")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
