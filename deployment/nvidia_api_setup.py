"""
Alternative Deployment: Use NVIDIA API Catalog (Build.nvidia.com)
This approach uses NVIDIA's hosted inference endpoints instead of self-hosting on SageMaker.
Still meets hackathon requirements as it uses NVIDIA NIM infrastructure.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def setup_nvidia_api_endpoints():
    """
    Configure NVIDIA API endpoints from build.nvidia.com
    These are hosted NVIDIA NIM microservices
    """
    
    print("=" * 60)
    print("🚀 NVIDIA API Catalog Setup")
    print("=" * 60)
    print()
    
    # NVIDIA hosted endpoints (correct API URLs)
    base_url = "https://integrate.api.nvidia.com/v1"
    nemotron_endpoint = base_url
    embedding_endpoint = base_url
    
    print("📋 Using NVIDIA Hosted NIM Endpoints:")
    print()
    print(f"✅ Nemotron LLM: {nemotron_endpoint}/chat/completions")
    print(f"   Model: nvidia/llama-3.1-nemotron-nano-8b-v1")
    print()
    print(f"✅ Embedding: {embedding_endpoint}/embeddings")
    print(f"   Model: nvidia/nv-embedqa-e5-v5")
    print()
    
    # Check for NGC API key
    ngc_api_key = os.getenv("NVIDIA_API_KEY")
    
    if not ngc_api_key or ngc_api_key == "nvapi-your-actual-key-here":
        print("❌ NVIDIA_API_KEY not found in .env file!")
        print()
        print("📝 To fix this:")
        print("   1. Get your NGC API key from: https://build.nvidia.com/")
        print("   2. Click on any model (e.g., Llama 3.1 Nemotron)")
        print("   3. Click 'Get API Key' button")
        print("   4. Copy the key (starts with 'nvapi-')")
        print("   5. Add to .env file: NVIDIA_API_KEY=your-key-here")
        print()
        return False
    
    print(f"✅ NGC API Key found: {ngc_api_key[:12]}...")
    print()
    
    # Update instructions for .env
    print("=" * 60)
    print("📝 UPDATE YOUR .env FILE")
    print("=" * 60)
    print()
    print("Add or update these lines in your .env file:")
    print()
    print(f"NEMOTRON_NIM_ENDPOINT={nemotron_endpoint}")
    print(f"EMBEDDING_NIM_ENDPOINT={embedding_endpoint}")
    print(f"NVIDIA_API_KEY={ngc_api_key}")
    print()
    print("=" * 60)
    print()
    
    # Test endpoints
    print("🧪 Testing NVIDIA API Access...")
    print()
    
    try:
        import requests
        
        # Test Nemotron endpoint
        print("Testing Nemotron LLM...")
        headers = {
            "Authorization": f"Bearer {ngc_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
            "messages": [{"role": "user", "content": "Say 'Hello' in one word"}],
            "max_tokens": 10,
            "temperature": 0.5
        }
        
        response = requests.post(
            f"{nemotron_endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Nemotron LLM: Working!")
            result = response.json()
            if 'choices' in result:
                print(f"   Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"⚠️  Nemotron LLM: Status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        
        print()
        
        # Test Embedding endpoint
        print("Testing Embedding API...")
        payload = {
            "input": "Hello world",
            "model": "nvidia/nv-embedqa-e5-v5",
            "input_type": "query",
            "encoding_format": "float"
        }
        
        response = requests.post(
            f"{embedding_endpoint}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Embedding API: Working!")
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                embedding_dim = len(result['data'][0]['embedding'])
                print(f"   Embedding dimension: {embedding_dim}")
        else:
            print(f"⚠️  Embedding API: Status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        
        print()
        print("=" * 60)
        print("✅ NVIDIA API Setup Complete!")
        print("=" * 60)
        print()
        print("🎉 You're ready to build the voice assistant!")
        print()
        print("Next steps:")
        print("   1. Update .env with the endpoints shown above")
        print("   2. Create voice assistant application")
        print("   3. Test locally")
        print()
        
        return True
        
    except ImportError:
        print("❌ 'requests' library not installed")
        print("   Run: pip3 install requests")
        return False
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")
        return False


if __name__ == "__main__":
    print()
    setup_nvidia_api_endpoints()
    print()
