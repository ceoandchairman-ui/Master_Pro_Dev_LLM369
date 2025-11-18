#!/bin/bash

# 10-Minute Quick Start Script
# This script sets up a minimal working demo of the LLM MLOps project

set -e

echo "🚀 Starting 10-minute LLM MLOps demo setup..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v python >/dev/null 2>&1 || { echo "❌ Python not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found"; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo "❌ gcloud not found"; exit 1; }

# Step 1: Environment setup (1 minute)
echo "🔧 Setting up environment..."
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn torch transformers --quiet

# Step 2: Minimal configuration (30 seconds)
echo "⚙️ Creating minimal config..."
cat > minimal_config.py << 'EOF'
import os
os.environ["MODEL_PATH"] = "./demo_model"
os.environ["LOG_LEVEL"] = "INFO"
EOF

# Step 3: Create demo model (1 minute)
echo "🤖 Creating demo model..."
python << 'EOF'
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import os

# Create minimal demo model
config = GPT2Config(vocab_size=1000, n_positions=256, n_embd=128, n_layer=2, n_head=2)
model = GPT2LMHeadModel(config)

# Create demo directory
os.makedirs("demo_model", exist_ok=True)
model.save_pretrained("demo_model")

# Create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("demo_model")

print("✅ Demo model created")
EOF

# Step 4: Create minimal API (2 minutes)
echo "🌐 Creating demo API..."
cat > demo_api.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import uuid

app = FastAPI(title="LLM Demo API")

# Load demo model
model = GPT2LMHeadModel.from_pretrained("demo_model")
tokenizer = GPT2Tokenizer.from_pretrained("demo_model")
model.eval()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 50

@app.get("/health")
def health():
    return {"status": "healthy", "model": "demo"}

@app.post("/v1/chat/completions")
def chat_completion(request: ChatRequest):
    # Simple demo response
    user_message = request.messages[-1].content if request.messages else "Hello"
    
    # Tokenize input
    inputs = tokenizer.encode(user_message, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=min(request.max_tokens, 20),
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    return {
        "id": f"demo-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "demo-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response or "Hello! This is a demo response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs[0]),
            "completion_tokens": 10,
            "total_tokens": len(inputs[0]) + 10
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

# Step 5: Start demo API (30 seconds)
echo "🎬 Starting demo API..."
python demo_api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Step 6: Test the API (1 minute)
echo "🧪 Testing the demo API..."
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 20
  }' | python -m json.tool

echo ""
echo "✅ Demo API is running at http://localhost:8080"
echo "📊 Health check: http://localhost:8080/health"
echo "📖 API docs: http://localhost:8080/docs"
echo ""
echo "🎉 10-minute demo complete!"
echo "💡 To stop: kill $API_PID"

# Keep script running
wait $API_PID
EOF