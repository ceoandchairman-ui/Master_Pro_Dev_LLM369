# 10-Minute Quick Demo - PowerShell Version

Write-Host "🚀 Starting 10-minute LLM MLOps demo setup..." -ForegroundColor Green

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker not found" -ForegroundColor Red
    exit 1
}

# Step 1: Environment setup (1 minute)
Write-Host "🔧 Setting up environment..." -ForegroundColor Yellow
python -m venv venv
& ".\venv\Scripts\Activate.ps1"
pip install fastapi uvicorn torch transformers --quiet

# Step 2: Create demo model (2 minutes)
Write-Host "🤖 Creating demo model..." -ForegroundColor Yellow
$demoModelScript = @"
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import os

# Create minimal demo model
config = GPT2Config(vocab_size=1000, n_positions=256, n_embd=128, n_layer=2, n_head=2)
model = GPT2LMHeadModel(config)

# Create demo directory
os.makedirs('demo_model', exist_ok=True)
model.save_pretrained('demo_model')

# Create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained('demo_model')

print('✅ Demo model created')
"@

$demoModelScript | python

# Step 3: Create minimal API (2 minutes)
Write-Host "🌐 Creating demo API..." -ForegroundColor Yellow
$apiContent = @"
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import uuid

app = FastAPI(title="LLM Demo API")

# Load demo model
try:
    model = GPT2LMHeadModel.from_pretrained("demo_model")
    tokenizer = GPT2Tokenizer.from_pretrained("demo_model")
    model.eval()
    model_loaded = True
except Exception as e:
    print(f"Model loading error: {e}")
    model_loaded = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 50

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded, "model": "demo"}

@app.post("/v1/chat/completions")
def chat_completion(request: ChatRequest):
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    # Simple demo response
    user_message = request.messages[-1].content if request.messages else "Hello"
    
    try:
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
        
    except Exception as e:
        response = f"Demo response to: {user_message}"
    
    return {
        "id": f"demo-{str(uuid.uuid4())[:8]}",
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
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🎬 Starting demo API on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
"@

$apiContent | Out-File -FilePath "demo_api.py" -Encoding UTF8

# Step 4: Start demo API
Write-Host "🎬 Starting demo API..." -ForegroundColor Yellow
Start-Process python -ArgumentList "demo_api.py" -WindowStyle Hidden

# Wait for API to start
Start-Sleep -Seconds 5

# Step 5: Test the API
Write-Host "🧪 Testing the demo API..." -ForegroundColor Yellow

$testData = @{
    messages = @(@{role = "user"; content = "Hello!"})
    max_tokens = 20
} | ConvertTo-Json -Depth 3

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/v1/chat/completions" -Method Post -Body $testData -ContentType "application/json"
    Write-Host "✅ API Response:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "⚠️ API test failed, but server might still be starting..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Demo API is running at http://localhost:8080" -ForegroundColor Green
Write-Host "📊 Health check: http://localhost:8080/health" -ForegroundColor Cyan
Write-Host "📖 API docs: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 10-minute demo complete!" -ForegroundColor Green
Write-Host "💡 To test manually:" -ForegroundColor Yellow
Write-Host "   Invoke-RestMethod -Uri 'http://localhost:8080/health'" -ForegroundColor White
Write-Host "💡 To stop: Find and kill the python process" -ForegroundColor Yellow