"""
Inference API for chat completion model
FastAPI-based service for model serving
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import json

from src.models.chat_completion_model import ChatCompletionModel, ChatCompletionConfig
from src.data.preprocessing import ChatDataProcessor
from src.utils.monitoring import InferenceMonitor
from src.utils.gcs_utils import GCSManager

logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
processor = None
monitor = None


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(512, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict] = Field(..., description="List of completion choices")
    usage: Dict = Field(..., description="Token usage information")


class StreamingChatCompletionResponse(BaseModel):
    """Streaming chat completion response model"""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict] = Field(..., description="List of completion choices")


async def load_model():
    """Load model and tokenizer"""
    global model, tokenizer, processor, monitor
    
    logger.info("Loading model and tokenizer...")
    
    try:
        model_path = os.getenv('MODEL_PATH', './models/chat-completion-model')
        
        # Load from GCS if path starts with gs://
        if model_path.startswith('gs://'):
            gcs_manager = GCSManager()
            local_model_path = './models/downloaded_model'
            gcs_manager.download_directory(model_path, local_model_path)
            model_path = local_model_path
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
        
        # Load model configuration and model
        config = ChatCompletionConfig.from_pretrained(model_path)
        model = ChatCompletionModel.from_pretrained(model_path, config=config)
        
        # Set model to evaluation mode
        model.eval()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Initialize processor
        processor = ChatDataProcessor(tokenizer=tokenizer, max_length=config.max_length)
        
        # Initialize monitoring
        monitor = InferenceMonitor()
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await load_model()
    yield
    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Chat Completion API",
    description="API for chat completion using custom LLM",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": int(time.time())
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "chat-completion-model",
        "vocab_size": model.config.vocab_size,
        "max_length": model.config.max_length,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": next(model.parameters()).device.type
    }


def generate_completion_id() -> str:
    """Generate unique completion ID"""
    import uuid
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


async def generate_response(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None
) -> str:
    """Generate response from model"""
    try:
        # Prepare input
        inputs = processor.prepare_inference_input(messages)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


async def generate_streaming_response(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None
) -> AsyncIterator[str]:
    """Generate streaming response from model"""
    try:
        # Prepare input
        inputs = processor.prepare_inference_input(messages)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs.get('attention_mask'),
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "use_cache": True
        }
        
        # Start generation in separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            
            # Check for stop sequences
            should_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        # Send remaining text before stop sequence
                        remaining = generated_text.split(stop_seq)[0]
                        if remaining:
                            yield remaining[len(generated_text) - len(new_text):]
                        should_stop = True
                        break
            
            if should_stop:
                break
            
            yield new_text
        
        thread.join()
        
    except Exception as e:
        logger.error(f"Error generating streaming response: {e}")
        yield f"data: {json.dumps({'error': 'Failed to generate response'})}\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks
):
    """Create chat completion"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate completion ID
        completion_id = generate_completion_id()
        created = int(time.time())
        
        # Log request
        monitor.log_request(request.dict(), completion_id)
        
        if request.stream:
            # Streaming response
            async def stream_generator():
                chunk_id = 0
                try:
                    async for chunk in generate_streaming_response(
                        messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop
                    ):
                        chunk_response = StreamingChatCompletionResponse(
                            id=completion_id,
                            created=created,
                            model="chat-completion-model",
                            choices=[{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        )
                        
                        yield f"data: {chunk_response.json()}\n\n"
                        chunk_id += 1
                    
                    # Send final chunk
                    final_chunk = StreamingChatCompletionResponse(
                        id=completion_id,
                        created=created,
                        model="chat-completion-model",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    )
                    
                    yield f"data: {final_chunk.json()}\n\n"
                    yield "data: [DONE]\n\n"
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_chunk = {
                        "error": {
                            "message": "Internal server error during streaming",
                            "type": "internal_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        else:
            # Non-streaming response
            response_text = await generate_response(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
            
            # Count tokens (approximation)
            prompt_tokens = len(tokenizer.encode(" ".join([msg["content"] for msg in messages])))
            completion_tokens = len(tokenizer.encode(response_text))
            total_tokens = prompt_tokens + completion_tokens
            
            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model="chat-completion-model",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            # Log response
            duration = time.time() - start_time
            background_tasks.add_task(
                monitor.log_response,
                completion_id,
                response.dict(),
                duration
            )
            
            return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def get_metrics():
    """Get inference metrics"""
    if monitor is None:
        return {"error": "Monitoring not initialized"}
    
    return monitor.get_metrics()


if __name__ == "__main__":
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        workers=1,
        log_level="info"
    )