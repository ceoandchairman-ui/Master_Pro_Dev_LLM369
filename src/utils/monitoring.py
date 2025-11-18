"""
Monitoring utilities for training and inference
"""

import logging
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    tokens_per_second: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceMetrics:
    """Inference metrics data structure"""
    request_id: str
    input_tokens: int
    output_tokens: int
    latency: float
    temperature: float
    model_name: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TrainingMonitor:
    """Monitor for training metrics and logging"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics_history = []
        self.step_count = 0
        
        # Prometheus metrics
        self.train_loss_metric = Gauge('training_loss', 'Training loss')
        self.eval_loss_metric = Gauge('evaluation_loss', 'Evaluation loss')
        self.learning_rate_metric = Gauge('learning_rate', 'Learning rate')
        self.tokens_per_second_metric = Gauge('tokens_per_second', 'Tokens processed per second')
        self.training_steps = Counter('training_steps_total', 'Total training steps')
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        self.metrics_history.append(metrics)
        self.step_count += 1
        
        # Update Prometheus metrics
        self.train_loss_metric.set(metrics.train_loss)
        if metrics.eval_loss is not None:
            self.eval_loss_metric.set(metrics.eval_loss)
        self.learning_rate_metric.set(metrics.learning_rate)
        if metrics.tokens_per_second is not None:
            self.tokens_per_second_metric.set(metrics.tokens_per_second)
        self.training_steps.inc()
        
        # Structured logging
        if self.step_count % self.log_interval == 0:
            logger.info(
                "Training step completed",
                epoch=metrics.epoch,
                step=metrics.step,
                train_loss=metrics.train_loss,
                eval_loss=metrics.eval_loss,
                learning_rate=metrics.learning_rate,
                tokens_per_second=metrics.tokens_per_second
            )
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get latest training metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 steps
        
        train_losses = [m.train_loss for m in recent_metrics]
        eval_losses = [m.eval_loss for m in recent_metrics if m.eval_loss is not None]
        
        summary = {
            'total_steps': len(self.metrics_history),
            'avg_train_loss': sum(train_losses) / len(train_losses),
            'min_train_loss': min(train_losses),
            'max_train_loss': max(train_losses),
            'latest_step': self.metrics_history[-1].step,
            'latest_epoch': self.metrics_history[-1].epoch
        }
        
        if eval_losses:
            summary.update({
                'avg_eval_loss': sum(eval_losses) / len(eval_losses),
                'min_eval_loss': min(eval_losses),
                'max_eval_loss': max(eval_losses)
            })
        
        return summary


class InferenceMonitor:
    """Monitor for inference metrics and performance"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.request_history = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Prometheus metrics
        self.request_count = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
        self.request_duration = Histogram('inference_request_duration_seconds', 'Request duration', ['model'])
        self.tokens_generated = Counter('inference_tokens_generated_total', 'Total tokens generated', ['model'])
        self.tokens_per_second = Gauge('inference_tokens_per_second', 'Tokens generated per second', ['model'])
        self.active_requests = Gauge('inference_active_requests', 'Number of active requests')
        
        # Request tracking
        self.active_request_count = 0
    
    def log_request(self, request_data: Dict[str, Any], request_id: str):
        """Log incoming request"""
        with self.lock:
            self.active_request_count += 1
            self.active_requests.set(self.active_request_count)
        
        logger.info(
            "Inference request received",
            request_id=request_id,
            message_count=len(request_data.get('messages', [])),
            max_tokens=request_data.get('max_tokens'),
            temperature=request_data.get('temperature'),
            stream=request_data.get('stream', False)
        )
    
    def log_response(self, request_id: str, response_data: Dict[str, Any], duration: float):
        """Log completed response"""
        with self.lock:
            self.active_request_count = max(0, self.active_request_count - 1)
            self.active_requests.set(self.active_request_count)
        
        # Extract metrics
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        model_name = response_data.get('model', 'unknown')
        
        # Create metrics object
        metrics = InferenceMetrics(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=duration,
            temperature=0.7,  # Default from request
            model_name=model_name
        )
        
        self.request_history.append(metrics)
        
        # Update Prometheus metrics
        self.request_count.labels(model=model_name, status='success').inc()
        self.request_duration.labels(model=model_name).observe(duration)
        self.tokens_generated.labels(model=model_name).inc(output_tokens)
        
        # Calculate tokens per second
        if duration > 0:
            tps = output_tokens / duration
            self.tokens_per_second.labels(model=model_name).set(tps)
        
        logger.info(
            "Inference request completed",
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=duration,
            tokens_per_second=output_tokens / duration if duration > 0 else 0
        )
    
    def log_error(self, request_id: str, error: str, model_name: str = 'unknown'):
        """Log request error"""
        with self.lock:
            self.active_request_count = max(0, self.active_request_count - 1)
            self.active_requests.set(self.active_request_count)
        
        self.request_count.labels(model=model_name, status='error').inc()
        
        logger.error(
            "Inference request failed",
            request_id=request_id,
            error=error,
            model=model_name
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics summary"""
        with self.lock:
            if not self.request_history:
                return {
                    'total_requests': 0,
                    'active_requests': self.active_request_count
                }
            
            recent_requests = list(self.request_history)
            
            # Calculate metrics
            total_requests = len(recent_requests)
            avg_latency = sum(r.latency for r in recent_requests) / total_requests
            avg_input_tokens = sum(r.input_tokens for r in recent_requests) / total_requests
            avg_output_tokens = sum(r.output_tokens for r in recent_requests) / total_requests
            
            # Calculate throughput (requests per minute)
            now = time.time()
            recent_requests_1min = [r for r in recent_requests if now - r.timestamp <= 60]
            throughput = len(recent_requests_1min)
            
            # Calculate tokens per second
            total_tokens = sum(r.output_tokens for r in recent_requests_1min)
            tokens_per_second = total_tokens / 60 if recent_requests_1min else 0
            
            return {
                'total_requests': total_requests,
                'active_requests': self.active_request_count,
                'avg_latency': avg_latency,
                'avg_input_tokens': avg_input_tokens,
                'avg_output_tokens': avg_output_tokens,
                'throughput_rpm': throughput,
                'tokens_per_second': tokens_per_second,
                'latest_request_time': recent_requests[-1].timestamp
            }


class ModelMonitor:
    """Monitor for model performance and health"""
    
    def __init__(self):
        self.model_health = Gauge('model_health_status', 'Model health status (1=healthy, 0=unhealthy)')
        self.model_load_time = Gauge('model_load_time_seconds', 'Time taken to load model')
        self.gpu_memory_usage = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        
        self.is_healthy = True
        self.load_time = 0
    
    def set_model_health(self, is_healthy: bool):
        """Set model health status"""
        self.is_healthy = is_healthy
        self.model_health.set(1 if is_healthy else 0)
        
        logger.info("Model health status updated", healthy=is_healthy)
    
    def set_load_time(self, seconds: float):
        """Set model load time"""
        self.load_time = seconds
        self.model_load_time.set(seconds)
        
        logger.info("Model load time recorded", duration=seconds)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            import psutil
            import torch
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                self.gpu_memory_usage.set(gpu_memory)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error("Error updating system metrics", error=str(e))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status"""
        return {
            'healthy': self.is_healthy,
            'load_time': self.load_time,
            'timestamp': time.time()
        }