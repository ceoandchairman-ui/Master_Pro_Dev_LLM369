"""
Logging utilities for MLOps project
"""

import logging
import logging.config
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
from google.cloud import logging as cloud_logging


def setup_logging(
    level: str = "INFO",
    format: str = "standard",
    enable_cloud_logging: bool = None
) -> None:
    """
    Setup logging configuration for the MLOps project
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Log format ('standard', 'json')
        enable_cloud_logging: Enable Google Cloud Logging integration
    """
    
    # Determine if we should enable cloud logging
    if enable_cloud_logging is None:
        enable_cloud_logging = bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if format == "standard" else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    if format == "json":
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JsonFormatter,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": sys.stdout,
                },
            },
            "root": {
                "level": level.upper(),
                "handlers": ["console"],
            },
        }
    else:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": sys.stdout,
                },
            },
            "root": {
                "level": level.upper(),
                "handlers": ["console"],
            },
        }
    
    logging.config.dictConfig(logging_config)
    
    # Setup Google Cloud Logging if enabled
    if enable_cloud_logging:
        try:
            client = cloud_logging.Client()
            client.setup_logging()
            logging.info("Google Cloud Logging enabled")
        except Exception as e:
            logging.warning(f"Failed to setup Google Cloud Logging: {e}")


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class MLOpsLogger:
    """Specialized logger for MLOps operations"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start event"""
        self.logger.info(
            "Training started",
            event_type="training_start",
            model_name=config.get("model", {}).get("name"),
            batch_size=config.get("training", {}).get("batch_size"),
            learning_rate=config.get("training", {}).get("learning_rate"),
            num_epochs=config.get("training", {}).get("num_epochs")
        )
    
    def log_training_step(self, step: int, epoch: int, loss: float, metrics: Optional[Dict] = None):
        """Log training step"""
        log_data = {
            "event_type": "training_step",
            "step": step,
            "epoch": epoch,
            "loss": loss
        }
        
        if metrics:
            log_data.update(metrics)
        
        self.logger.info("Training step completed", **log_data)
    
    def log_training_end(self, final_metrics: Dict[str, Any]):
        """Log training completion"""
        self.logger.info(
            "Training completed",
            event_type="training_end",
            **final_metrics
        )
    
    def log_model_save(self, model_path: str, model_size: Optional[int] = None):
        """Log model save event"""
        self.logger.info(
            "Model saved",
            event_type="model_save",
            model_path=model_path,
            model_size_mb=model_size / (1024 * 1024) if model_size else None
        )
    
    def log_inference_request(self, request_id: str, input_length: int, config: Dict[str, Any]):
        """Log inference request"""
        self.logger.info(
            "Inference request received",
            event_type="inference_request",
            request_id=request_id,
            input_length=input_length,
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            stream=config.get("stream", False)
        )
    
    def log_inference_response(self, request_id: str, output_length: int, latency: float):
        """Log inference response"""
        self.logger.info(
            "Inference response generated",
            event_type="inference_response",
            request_id=request_id,
            output_length=output_length,
            latency_seconds=latency,
            tokens_per_second=output_length / latency if latency > 0 else 0
        )
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Log error event"""
        log_data = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message
        }
        
        if context:
            log_data.update(context)
        
        self.logger.error("Error occurred", **log_data)
    
    def log_data_processing(self, dataset_name: str, num_samples: int, processing_time: float):
        """Log data processing event"""
        self.logger.info(
            "Data processing completed",
            event_type="data_processing",
            dataset_name=dataset_name,
            num_samples=num_samples,
            processing_time_seconds=processing_time,
            samples_per_second=num_samples / processing_time if processing_time > 0 else 0
        )
    
    def log_deployment(self, deployment_type: str, endpoint_url: str, status: str):
        """Log deployment event"""
        self.logger.info(
            "Model deployment",
            event_type="deployment",
            deployment_type=deployment_type,
            endpoint_url=endpoint_url,
            status=status
        )


class PerformanceLogger:
    """Logger for performance metrics and profiling"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(f"{name}.performance")
    
    def log_execution_time(self, operation: str, duration: float, context: Optional[Dict] = None):
        """Log operation execution time"""
        log_data = {
            "operation": operation,
            "duration_seconds": duration,
            "event_type": "performance"
        }
        
        if context:
            log_data.update(context)
        
        self.logger.info("Operation completed", **log_data)
    
    def log_memory_usage(self, operation: str, memory_mb: float, peak_memory_mb: Optional[float] = None):
        """Log memory usage"""
        log_data = {
            "operation": operation,
            "memory_mb": memory_mb,
            "event_type": "memory_usage"
        }
        
        if peak_memory_mb is not None:
            log_data["peak_memory_mb"] = peak_memory_mb
        
        self.logger.info("Memory usage recorded", **log_data)
    
    def log_gpu_usage(self, operation: str, gpu_utilization: float, gpu_memory_mb: float):
        """Log GPU usage"""
        self.logger.info(
            "GPU usage recorded",
            event_type="gpu_usage",
            operation=operation,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_mb=gpu_memory_mb
        )


def get_logger(name: str) -> MLOpsLogger:
    """Get MLOps logger instance"""
    return MLOpsLogger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger instance"""
    return PerformanceLogger(name)