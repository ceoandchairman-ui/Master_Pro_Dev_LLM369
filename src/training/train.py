"""
Training script for chat completion model
Supports distributed training on Vertex AI
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import wandb
import mlflow
from google.cloud import storage
import yaml

from src.models.chat_completion_model import ChatCompletionModel, ChatCompletionConfig
from src.data.preprocessing import ChatDataProcessor, ChatDataset, DataCollator, load_tokenizer
from src.utils.logging_utils import setup_logging
from src.utils.gcs_utils import GCSManager
from src.utils.monitoring import TrainingMonitor

logger = logging.getLogger(__name__)


class ChatCompletionTrainer:
    """Custom trainer for chat completion model"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.gcs_manager = GCSManager(self.config['project_id'])
        self.setup_logging()
        self.setup_model_and_tokenizer()
        self.setup_monitoring()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config['project_id'] = os.getenv('GCP_PROJECT_ID', config['project_id'])
        config['bucket_name'] = os.getenv('GCS_BUCKET_NAME', config['bucket_name'])
        
        return config
    
    def setup_logging(self):
        """Setup logging configuration"""
        setup_logging(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format='json' if os.getenv('LOG_FORMAT') == 'json' else 'standard'
        )
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        tokenizer_name = self.config.get('tokenizer_name', 'gpt2')
        self.tokenizer = load_tokenizer(tokenizer_name)
        
        # Update vocab size in config if needed
        model_config = ChatCompletionConfig(**self.config['model'])
        if model_config.vocab_size != len(self.tokenizer):
            logger.info(f"Updating vocab size from {model_config.vocab_size} to {len(self.tokenizer)}")
            model_config.vocab_size = len(self.tokenizer)
        
        # Initialize model
        self.model = ChatCompletionModel(model_config)
        
        # Resize token embeddings if necessary
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_monitoring(self):
        """Setup monitoring and experiment tracking"""
        self.monitor = TrainingMonitor()
        
        # Initialize Weights & Biases
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb'].get('run_name'),
                config=self.config
            )
        
        # Initialize MLflow
        if self.config.get('mlflow', {}).get('enabled', False):
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.start_run(run_name=self.config['mlflow'].get('run_name'))
            mlflow.log_params(self.config)
    
    def load_datasets(self):
        """Load training and validation datasets"""
        logger.info("Loading datasets...")
        
        processor = ChatDataProcessor(
            tokenizer=self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        # Download data from GCS if needed
        train_data_path = self.config['data']['train_file']
        val_data_path = self.config['data']['validation_file']
        
        if train_data_path.startswith('gs://'):
            local_train_path = 'data/train.jsonl'
            self.gcs_manager.download_file(train_data_path, local_train_path)
            train_data_path = local_train_path
        
        if val_data_path.startswith('gs://'):
            local_val_path = 'data/validation.jsonl'
            self.gcs_manager.download_file(val_data_path, local_val_path)
            val_data_path = local_val_path
        
        # Create datasets
        self.train_dataset = ChatDataset(train_data_path, processor, split='train')
        self.val_dataset = ChatDataset(val_data_path, processor, split='validation')
        
        logger.info(f"Loaded {len(self.train_dataset)} training examples")
        logger.info(f"Loaded {len(self.val_dataset)} validation examples")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir='./models/checkpoints',
            overwrite_output_dir=True,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config['warmup_steps'],
            max_steps=training_config.get('max_steps', -1),
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            evaluation_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_total_limit=3,
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', False),
            gradient_checkpointing=training_config.get('gradient_checkpointing', False),
            deepspeed=training_config.get('deepspeed_config'),
            report_to=['wandb'] if self.config.get('wandb', {}).get('enabled', False) else [],
            run_name=self.config.get('run_name', 'chat-completion-training'),
            logging_dir='./logs',
        )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load datasets
        self.load_datasets()
        
        # Setup data collator
        data_collator = DataCollator(self.tokenizer)
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[self.monitor] if hasattr(self.monitor, 'on_log') else []
        )
        
        # Start training
        try:
            train_result = trainer.train()
            
            # Log final metrics
            self.log_metrics(train_result.metrics)
            
            # Save final model
            self.save_model(trainer)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            self.cleanup()
    
    def save_model(self, trainer: Trainer):
        """Save model to GCS"""
        logger.info("Saving model...")
        
        # Save locally first
        local_model_path = './models/final_model'
        trainer.save_model(local_model_path)
        
        # Upload to GCS
        gcs_model_path = f"gs://{self.config['bucket_name']}/models/{self.config['model']['name']}/{self.config['model']['version']}"
        self.gcs_manager.upload_directory(local_model_path, gcs_model_path)
        
        # Save tokenizer
        tokenizer_path = './models/tokenizer'
        self.tokenizer.save_pretrained(tokenizer_path)
        self.gcs_manager.upload_directory(tokenizer_path, f"{gcs_model_path}/tokenizer")
        
        logger.info(f"Model saved to {gcs_model_path}")
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to monitoring systems"""
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.log(metrics)
        
        if self.config.get('mlflow', {}).get('enabled', False):
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.finish()
        
        if self.config.get('mlflow', {}).get('enabled', False):
            mlflow.end_run()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train chat completion model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed training if applicable
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    # Initialize trainer
    trainer = ChatCompletionTrainer(args.config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()