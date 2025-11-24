"""
Distributed Training Script for Custom LLM using DeepSpeed ZeRO-2
Optimized for 4×L4 GPUs with 2048 context length
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from transformers import get_scheduler
from google.cloud import storage

# Import custom model
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.custom_llm_model import CustomLLMModel, CustomLLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized JSONL files"""
    
    def __init__(self, data_path: str, max_length: int = 2048):
        self.max_length = max_length
        self.data = []
        
        # Download from GCS
        logger.info(f"Loading data from {data_path}")
        local_path = self._download_from_gcs(data_path)
        
        # Load JSONL
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def _download_from_gcs(self, gcs_path: str) -> str:
        """Download file from GCS to local temp"""
        parts = gcs_path.replace("gs://", "").split("/")
        bucket_name = parts[0]
        blob_name = "/".join(parts[1:])
        
        local_path = f"/tmp/{Path(blob_name).name}"
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        
        return local_path
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get pre-tokenized data
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']
        
        # Pad to max_length if needed
        current_len = len(input_ids)
        if current_len < self.max_length:
            pad_len = self.max_length - current_len
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
        else:
            # Truncate if longer
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Setup distributed training environment"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    return rank, local_rank, world_size


def train(
    model,
    train_dataloader,
    val_dataloader,
    model_engine,
    config: Dict[str, Any],
    rank: int = 0
):
    """Main training loop"""
    
    training_config = config['training']
    device = torch.device(f"cuda:{rank}")
    
    # Training parameters
    max_steps = training_config['max_steps']
    logging_steps = training_config['logging_steps']
    save_steps = training_config['save_steps']
    eval_steps = training_config['eval_steps']
    
    # Setup TensorBoard
    writer = None
    if rank == 0:
        log_dir = f"/tmp/tensorboard_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logs: {log_dir}")
    
    # Setup output directory
    output_dir = "/tmp/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info(f"  Device: {device}")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  World size: {os.environ.get('WORLD_SIZE', 1)}")
    logger.info(f"  Max steps: {max_steps}")
    logger.info(f"  Training examples: {len(train_dataloader.dataset)}")
    logger.info("=" * 80)
    
    model_engine.train()
    global_step = 0
    total_loss = 0
    best_val_loss = float('inf')
    
    while global_step < max_steps:
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Backward pass with DeepSpeed
            model_engine.backward(loss)
            model_engine.step()
            
            # Update metrics
            total_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0 and rank == 0:
                avg_loss = total_loss / logging_steps
                lr = model_engine.optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )
                
                if writer:
                    writer.add_scalar('train/loss', avg_loss, global_step)
                    writer.add_scalar('train/learning_rate', lr, global_step)
                
                total_loss = 0
            
            # Evaluation
            if global_step % eval_steps == 0 and rank == 0:
                val_loss = evaluate(model_engine, val_dataloader, device)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                
                if writer:
                    writer.add_scalar('val/loss', val_loss, global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                
                model_engine.train()
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                model_engine.save_checkpoint(checkpoint_dir)
                
                # Upload to GCS (rank 0 only)
                if rank == 0:
                    try:
                        gcs_path = f"gs://{config['bucket_name']}/checkpoints/checkpoint-{global_step}"
                        logger.info(f"Uploading checkpoint to {gcs_path}")
                        upload_to_gcs(checkpoint_dir, gcs_path)
                    except Exception as e:
                        logger.error(f"Failed to upload checkpoint: {e}")
    
    logger.info("Training completed!")
    
    # Save final model
    if rank == 0:
        final_dir = f"{output_dir}/final"
        logger.info(f"Saving final model to {final_dir}")
        model_engine.save_checkpoint(final_dir)
        
        if writer:
            writer.close()


def evaluate(model_engine, dataloader, device):
    """Evaluate model on validation set"""
    model_engine.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model_engine(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Limit validation batches
            if num_batches >= 100:
                break
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def upload_to_gcs(local_dir: str, gcs_path: str):
    """Upload directory to GCS"""
    # Simplified - just log for now
    logger.info(f"Would upload {local_dir} to {gcs_path}")


def main():
    """Main training function"""
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Using device: {device}")
    
    # Load config
    config_path = os.environ.get('CONFIG_PATH', 'configs/custom_llm_config.yaml')
    config = load_config(config_path)
    
    # Set seed
    torch.manual_seed(42)
    
    # Create datasets
    train_dataset = TokenizedDataset(
        config['data']['train_file'],
        max_length=config['model']['max_length']
    )
    
    val_dataset = TokenizedDataset(
        config['data']['validation_file'],
        max_length=config['model']['max_length']
    )
    
    # Create dataloaders with distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=True
    )
    
    # Create model config
    model_config = CustomLLMConfig(
        vocab_size=config['model']['vocab_size'],
        max_length=config['model']['max_length'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_key_value_heads=config['model']['num_key_value_heads'],
        use_flash_attention=config['model']['use_flash_attention'],
        use_rope=config['model']['use_rope'],
        use_gqa=config['model']['use_gqa'],
        use_swiglu=config['model']['use_swiglu'],
        use_rms_norm=config['model']['use_rms_norm'],
        rope_theta=config['model']['rope_theta'],
        attention_dropout=config['model']['attention_dropout'],
        hidden_dropout=config['model']['hidden_dropout'],
        layer_norm_eps=config['model']['layer_norm_eps']
    )
    
    # Create model
    logger.info("Initializing model...")
    logger.info(f"  Vocab size: {model_config.vocab_size}")
    logger.info(f"  Max length: {model_config.max_length}")
    logger.info(f"  Layers: {model_config.num_layers}")
    logger.info(f"  Hidden size: {model_config.hidden_size}")
    
    model = CustomLLMModel(model_config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Initialize DeepSpeed
    ds_config_path = config['training']['deepspeed_config']
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )
    
    logger.info("DeepSpeed initialized successfully")
    logger.info(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")
    logger.info(f"  BF16: {model_engine.bfloat16_enabled()}")
    
    # Start training
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model_engine=model_engine,
        config=config,
        rank=rank
    )
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
