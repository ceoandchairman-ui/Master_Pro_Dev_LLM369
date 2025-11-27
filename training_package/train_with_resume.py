"""
PyTorch LLM Training Script for Vertex AI Managed Training with Preemptible GPU Resume
Includes checkpointing and resume logic for GCS
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from google.cloud import storage

# Import custom model
from src.models.custom_llm_model import CustomLLMModel, CustomLLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizedDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 2048):
        self.max_length = max_length
        self.data = []
        logger.info(f"Loading data from {data_path}")
        local_path = self._download_from_gcs(data_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} examples")
    def _download_from_gcs(self, gcs_path: str) -> str:
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
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']
        current_len = len(input_ids)
        if current_len < self.max_length:
            pad_len = self.max_length - current_len
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_latest_checkpoint(bucket_name, checkpoint_prefix="checkpoints"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=checkpoint_prefix))
    checkpoints = [blob.name for blob in blobs if "checkpoint-" in blob.name]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]

def download_checkpoint(bucket_name, checkpoint_name, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(checkpoint_name)
    local_path = os.path.join(local_dir, os.path.basename(checkpoint_name))
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(local_dir: str, gcs_path: str):
    logger.info(f"Would upload {local_dir} to {gcs_path}")
    # Implement actual upload logic as needed

def train(model, train_dataloader, val_dataloader, optimizer, config, start_step=0):
    training_config = config['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = training_config['max_steps']
    logging_steps = training_config['logging_steps']
    save_steps = training_config['save_steps']
    eval_steps = training_config['eval_steps']
    writer = SummaryWriter(f"/tmp/tensorboard_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    output_dir = "/tmp/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    global_step = start_step
    total_loss = 0
    best_val_loss = float('inf')
    model.train()
    while global_step < max_steps:
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            global_step += 1
            if global_step % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                logger.info(f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f}")
                writer.add_scalar('train/loss', avg_loss, global_step)
                total_loss = 0
            if global_step % eval_steps == 0:
                val_loss = evaluate(model, val_dataloader, device)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                writer.add_scalar('val/loss', val_loss, global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            if global_step % save_steps == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                torch.save(model.state_dict(), f"{checkpoint_dir}/model.pt")
                upload_to_gcs(checkpoint_dir, f"gs://{config['bucket_name']}/checkpoints/checkpoint-{global_step}")
    writer.close()
    logger.info("Training completed!")
    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
    upload_to_gcs(output_dir, f"gs://{config['bucket_name']}/checkpoints/final_model")

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            total_loss += outputs['loss'].item()
            num_batches += 1
            if num_batches >= 100:
                break
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')

def main():
    config_path = os.environ.get('CONFIG_PATH', 'configs/custom_llm_config.yaml')
    config = load_config(config_path)
    torch.manual_seed(42)
    train_dataset = TokenizedDataset(
        config['data']['train_file'],
        max_length=config['model']['max_length']
    )
    val_dataset = TokenizedDataset(
        config['data']['validation_file'],
        max_length=config['model']['max_length']
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
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
    model = CustomLLMModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    # Resume logic
    start_step = 0
    latest_ckpt = get_latest_checkpoint(config['bucket_name'])
    if latest_ckpt:
        logger.info(f"Resuming from checkpoint: {latest_ckpt}")
        local_ckpt = download_checkpoint(config['bucket_name'], latest_ckpt, "/tmp/checkpoints")
        model.load_state_dict(torch.load(local_ckpt))
        # Parse step from checkpoint name
        try:
            start_step = int(latest_ckpt.split('-')[-1])
        except Exception:
            start_step = 0
    else:
        logger.info("No checkpoint found, starting fresh training.")
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        config=config,
        start_step=start_step
    )

if __name__ == "__main__":
    main()
