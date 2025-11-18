"""
Vertex AI Pipeline for OpenHermes 2.5 Data Preprocessing

This pipeline processes the raw OpenHermes dataset and prepares it for LLM training:
1. Load raw conversations from GCS
2. Convert to training format with proper chat templates
3. Tokenize conversations using specified tokenizer
4. Create train/validation splits
5. Save processed data back to GCS
"""

from typing import NamedTuple
import json
from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline, Output, Input, Dataset, Model
from google.cloud import aiplatform


# Configuration
PROJECT_ID = "newllm369-478400"
REGION = "us-central1"
BUCKET_NAME = "newllm369-478400-llm-data"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines"


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-storage>=2.10.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "tqdm>=4.65.0"
    ]
)
def preprocess_openhermes_data(
    input_data_path: str,
    output_data_path: str,
    tokenizer_name: str,
    max_length: int,
    train_split_ratio: float,
    processed_dataset: Output[Dataset],
    preprocessing_stats: Output[Dataset]
):
    """
    Preprocess OpenHermes conversations for LLM training
    """
    import json
    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer
    from google.cloud import storage
    import logging
    from tqdm import tqdm
    import random
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting OpenHermes preprocessing...")
    logger.info(f"   Input: {input_data_path}")
    logger.info(f"   Output: {output_data_path}")
    logger.info(f"   Tokenizer: {tokenizer_name}")
    logger.info(f"   Max Length: {max_length}")
    logger.info(f"   Train Split: {train_split_ratio}")
    
    # Initialize tokenizer
    logger.info("🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize GCS client
    client = storage.Client()
    bucket_name = input_data_path.split("/")[2]
    input_blob_name = "/".join(input_data_path.split("/")[3:])
    
    # Read and process data
    logger.info("📖 Reading raw data from GCS...")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(input_blob_name)
    
    processed_examples = []
    total_examples = 0
    skipped_examples = 0
    
    with blob.open("r", encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing conversations")):
            try:
                data = json.loads(line.strip())
                total_examples += 1
                
                # Extract conversation
                if 'conversations' not in data or not data['conversations']:
                    skipped_examples += 1
                    continue
                
                conversations = data['conversations']
                
                # Convert to chat format
                chat_text = ""
                for msg in conversations:
                    role = msg.get('from', 'unknown')
                    content = msg.get('value', '').strip()
                    
                    if not content:
                        continue
                    
                    if role == 'human':
                        chat_text += f"<|user|>{content}<|end|>\n"
                    elif role in ['gpt', 'assistant']:
                        chat_text += f"<|assistant|>{content}<|end|>\n"
                
                if not chat_text:
                    skipped_examples += 1
                    continue
                
                # Tokenize
                tokens = tokenizer(
                    chat_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None
                )
                
                # Skip if too short
                if len(tokens['input_ids']) < 10:
                    skipped_examples += 1
                    continue
                
                # Create training example
                example = {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                    'labels': tokens['input_ids'].copy(),  # For causal LM
                    'text': chat_text[:500] + "..." if len(chat_text) > 500 else chat_text,  # For debugging
                    'source': data.get('source', 'unknown'),
                    'category': data.get('category', 'unknown'),
                    'length': len(tokens['input_ids'])
                }
                
                processed_examples.append(example)
                
                # Progress update
                if len(processed_examples) % 1000 == 0:
                    logger.info(f"   Processed {len(processed_examples):,} examples...")
                
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                skipped_examples += 1
                continue
    
    logger.info(f"✅ Processing completed!")
    logger.info(f"   Total examples: {total_examples:,}")
    logger.info(f"   Processed: {len(processed_examples):,}")
    logger.info(f"   Skipped: {skipped_examples:,}")
    
    # Create train/validation splits
    logger.info("🔄 Creating train/validation splits...")
    random.shuffle(processed_examples)
    
    split_idx = int(len(processed_examples) * train_split_ratio)
    train_examples = processed_examples[:split_idx]
    val_examples = processed_examples[split_idx:]
    
    logger.info(f"   Train examples: {len(train_examples):,}")
    logger.info(f"   Validation examples: {len(val_examples):,}")
    
    # Save processed data
    logger.info("💾 Saving processed data to GCS...")
    
    output_bucket_name = output_data_path.split("/")[2]
    output_base_path = "/".join(output_data_path.split("/")[3:])
    
    output_bucket = client.bucket(output_bucket_name)
    
    # Save train data
    train_blob = output_bucket.blob(f"{output_base_path}/train_data.jsonl")
    with train_blob.open("w", encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save validation data
    val_blob = output_bucket.blob(f"{output_base_path}/val_data.jsonl")
    with val_blob.open("w", encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Generate statistics
    stats = {
        'total_raw_examples': total_examples,
        'processed_examples': len(processed_examples),
        'skipped_examples': skipped_examples,
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'train_split_ratio': train_split_ratio,
        'tokenizer_name': tokenizer_name,
        'max_length': max_length,
        'avg_length': np.mean([ex['length'] for ex in processed_examples]),
        'median_length': np.median([ex['length'] for ex in processed_examples]),
        'vocab_size': tokenizer.vocab_size,
        'processing_success_rate': len(processed_examples) / total_examples * 100
    }
    
    # Save statistics
    stats_blob = output_bucket.blob(f"{output_base_path}/preprocessing_stats.json")
    with stats_blob.open("w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("🎉 Preprocessing pipeline completed successfully!")
    
    # Set outputs
    processed_dataset.path = output_data_path
    preprocessing_stats.path = f"{output_data_path}/preprocessing_stats.json"
    
    return stats


@pipeline(
    name="openhermes-preprocessing-pipeline",
    description="Preprocess OpenHermes 2.5 dataset for LLM training",
    pipeline_root=PIPELINE_ROOT,
)
def openhermes_preprocessing_pipeline(
    input_data_path: str = f"gs://{BUCKET_NAME}/raw_data/openhermes-2.5-dataset.jsonl",
    output_data_path: str = f"gs://{BUCKET_NAME}/processed_data/openhermes-2.5",
    tokenizer_name: str = "microsoft/DialoGPT-medium",  # Good for chat
    max_length: int = 1024,
    train_split_ratio: float = 0.9,
):
    """
    OpenHermes preprocessing pipeline
    """
    
    # Preprocessing step
    preprocessing_task = preprocess_openhermes_data(
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        train_split_ratio=train_split_ratio,
    )
    
    return preprocessing_task.outputs


def compile_and_run_pipeline():
    """
    Compile and run the preprocessing pipeline
    """
    from kfp.v2 import compiler
    
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=openhermes_preprocessing_pipeline,
        package_path="openhermes_preprocessing_pipeline.json"
    )
    
    print("✅ Pipeline compiled successfully!")
    print("📁 Pipeline file: openhermes_preprocessing_pipeline.json")
    
    # Create and run pipeline job
    job = aiplatform.PipelineJob(
        display_name="openhermes-preprocessing-run",
        template_path="openhermes_preprocessing_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "input_data_path": f"gs://{BUCKET_NAME}/raw_data/openhermes-2.5-dataset.jsonl",
            "output_data_path": f"gs://{BUCKET_NAME}/processed_data/openhermes-2.5",
            "tokenizer_name": "microsoft/DialoGPT-medium",
            "max_length": 1024,
            "train_split_ratio": 0.9,
        }
    )
    
    print("🚀 Starting pipeline execution...")
    job.run(sync=True)
    print("🎉 Pipeline execution completed!")


if __name__ == "__main__":
    compile_and_run_pipeline()