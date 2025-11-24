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
from kfp import dsl
from kfp.dsl import component, pipeline, Output, Input, Dataset, Model
from google.cloud import aiplatform


# Configuration
PROJECT_ID = "newllm369-478400"
REGION = "us-central1"
BUCKET_NAME = "newllm369-478400-llm-data"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines"


@component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-storage==2.14.0",
        "transformers==4.36.0",
        "tokenizers==0.15.0",
        "numpy==1.24.3",
        "tqdm==4.66.1"
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
    import numpy as np
    from transformers import AutoTokenizer
    from google.cloud import storage
    import logging
    import random
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting OpenHermes preprocessing...")
    logger.info(f"   Input: {input_data_path}")
    logger.info(f"   Output: {output_data_path}")
    logger.info(f"   Tokenizer: {tokenizer_name}")
    logger.info(f"   Max Length: {max_length}")
    logger.info(f"   Train Split: {train_split_ratio}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize GCS client
    client = storage.Client()
    bucket_name = input_data_path.split("/")[2]
    input_blob_name = "/".join(input_data_path.split("/")[3:])
    
    # Read and process data
    logger.info("Reading raw data from GCS...")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(input_blob_name)
    
    # Single-pass processing with reservoir sampling for train/val split
    logger.info("Processing and writing data (single pass)...")
    
    # Use random sampling to decide train/val/test split on-the-fly
    random.seed(42)  # For reproducibility
    
    processed_count = 0
    skipped_examples = 0
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Statistics tracking (incremental computation)
    sum_lengths = 0
    sum_sq_lengths = 0
    min_length = float('inf')
    max_length_actual = 0
    
    # Prepare output files
    output_bucket_name = output_data_path.split("/")[2]
    output_base_path = "/".join(output_data_path.split("/")[3:])
    output_bucket = client.bucket(output_bucket_name)
    
    train_blob = output_bucket.blob(f"{output_base_path}/train_data.jsonl")
    val_blob = output_bucket.blob(f"{output_base_path}/val_data.jsonl")
    test_blob = output_bucket.blob(f"{output_base_path}/test_data.jsonl")
    
    with blob.open("r", encoding='utf-8') as f_in, \
         train_blob.open("w", encoding='utf-8') as f_train, \
         val_blob.open("w", encoding='utf-8') as f_val, \
         test_blob.open("w", encoding='utf-8') as f_test:
        
        total_lines = 0
        for line_num, line in enumerate(f_in):
            total_lines += 1
            
            if line_num % 10000 == 0:
                logger.info(f"   Processing line {line_num:,}...")
            
            try:
                data = json.loads(line.strip())
                
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
                token_length = len(tokens['input_ids'])
                if token_length < 10:
                    skipped_examples += 1
                    continue
                
                # Create training example (without 'length' field to save memory)
                example = {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                    'labels': tokens['input_ids'].copy(),
                    'text': chat_text[:500] + "..." if len(chat_text) > 500 else chat_text,
                    'source': data.get('source', 'unknown'),
                    'category': data.get('category', 'unknown')
                }
                
                # Randomly assign to train/val/test (no memory accumulation)
                # Using 70/15/15 split for proper ML evaluation
                random_split = random.random()
                if random_split < 0.70:  # 70% train
                    f_train.write(json.dumps(example, ensure_ascii=False) + '\n')
                    train_count += 1
                elif random_split < 0.85:  # 15% val (0.70 to 0.85)
                    f_val.write(json.dumps(example, ensure_ascii=False) + '\n')
                    val_count += 1
                else:  # 15% test (0.85 to 1.0)
                    f_test.write(json.dumps(example, ensure_ascii=False) + '\n')
                    test_count += 1
                
                # Update statistics incrementally
                sum_lengths += token_length
                sum_sq_lengths += token_length * token_length
                min_length = min(min_length, token_length)
                max_length_actual = max(max_length_actual, token_length)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {str(e)[:100]}")
                skipped_examples += 1
                continue
    
    logger.info("Processing completed!")
    logger.info(f"   Total lines: {total_lines:,}")
    logger.info(f"   Total processed: {processed_count:,}")
    logger.info(f"   Train examples: {train_count:,}")
    logger.info(f"   Validation examples: {val_count:,}")
    logger.info(f"   Test examples: {test_count:,}")
    logger.info(f"   Skipped: {skipped_examples:,}")
    
    # Validate that we have data in all splits
    if processed_count == 0:
        raise ValueError(
            f"CRITICAL: Zero examples processed!\n"
            f"Total lines: {total_lines:,}\n"
            f"Skipped: {skipped_examples:,}"
        )
    
    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError(
            f"CRITICAL: One or more splits are empty!\n"
            f"Train: {train_count:,}\n"
            f"Val: {val_count:,}\n"
            f"Test: {test_count:,}"
        )
    
    # Generate statistics
    logger.info("Generating statistics...")
    
    # Calculate mean and std from incremental sums
    avg_length = sum_lengths / processed_count if processed_count > 0 else 0
    variance = (sum_sq_lengths / processed_count - avg_length ** 2) if processed_count > 0 else 0
    std_length = variance ** 0.5 if variance > 0 else 0
    
    stats = {
        'total_raw_examples': total_lines,
        'processed_examples': processed_count,
        'skipped_examples': skipped_examples,
        'train_examples': train_count,
        'val_examples': val_count,
        'test_examples': test_count,
        'target_train_ratio': 0.70,
        'target_val_ratio': 0.15,
        'target_test_ratio': 0.15,
        'actual_train_ratio': train_count / processed_count if processed_count > 0 else 0,
        'actual_val_ratio': val_count / processed_count if processed_count > 0 else 0,
        'actual_test_ratio': test_count / processed_count if processed_count > 0 else 0,
        'tokenizer_name': tokenizer_name,
        'max_length': max_length,
        'avg_length': float(avg_length),
        'std_length': float(std_length),
        'min_length': int(min_length) if min_length != float('inf') else 0,
        'max_length_actual': int(max_length_actual),
        'vocab_size': tokenizer.vocab_size,
        'processing_success_rate': (processed_count / total_lines * 100) if total_lines > 0 else 0
    }
    
    # Save statistics
    stats_blob = output_bucket.blob(f"{output_base_path}/preprocessing_stats.json")
    with stats_blob.open("w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Preprocessing pipeline completed successfully!")
    
    # Set outputs
    processed_dataset.path = output_data_path
    preprocessing_stats.path = f"{output_data_path}/preprocessing_stats.json"


@pipeline(
    name="openhermes-preprocessing-pipeline",
    description="Preprocess OpenHermes 2.5 dataset for LLM training",
    pipeline_root=PIPELINE_ROOT,
)
def openhermes_preprocessing_pipeline(
    input_data_path: str,
    output_data_path: str,
    tokenizer_name: str = "microsoft/DialoGPT-medium",
    max_length: int = 1024,
    train_split_ratio: float = 0.70,  # Updated to 70% for 3-way split
):
    """
    OpenHermes preprocessing pipeline
    """
    
    # Preprocessing step with memory specification
    preprocessing_task = preprocess_openhermes_data(
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        train_split_ratio=train_split_ratio,
    )
    
    # Configure machine resources
    preprocessing_task.set_memory_limit("32G")
    preprocessing_task.set_cpu_limit("8")


def compile_and_run_pipeline():
    """
    Compile and run the preprocessing pipeline
    """
    from kfp import compiler
    
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=openhermes_preprocessing_pipeline,
        package_path="openhermes_preprocessing_pipeline.json"
    )
    
    print("Pipeline compiled successfully!")
    print("Pipeline file: openhermes_preprocessing_pipeline.json")
    
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
    
    print("Starting pipeline execution...")
    job.run(sync=True)
    print("Pipeline execution completed!")


if __name__ == "__main__":
    compile_and_run_pipeline()