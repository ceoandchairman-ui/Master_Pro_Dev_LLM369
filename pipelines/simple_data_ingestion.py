"""
Simple Data Ingestion Script for OpenHermes 2.5 Dataset

This script downloads the OpenHermes 2.5 dataset from Hugging Face
and uploads it directly to GCS bucket for LLM training.
"""

import json
import tempfile
import os
from datasets import load_dataset
from google.cloud import storage


def download_and_upload_openhermes():
    """
    Download OpenHermes 2.5 dataset and upload to GCS bucket.
    """
    # Configuration
    BUCKET_NAME = "newllm369-478400-llm-data"
    DATASET_PATH = "raw_data/openhermes-2.5-dataset.jsonl"
    
    print("🚀 Starting OpenHermes 2.5 data ingestion...")
    print(f"   Target: gs://{BUCKET_NAME}/{DATASET_PATH}")
    print()
    
    try:
        # Step 1: Load dataset from Hugging Face
        print("🔄 Loading OpenHermes 2.5 dataset from Hugging Face...")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
        dataset_size = len(dataset)
        print(f"📊 Dataset loaded: {dataset_size:,} examples")
        
        # Step 2: Create temporary file and convert to JSONL
        print("💾 Converting dataset to JSONL format...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            for idx, example in enumerate(dataset):
                json_line = json.dumps(example, ensure_ascii=False)
                temp_file.write(json_line + '\n')
                
                # Progress indicator
                if (idx + 1) % 25000 == 0:
                    progress = ((idx + 1) / dataset_size) * 100
                    print(f"  Progress: {idx + 1:,} examples ({progress:.1f}%)")
            
            temp_file_path = temp_file.name
        
        print(f"✅ Dataset converted to JSONL: {temp_file_path}")
        
        # Step 3: Upload to GCS
        print(f"☁️ Uploading to GCS bucket: {BUCKET_NAME}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(DATASET_PATH)
        
        # Upload with progress
        print(f"  Uploading {os.path.getsize(temp_file_path) / (1024*1024):.1f} MB...")
        blob.upload_from_filename(temp_file_path)
        
        # Step 4: Verify upload
        print("🔍 Verifying upload...")
        blob_info = blob
        blob_info.reload()
        
        print(f"✅ Upload successful!")
        print(f"   Location: gs://{BUCKET_NAME}/{DATASET_PATH}")
        print(f"   Size: {blob_info.size / (1024*1024):.1f} MB")
        print(f"   Examples: {dataset_size:,}")
        
        # Step 5: Cleanup
        print("🧹 Cleaning up temporary files...")
        os.unlink(temp_file_path)
        
        # Step 6: Summary
        print()
        print("🎉 Data ingestion completed successfully!")
        print(f"   Dataset: OpenHermes-2.5")
        print(f"   Examples: {dataset_size:,}")
        print(f"   Format: JSONL")
        print(f"   License: Apache-2.0")
        print(f"   GCS Path: gs://{BUCKET_NAME}/{DATASET_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {str(e)}")
        # Cleanup on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return False


def verify_gcs_dataset():
    """
    Verify the dataset exists in GCS bucket.
    """
    BUCKET_NAME = "newllm369-478400-llm-data"
    DATASET_PATH = "raw_data/openhermes-2.5-dataset.jsonl"
    
    try:
        print("🔍 Verifying dataset in GCS...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(DATASET_PATH)
        
        if blob.exists():
            blob.reload()
            print(f"✅ Dataset verified:")
            print(f"   Path: gs://{BUCKET_NAME}/{DATASET_PATH}")
            print(f"   Size: {blob.size / (1024*1024):.1f} MB")
            print(f"   Created: {blob.time_created}")
            return True
        else:
            print(f"❌ Dataset not found at gs://{BUCKET_NAME}/{DATASET_PATH}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying dataset: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("OpenHermes 2.5 Dataset Ingestion Script")
    print("=" * 60)
    
    # Check if dataset already exists
    if verify_gcs_dataset():
        print("📋 Dataset already exists in GCS. Skipping download.")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("👍 Using existing dataset.")
            exit(0)
    
    # Download and upload dataset
    success = download_and_upload_openhermes()
    
    if success:
        print("🎯 Ready for next step: Model training pipeline!")
    else:
        print("💡 Please check your GCS permissions and try again.")
        exit(1)