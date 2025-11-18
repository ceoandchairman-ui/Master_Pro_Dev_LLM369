"""
Download OpenHermes 2.5 dataset and upload to GCS bucket.

This script downloads the OpenHermes 2.5 dataset from Hugging Face
and uploads it directly to our GCS bucket for cloud-native processing.
"""

import os
import json
import tempfile
from typing import Dict, Any
from datasets import load_dataset
from google.cloud import storage


def download_openhermes_dataset() -> str:
    """
    Download OpenHermes 2.5 dataset from Hugging Face.
    
    Returns:
        str: Path to the downloaded dataset file
    """
    print("🔄 Loading OpenHermes 2.5 dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    
    print(f"📊 Dataset loaded: {len(dataset)} examples")
    
    # Create temporary file for the dataset
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    print("💾 Converting dataset to JSONL format...")
    
    # Convert to JSONL format for efficient processing
    for idx, example in enumerate(dataset):
        json_line = json.dumps(example, ensure_ascii=False)
        temp_file.write(json_line + '\n')
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} examples...")
    
    temp_file.close()
    print(f"✅ Dataset saved to temporary file: {temp_file.name}")
    
    return temp_file.name


def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    """
    Upload file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to local file
        bucket_name: Name of the GCS bucket
        destination_blob_name: Name for the file in GCS
    """
    print(f"☁️ Uploading to GCS bucket: {bucket_name}")
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Upload file
    print(f"  Uploading {local_file_path} to {destination_blob_name}...")
    blob.upload_from_filename(local_file_path)
    
    print(f"✅ File uploaded successfully to gs://{bucket_name}/{destination_blob_name}")


def main():
    """Main function to orchestrate dataset download and upload."""
    
    # Configuration
    BUCKET_NAME = "newllm369-478400-llm-data"
    DATASET_FILENAME = "openhermes-2.5-dataset.jsonl"
    
    print("🚀 Starting OpenHermes 2.5 dataset download and upload process...")
    print(f"   Target bucket: gs://{BUCKET_NAME}")
    print(f"   Target filename: {DATASET_FILENAME}")
    print()
    
    try:
        # Step 1: Download dataset
        local_file_path = download_openhermes_dataset()
        
        # Step 2: Upload to GCS
        upload_to_gcs(local_file_path, BUCKET_NAME, f"raw_data/{DATASET_FILENAME}")
        
        # Step 3: Cleanup
        print("🧹 Cleaning up temporary files...")
        os.unlink(local_file_path)
        
        print()
        print("🎉 Dataset download and upload completed successfully!")
        print(f"   Dataset location: gs://{BUCKET_NAME}/raw_data/{DATASET_FILENAME}")
        print(f"   Dataset size: ~250K conversational examples")
        print(f"   License: Apache-2.0")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()