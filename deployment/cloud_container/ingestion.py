"""
Cloud Streaming Data Ingestion for OpenHermes 2.5 Dataset

This script streams the OpenHermes 2.5 dataset from Hugging Face
and uploads it directly to GCS bucket without local storage.
"""

import json
import os
from datasets import load_dataset
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stream_to_gcs():
    """
    Stream OpenHermes 2.5 dataset directly to GCS bucket.
    """
    # Configuration
    BUCKET_NAME = "newllm369-478400-llm-data"
    DATASET_PATH = "raw_data/openhermes-2.5-dataset.jsonl"
    BATCH_SIZE = 1000  # Process in batches for efficiency
    
    logger.info("🚀 Starting OpenHermes 2.5 streaming ingestion...")
    logger.info(f"   Target: gs://{BUCKET_NAME}/{DATASET_PATH}")
    
    try:
        # Step 1: Initialize GCS client
        logger.info("☁️ Initializing GCS client...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(DATASET_PATH)
        
        # Step 2: Load streaming dataset
        logger.info("🔄 Loading streaming dataset from Hugging Face...")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        
        # Step 3: Stream directly to GCS
        logger.info("📤 Starting streaming upload to GCS...")
        
        with blob.open("w") as gcs_file:
            batch_count = 0
            total_examples = 0
            
            for idx, example in enumerate(dataset):
                # Write JSON line to GCS
                json_line = json.dumps(example, ensure_ascii=False)
                gcs_file.write(json_line + '\n')
                
                total_examples += 1
                
                # Progress logging every batch
                if (idx + 1) % BATCH_SIZE == 0:
                    batch_count += 1
                    logger.info(f"  📊 Processed {total_examples:,} examples (Batch {batch_count})")
                    
                # Memory management - flush periodically
                if (idx + 1) % (BATCH_SIZE * 5) == 0:
                    gcs_file.flush()
        
        # Step 4: Verify upload
        logger.info("🔍 Verifying upload...")
        blob.reload()
        
        logger.info("✅ Streaming ingestion completed successfully!")
        logger.info(f"   📊 Total examples: {total_examples:,}")
        logger.info(f"   📁 Location: gs://{BUCKET_NAME}/{DATASET_PATH}")
        logger.info(f"   💾 Size: {blob.size / (1024*1024):.1f} MB")
        logger.info(f"   📅 Upload time: {blob.time_created}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during streaming ingestion: {str(e)}")
        return False

def main():
    """
    Main function to execute streaming ingestion.
    """
    logger.info("=" * 60)
    logger.info("OpenHermes 2.5 Cloud Streaming Ingestion")
    logger.info("=" * 60)
    
    # Check environment
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'newllm369-478400')
    logger.info(f"🔧 Project ID: {project_id}")
    
    # Execute streaming ingestion
    success = stream_to_gcs()
    
    if success:
        logger.info("🎉 Container execution completed successfully!")
        logger.info("🎯 Ready for next step: Model training pipeline!")
        exit(0)
    else:
        logger.error("💡 Container execution failed. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    main()