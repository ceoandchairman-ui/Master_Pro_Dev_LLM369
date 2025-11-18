"""
Vertex AI Pipeline for OpenHermes 2.5 Dataset Ingestion

This pipeline downloads the OpenHermes 2.5 dataset from Hugging Face
and stores it in GCS bucket for training our custom LLM.
"""
from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline, Output, Dataset
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=["datasets", "google-cloud-storage", "pandas", "pyarrow"]
)
def download_openhermes_dataset(
    gcs_bucket: str,
    output_dataset: Output[Dataset]
) -> NamedTuple("Outputs", [("dataset_size", int), ("dataset_path", str)]):
    """
    Component to download OpenHermes 2.5 dataset and upload to GCS.
    
    Args:
        gcs_bucket: GCS bucket name for storing the dataset
        output_dataset: Output dataset artifact
        
    Returns:
        Tuple containing dataset size and GCS path
    """
    import json
    import tempfile
    import os
    from datasets import load_dataset
    from google.cloud import storage
    from collections import namedtuple
    
    print("🔄 Loading OpenHermes 2.5 dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    dataset_size = len(dataset)
    
    print(f"📊 Dataset loaded: {dataset_size} examples")
    
    # Create temporary file for the dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
        print("💾 Converting dataset to JSONL format...")
        
        # Convert to JSONL format for efficient processing
        for idx, example in enumerate(dataset):
            json_line = json.dumps(example, ensure_ascii=False)
            temp_file.write(json_line + '\n')
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} examples...")
        
        temp_file_path = temp_file.name
    
    print(f"✅ Dataset saved to temporary file: {temp_file_path}")
    
    # Upload to GCS
    print(f"☁️ Uploading to GCS bucket: {gcs_bucket}")
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    # Create the dataset path
    dataset_blob_name = "raw_data/openhermes-2.5-dataset.jsonl"
    blob = bucket.blob(dataset_blob_name)
    
    # Upload file
    print(f"  Uploading to {dataset_blob_name}...")
    blob.upload_from_filename(temp_file_path)
    
    # Clean up temporary file
    os.unlink(temp_file_path)
    
    dataset_path = f"gs://{gcs_bucket}/{dataset_blob_name}"
    print(f"✅ File uploaded successfully to {dataset_path}")
    
    # Set output dataset metadata
    output_dataset.uri = dataset_path
    output_dataset.metadata = {
        "dataset_name": "OpenHermes-2.5",
        "dataset_size": dataset_size,
        "format": "jsonl",
        "license": "Apache-2.0",
        "source": "teknium/OpenHermes-2.5"
    }
    
    # Return outputs
    outputs = namedtuple("Outputs", ["dataset_size", "dataset_path"])
    return outputs(dataset_size, dataset_path)


@pipeline(
    name="openhermes-data-ingestion-pipeline",
    description="Pipeline to ingest OpenHermes 2.5 dataset for LLM training",
    pipeline_root="gs://newllm369-478400-llm-data/pipeline_artifacts"
)
def data_ingestion_pipeline(
    gcs_bucket: str = "newllm369-478400-llm-data"
):
    """
    Main pipeline to ingest OpenHermes 2.5 dataset.
    
    Args:
        gcs_bucket: GCS bucket name for storing the dataset
    """
    
    # Step 1: Download dataset
    download_task = download_openhermes_dataset(
        gcs_bucket=gcs_bucket
    )
    download_task.set_display_name("Download OpenHermes 2.5 Dataset")
    download_task.set_cpu_limit("4")
    download_task.set_memory_limit("16Gi")


def run_pipeline():
    """
    Function to compile and run the data ingestion pipeline.
    """
    from kfp.v2 import compiler
    
    # Initialize Vertex AI
    aiplatform.init(
        project="newllm369-478400",
        location="us-central1"
    )
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=data_ingestion_pipeline,
        package_path="data_ingestion_pipeline.json"
    )
    
    print("✅ Pipeline compiled successfully!")
    print("📄 Pipeline definition saved as: data_ingestion_pipeline.json")
    
    # Create and run pipeline job
    job = aiplatform.PipelineJob(
        display_name="openhermes-data-ingestion",
        template_path="data_ingestion_pipeline.json",
        pipeline_root="gs://newllm369-478400-llm-data/pipeline_artifacts",
        parameter_values={
            "gcs_bucket": "newllm369-478400-llm-data"
        }
    )
    
    print("🚀 Starting pipeline execution...")
    job.run(sync=True)
    
    print("🎉 Pipeline execution completed!")
    return job


if __name__ == "__main__":
    run_pipeline()