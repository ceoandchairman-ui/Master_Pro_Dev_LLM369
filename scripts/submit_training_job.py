"""
Submit Training Job to Vertex AI
Builds Docker image, pushes to GCR, and submits training job with 4×L4 GPUs
"""

import os
import subprocess
import argparse
from datetime import datetime
from google.cloud import aiplatform

# Configuration
PROJECT_ID = "newllm369-478400"
REGION = "us-central1"
BUCKET_NAME = "newllm369-478400-llm-data"
IMAGE_NAME = "custom-llm-training"
IMAGE_TAG = datetime.now().strftime('%Y%m%d-%H%M%S')
IMAGE_URI = f"gcr.io/{PROJECT_ID}/{IMAGE_NAME}:{IMAGE_TAG}"


def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(f"stderr: {result.stderr}")
        raise Exception(f"Command failed: {cmd}")
    
    print(result.stdout)
    print(f"✅ {description} completed successfully\n")
    return result.stdout


def build_docker_image():
    """Build Docker image for training"""
    cmd = f"docker build -t {IMAGE_URI} -f deployment/training/Dockerfile ."
    run_command(cmd, "Building Docker image")


def push_docker_image():
    """Push Docker image to Google Container Registry"""
    # Configure Docker for GCR
    run_command("gcloud auth configure-docker", "Configuring Docker for GCR")
    
    # Push image
    cmd = f"docker push {IMAGE_URI}"
    run_command(cmd, "Pushing Docker image to GCR")


def submit_training_job():
    """Submit training job to Vertex AI"""
    print(f"\n{'='*80}")
    print("Submitting Training Job to Vertex AI")
    print(f"{'='*80}\n")
    
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Job display name
    job_name = f"custom-llm-training-{IMAGE_TAG}"
    
    # Create custom job
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "g2-standard-48",  # 4×L4 GPUs
                    "accelerator_type": "NVIDIA_L4",
                    "accelerator_count": 4,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": IMAGE_URI,
                    "env": [
                        {"name": "CONFIG_PATH", "value": "configs/custom_llm_config.yaml"},
                        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                    ],
                },
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": 200,
                },
            }
        ],
        base_output_dir=f"gs://{BUCKET_NAME}/training_output",
    )
    
    print(f"Job Name: {job_name}")
    print(f"Image URI: {IMAGE_URI}")
    print(f"Machine Type: g2-standard-48 (4×L4 GPUs)")
    print(f"Output Directory: gs://{BUCKET_NAME}/training_output")
    print(f"\n🚀 Submitting job to Vertex AI...")
    
    # Submit job (non-blocking)
    job.submit()
    
    print(f"\n✅ Job submitted successfully!")
    print(f"\n{'='*80}")
    print("Monitoring Links:")
    print(f"{'='*80}")
    print(f"Vertex AI Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"Job Details: {job.resource_name}")
    print(f"\nJob will run for approximately 24-30 hours.")
    print(f"Estimated cost: ~$86")
    print(f"\nTo monitor:")
    print(f"1. Go to Vertex AI Console (link above)")
    print(f"2. Click on job: {job_name}")
    print(f"3. View TensorBoard tab for training metrics")
    print(f"4. Check Logs tab for detailed progress")
    
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit training job to Vertex AI")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build (use existing image)")
    parser.add_argument("--skip-push", action="store_true", help="Skip Docker push (use existing image)")
    parser.add_argument("--build-only", action="store_true", help="Only build Docker image, don't submit job")
    args = parser.parse_args()
    
    try:
        # Step 1: Build Docker image
        if not args.skip_build:
            build_docker_image()
        else:
            print("⏭️  Skipping Docker build")
        
        if args.build_only:
            print("\n✅ Build complete. Exiting (--build-only flag set)")
            return
        
        # Step 2: Push to GCR
        if not args.skip_push:
            push_docker_image()
        else:
            print("⏭️  Skipping Docker push")
        
        # Step 3: Submit training job
        job = submit_training_job()
        
        print(f"\n{'='*80}")
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nYour training job is now running on Vertex AI!")
        print(f"Check status at: https://console.cloud.google.com/vertex-ai/training")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*80}")
        raise


if __name__ == "__main__":
    main()
