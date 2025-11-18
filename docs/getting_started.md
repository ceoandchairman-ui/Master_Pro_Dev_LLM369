# Getting Started with LLM MLOps Project

This guide will help you get started with the LLM MLOps project, from initial setup to deploying your first chat completion model.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.11 or higher
- Docker and Docker Compose
- Google Cloud SDK (`gcloud`)
- Git
- A Google Cloud Platform account with billing enabled

## Step 1: Project Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/your-org/llm-mlops-project.git
cd llm-mlops-project
```

### 1.2 Create Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Install Pre-commit Hooks

```bash
pre-commit install
```

## Step 2: Google Cloud Configuration

### 2.1 Setup GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
```

### 2.2 Create Service Account

```bash
# Create service account
gcloud iam service-accounts create llm-mlops-sa \
    --display-name="LLM MLOps Service Account" \
    --description="Service account for LLM MLOps operations"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:llm-mlops-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:llm-mlops-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:llm-mlops-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

# Create and download service account key
gcloud iam service-accounts keys create gcp-key.json \
    --iam-account=llm-mlops-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### 2.3 Create Cloud Storage Bucket

```bash
# Create bucket for storing data and models
gsutil mb gs://$PROJECT_ID-llm-mlops

# Set bucket permissions
gsutil iam ch serviceAccount:llm-mlops-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-llm-mlops
```

## Step 3: Environment Configuration

### 3.1 Configure Environment Variables

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your configuration
nano .env
```

Update the following variables in `.env`:

```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
GCS_BUCKET_NAME=your-project-id-llm-mlops
MODEL_NAME=chat-completion-llm
MODEL_VERSION=v1
```

### 3.2 Update Configuration File

Edit `configs/config.yaml` with your project-specific settings:

```yaml
project_id: "your-project-id"
region: "us-central1"
bucket_name: "your-project-id-llm-mlops"

model:
  name: "chat-completion-llm"
  version: "v1"
  # ... other model settings

data:
  train_file: "gs://your-project-id-llm-mlops/data/train.jsonl"
  validation_file: "gs://your-project-id-llm-mlops/data/validation.jsonl"
  # ... other data settings
```

## Step 4: Prepare Training Data

### 4.1 Data Format

Your training data should be in JSONL format with the following structure:

```json
{"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing well, thank you for asking!"}]}
{"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you find weather information!"}]}
```

### 4.2 Upload Training Data

```bash
# Upload your training data
gsutil cp train.jsonl gs://$PROJECT_ID-llm-mlops/data/train.jsonl
gsutil cp validation.jsonl gs://$PROJECT_ID-llm-mlops/data/validation.jsonl

# Validate data format
python scripts/utils.py validate_data --input gs://$PROJECT_ID-llm-mlops/data/train.jsonl
```

## Step 5: Local Development and Testing

### 5.1 Start Development Environment

```bash
# Start local services with Docker Compose
docker-compose up -d

# Check services are running
docker-compose ps
```

### 5.2 Run Local Training

```bash
# Train model locally (small dataset recommended)
python src/training/train.py \
    --config configs/config.yaml
```

### 5.3 Test Inference API

```bash
# Start the inference API locally
python src/inference/api.py

# In another terminal, test the API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Step 6: Cloud Training

### 6.1 Submit Training Job

```bash
# Submit training job to Vertex AI
python scripts/utils.py submit_training_job \
    --project-id $PROJECT_ID \
    --region us-central1 \
    --dataset-path gs://$PROJECT_ID-llm-mlops/data/train.jsonl \
    --model-name chat-completion-v1 \
    --num-epochs 3
```

### 6.2 Monitor Training

```bash
# Monitor training job in Vertex AI console
gcloud ai custom-jobs list --region=us-central1

# View training logs
gcloud logging read "resource.type=gce_instance AND jsonPayload.job_name:chat-completion" --limit=50
```

## Step 7: Deployment

### 7.1 Deploy to Cloud Run

```bash
# Build and deploy to Cloud Run
cd deployment/cloud_run
chmod +x deploy.sh
./deploy.sh

# Test the deployed service
export SERVICE_URL=$(gcloud run services describe llm-chat-api --region=us-central1 --format="value(status.url)")
curl -X POST $SERVICE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

### 7.2 Deploy to Vertex AI Endpoint

```bash
# Deploy to Vertex AI endpoint
python deployment/vertex_ai/deploy.py \
    --config configs/config.yaml \
    --model-path gs://$PROJECT_ID-llm-mlops/models/chat-completion-v1 \
    --test
```

## Step 8: Set Up Monitoring

### 8.1 Configure Monitoring

```bash
# Set up monitoring dashboard
gcloud monitoring dashboards create --config-from-file=monitoring/dashboard.json

# Create alerting policies
gcloud alpha monitoring policies create --policy-from-file=monitoring/alerts.yaml
```

### 8.2 View Metrics

- **Prometheus Metrics**: http://localhost:9090
- **Cloud Monitoring**: https://console.cloud.google.com/monitoring
- **Cloud Logging**: https://console.cloud.google.com/logs

## Step 9: CI/CD Setup

### 9.1 GitHub Secrets

Configure the following secrets in your GitHub repository:

- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_SA_KEY`: Contents of your service account key file
- `GCS_BUCKET`: Your Cloud Storage bucket name

### 9.2 Enable GitHub Actions

The workflows will automatically trigger on:
- Push to `main` branch (CI/CD pipeline)
- Manual trigger (Model training pipeline)
- Schedule (Monitoring pipeline)

## Next Steps

1. **Customize the Model**: Modify `src/models/chat_completion_model.py` for your specific use case
2. **Fine-tune Training**: Adjust hyperparameters in `configs/config.yaml`
3. **Add Custom Features**: Extend the API with additional endpoints
4. **Scale Deployment**: Configure auto-scaling and load balancing
5. **Implement A/B Testing**: Add model versioning and traffic splitting

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure service account has necessary permissions
   - Check `GOOGLE_APPLICATION_CREDENTIALS` environment variable

2. **Training Failures**
   - Verify data format and accessibility
   - Check Vertex AI quotas and limits
   - Review training logs for specific errors

3. **API Deployment Issues**
   - Verify Docker image builds successfully
   - Check Cloud Run service logs
   - Ensure all environment variables are set

4. **Monitoring Problems**
   - Verify monitoring APIs are enabled
   - Check service account monitoring permissions
   - Review metric export configuration

### Getting Help

- Check the [FAQ](docs/faq.md)
- Review [troubleshooting guide](docs/troubleshooting.md)
- Open an issue on GitHub
- Check Google Cloud documentation

## Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

Happy building! 🚀