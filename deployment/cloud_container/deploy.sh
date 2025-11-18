#!/bin/bash

# Cloud Container Deployment Script for OpenHermes Data Ingestion
# This script builds and deploys the container to Google Cloud Run

set -e

# Configuration
PROJECT_ID="newllm369-478400"
REGION="us-central1"
SERVICE_NAME="openhermes-ingestion"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting Cloud Container Deployment..."
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service: ${SERVICE_NAME}"
echo ""

# Step 1: Build container image
echo "🔨 Building container image..."
docker build -t ${IMAGE_NAME} .

if [ $? -eq 0 ]; then
    echo "✅ Container built successfully!"
else
    echo "❌ Container build failed!"
    exit 1
fi

# Step 2: Push to Container Registry
echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo "✅ Image pushed successfully!"
else
    echo "❌ Image push failed!"
    exit 1
fi

# Step 3: Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --max-instances 1 \
    --no-traffic

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Trigger the service via Cloud Console or gcloud"
    echo "   2. Monitor logs: gcloud logs read --service=${SERVICE_NAME}"
    echo "   3. Check GCS bucket for uploaded dataset"
else
    echo "❌ Deployment failed!"
    exit 1
fi

echo ""
echo "🎉 Cloud container deployment completed!"