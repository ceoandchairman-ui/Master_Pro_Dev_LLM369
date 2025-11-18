# LLM MLOps Project Instructions

This project is a comprehensive MLOps implementation for developing, training, and deploying a custom Large Language Model (LLM) chat completion model using Google Cloud Platform services.

## Project Overview
- **Type**: MLOps LLM Chat Completion Model
- **Platform**: Google Cloud Platform (Vertex AI)
- **Languages**: Python, YAML, Dockerfile
- **Key Services**: Vertex AI, Cloud Run, Cloud Storage, Cloud Build

## Architecture Components
- Data pipeline for preprocessing and tokenization
- Custom transformer model training on Vertex AI
- CI/CD pipelines for automated deployment
- Model serving via Cloud Run and Vertex AI endpoints
- Monitoring and observability for production models

## Development Guidelines
- Follow MLOps best practices for version control and reproducibility
- Use Vertex AI Pipelines for orchestration
- Implement proper logging and monitoring
- Maintain separation between training and inference code
- Use containerization for all components

## Code Organization
- `src/`: Core model and pipeline code
- `pipelines/`: Vertex AI pipeline definitions
- `deployment/`: Cloud Run and endpoint configurations
- `data/`: Data processing and validation scripts
- `tests/`: Comprehensive test suites
- `configs/`: Environment and model configurations