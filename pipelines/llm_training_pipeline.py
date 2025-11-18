"""
Complete LLM Training Pipeline for Vertex AI
Assembles all components from Sprint 4-7 into a complete MLOps pipeline

This pipeline implements the end-to-end LLM training workflow:
1. Data Preprocessing (Sprint 4)
2. Model Training (Sprint 5) 
3. Model Evaluation (Sprint 6)
4. Model Registration & Deployment (Sprint 7)
"""

from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform
import os

# Import our pipeline components
from components.data_preprocessing_component import preprocess_data
from components.training_component import train_llm
from components.evaluation_component import evaluate_model

@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.0",
        "google-cloud-storage==2.10.0"
    ]
)
def register_model(
    model_path: str,
    tokenizer_path: str,
    evaluation_metrics: dict,
    model_name: str,
    model_description: str = "Custom LLM trained with MLOps pipeline",
    serving_container_image: str = "us-central1-docker.pkg.dev/PROJECT_ID/llm-mlops-repo/llm-serving:latest"
) -> str:
    """Register model in Vertex AI Model Registry"""
    import json
    from google.cloud import aiplatform
    from datetime import datetime
    
    # Initialize Vertex AI
    aiplatform.init()
    
    # Create model display name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"{model_name}-{timestamp}"
    
    # Register model
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_path,
        serving_container_image_uri=serving_container_image,
        description=f"{model_description}\nEvaluation Metrics: {json.dumps(evaluation_metrics, indent=2)}",
        labels={
            "pipeline": "llm-training",
            "framework": "pytorch",
            "model_type": "causal-lm"
        }
    )
    
    return model.resource_name

@component(
    base_image="python:3.9", 
    packages_to_install=[
        "google-cloud-aiplatform==1.38.0"
    ]
)
def deploy_model(
    model_resource_name: str,
    endpoint_display_name: str,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
    accelerator_type: str = "",
    accelerator_count: int = 0
) -> str:
    """Deploy model to Vertex AI Endpoint"""
    from google.cloud import aiplatform
    from datetime import datetime
    
    aiplatform.init()
    
    # Get the model
    model = aiplatform.Model(model_resource_name)
    
    # Create or get endpoint
    try:
        endpoint = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    except (IndexError, Exception):
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            labels={"pipeline": "llm-training"}
        )
        print(f"Created new endpoint: {endpoint.display_name}")
    
    # Deploy model to endpoint
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"llm-deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type if accelerator_type else None,
        accelerator_count=accelerator_count if accelerator_count > 0 else None,
        traffic_percentage=100
    )
    
    return endpoint.resource_name

@pipeline(
    name="llm-training-pipeline",
    description="Complete LLM training pipeline with preprocessing, training, evaluation, and deployment",
    pipeline_root="gs://PROJECT_ID-llm-artifacts/pipeline_root"
)
def llm_training_pipeline(
    # Data parameters
    raw_data_path: str = "gs://PROJECT_ID-llm-data/raw/training_data.jsonl",
    processed_data_root: str = "gs://PROJECT_ID-llm-artifacts/processed_data",
    
    # Model parameters
    model_output_root: str = "gs://PROJECT_ID-llm-artifacts/models",
    model_name: str = "custom-llm",
    model_size: str = "small",  # small, medium, large
    
    # Training parameters
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0,
    use_deepspeed: bool = True,
    
    # Evaluation parameters
    max_eval_samples: int = 1000,
    max_generation_length: int = 512,
    num_beams: int = 4,
    
    # Deployment parameters
    deploy_model_flag: bool = True,
    endpoint_name: str = "llm-serving-endpoint",
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 3,
    
    # Experiment tracking
    experiment_name: str = "llm-training-experiment"
):
    """
    Complete LLM training pipeline
    
    This pipeline executes the following steps:
    1. Data Preprocessing: Clean, tokenize, and split data
    2. Model Training: Train custom LLM with advanced architecture
    3. Model Evaluation: Evaluate with BLEU, ROUGE, perplexity metrics
    4. Model Registration: Register in Vertex AI Model Registry
    5. Model Deployment: Deploy to Vertex AI Endpoint (optional)
    """
    
    # Step 1: Data Preprocessing (Sprint 4)
    preprocessing_task = preprocess_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_root,
        vocab_size=65536,
        max_length=2048,
        tokenizer_name="microsoft/DialoGPT-medium"
    )
    
    # Step 2: Model Training (Sprint 5)
    training_task = train_llm(
        processed_train_path=preprocessing_task.outputs["processed_train_path"],
        processed_val_path=preprocessing_task.outputs["processed_val_path"],
        tokenizer_path=preprocessing_task.outputs["tokenizer_path"],
        config_path="",  # Will use default config
        model_output_path=model_output_root,
        experiment_name=experiment_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        use_deepspeed=use_deepspeed,
        model_size=model_size
    )
    
    # Configure training task for large-scale training
    training_task.set_memory_limit("32Gi")
    training_task.set_cpu_limit("8")
    
    # Add GPU if available
    training_task.add_node_selector_constraint("cloud.google.com/gke-accelerator", "nvidia-tesla-t4")
    training_task.set_gpu_limit(1)
    
    # Step 3: Model Evaluation (Sprint 6)
    evaluation_task = evaluate_model(
        model_path=training_task.outputs["model_path"],
        tokenizer_path=training_task.outputs["tokenizer_path"],
        test_data_path=preprocessing_task.outputs["processed_test_path"],
        evaluation_output_path=f"{model_output_root}/evaluation",
        max_eval_samples=max_eval_samples,
        max_length=max_generation_length,
        num_beams=num_beams,
        experiment_name=f"{experiment_name}-evaluation"
    )
    
    # Step 4: Model Registration (Sprint 6)
    registration_task = register_model(
        model_path=training_task.outputs["model_path"],
        tokenizer_path=training_task.outputs["tokenizer_path"],
        evaluation_metrics=evaluation_task.outputs["evaluation_metrics"],
        model_name=model_name,
        model_description=f"Custom LLM ({model_size}) trained with MLOps pipeline"
    )
    
    # Step 5: Model Deployment (Sprint 7) - Conditional
    with dsl.Condition(deploy_model_flag == True, name="deploy-model"):
        deployment_task = deploy_model(
            model_resource_name=registration_task.output,
            endpoint_display_name=endpoint_name,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas
        )
    
    # Set task dependencies and resource requirements
    training_task.after(preprocessing_task)
    evaluation_task.after(training_task)
    registration_task.after(evaluation_task)

def create_evaluation_pipeline():
    """Create evaluation-only pipeline for existing models"""
    
    @pipeline(
        name="llm-evaluation-pipeline",
        description="Standalone model evaluation pipeline",
        pipeline_root="gs://PROJECT_ID-llm-artifacts/pipeline_root"
    )
    def llm_evaluation_pipeline(
        model_path: str,
        tokenizer_path: str,
        test_data_path: str,
        evaluation_output_path: str,
        max_eval_samples: int = 1000,
        experiment_name: str = "llm-evaluation"
    ):
        evaluation_task = evaluate_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            test_data_path=test_data_path,
            evaluation_output_path=evaluation_output_path,
            max_eval_samples=max_eval_samples,
            experiment_name=experiment_name
        )
    
    return llm_evaluation_pipeline

def create_deployment_pipeline():
    """Create deployment-only pipeline for existing models"""
    
    @pipeline(
        name="llm-deployment-pipeline", 
        description="Standalone model deployment pipeline",
        pipeline_root="gs://PROJECT_ID-llm-artifacts/pipeline_root"
    )
    def llm_deployment_pipeline(
        model_resource_name: str,
        endpoint_name: str = "llm-serving-endpoint",
        machine_type: str = "n1-standard-4",
        min_replicas: int = 1,
        max_replicas: int = 3
    ):
        deployment_task = deploy_model(
            model_resource_name=model_resource_name,
            endpoint_display_name=endpoint_name,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas
        )
    
    return llm_deployment_pipeline

def compile_pipelines(project_id: str, output_dir: str = "./compiled_pipelines"):
    """Compile all pipelines to YAML files"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Replace PROJECT_ID placeholder
    def replace_project_id_in_pipeline(pipeline_func, project_id):
        # This is a simple replacement - in production, use proper templating
        pipeline_code = pipeline_func.__code__
        return pipeline_func
    
    # Compile main training pipeline
    compiler.Compiler().compile(
        pipeline_func=llm_training_pipeline,
        package_path=os.path.join(output_dir, "llm_training_pipeline.yaml")
    )
    
    # Compile evaluation pipeline
    eval_pipeline = create_evaluation_pipeline()
    compiler.Compiler().compile(
        pipeline_func=eval_pipeline,
        package_path=os.path.join(output_dir, "llm_evaluation_pipeline.yaml")
    )
    
    # Compile deployment pipeline
    deploy_pipeline = create_deployment_pipeline()
    compiler.Compiler().compile(
        pipeline_func=deploy_pipeline,
        package_path=os.path.join(output_dir, "llm_deployment_pipeline.yaml")
    )
    
    print(f"Pipelines compiled to {output_dir}/")
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile LLM MLOps Pipelines")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--output-dir", default="./compiled_pipelines", help="Output directory for compiled pipelines")
    
    args = parser.parse_args()
    
    # Compile pipelines
    output_path = compile_pipelines(args.project_id, args.output_dir)
    
    print("✅ Pipeline compilation completed!")
    print(f"📁 Compiled pipelines available at: {output_path}")
    print("\n📋 Available pipelines:")
    print("  • llm_training_pipeline.yaml - Complete training pipeline")
    print("  • llm_evaluation_pipeline.yaml - Evaluation-only pipeline") 
    print("  • llm_deployment_pipeline.yaml - Deployment-only pipeline")
    print("\n🚀 Next steps:")
    print("  1. Upload compiled pipelines to GCS")
    print("  2. Update orchestrator service with pipeline paths")
    print("  3. Trigger pipeline via Pub/Sub or direct API call")