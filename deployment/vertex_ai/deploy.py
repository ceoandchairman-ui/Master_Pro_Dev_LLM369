"""
Vertex AI endpoint deployment script
"""

import os
import logging
import argparse
import yaml
from typing import Dict, Any
from google.cloud import aiplatform
from google.cloud.aiplatform import Model, Endpoint
import time

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class VertexAIDeployer:
    """Deployer for Vertex AI endpoints"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_vertex_ai()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config['project_id'] = os.getenv('GCP_PROJECT_ID', config['project_id'])
        config['region'] = os.getenv('GCP_REGION', config['region'])
        
        return config
    
    def _setup_vertex_ai(self):
        """Initialize Vertex AI client"""
        aiplatform.init(
            project=self.config['project_id'],
            location=self.config['region']
        )
        
        logger.info(f"Initialized Vertex AI for project {self.config['project_id']} in {self.config['region']}")
    
    def upload_model(self, model_path: str, display_name: str) -> Model:
        """Upload model to Vertex AI Model Registry"""
        logger.info(f"Uploading model from {model_path}...")
        
        # Check if model already exists
        existing_models = Model.list(
            filter=f'display_name="{display_name}"',
            order_by="create_time desc"
        )
        
        if existing_models:
            logger.info(f"Found existing model: {display_name}")
            return existing_models[0]
        
        # Upload new model
        model = Model.upload(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri="gcr.io/vertex-ai/prediction/pytorch-gpu.1-12:latest",
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            description=f"Chat completion model - {display_name}",
            labels={
                "model_type": "chat_completion",
                "framework": "pytorch",
                "version": self.config.get('model', {}).get('version', 'v1')
            }
        )
        
        logger.info(f"Model uploaded successfully: {model.resource_name}")
        return model
    
    def create_endpoint(self, display_name: str) -> Endpoint:
        """Create Vertex AI endpoint"""
        logger.info(f"Creating endpoint: {display_name}")
        
        # Check if endpoint already exists
        existing_endpoints = Endpoint.list(
            filter=f'display_name="{display_name}"',
            order_by="create_time desc"
        )
        
        if existing_endpoints:
            logger.info(f"Found existing endpoint: {display_name}")
            return existing_endpoints[0]
        
        # Create new endpoint
        endpoint = Endpoint.create(
            display_name=display_name,
            description=f"Chat completion endpoint - {display_name}",
            labels={
                "service_type": "chat_completion",
                "environment": os.getenv('ENVIRONMENT', 'production')
            }
        )
        
        logger.info(f"Endpoint created successfully: {endpoint.resource_name}")
        return endpoint
    
    def deploy_model_to_endpoint(
        self,
        model: Model,
        endpoint: Endpoint,
        deployment_config: Dict[str, Any]
    ) -> None:
        """Deploy model to endpoint"""
        logger.info("Deploying model to endpoint...")
        
        deployed_model = endpoint.deploy(
            model=model,
            deployed_model_display_name=f"{model.display_name}-deployment",
            machine_type=deployment_config.get('machine_type', 'n1-standard-4'),
            min_replica_count=deployment_config.get('min_replica_count', 1),
            max_replica_count=deployment_config.get('max_replica_count', 5),
            accelerator_type=deployment_config.get('accelerator_type'),
            accelerator_count=deployment_config.get('accelerator_count', 0),
            traffic_percentage=100,
            sync=True
        )
        
        logger.info(f"Model deployed successfully: {deployed_model.id}")
    
    def deploy(self, model_path: str) -> Dict[str, str]:
        """Main deployment function"""
        try:
            # Configuration
            model_name = self.config['model']['name']
            version = self.config['model']['version']
            display_name = f"{model_name}-{version}"
            endpoint_name = self.config['deployment']['vertex_endpoint']['display_name']
            
            # Upload model
            model = self.upload_model(model_path, display_name)
            
            # Create endpoint
            endpoint = self.create_endpoint(endpoint_name)
            
            # Deploy model to endpoint
            deployment_config = self.config['deployment']['vertex_endpoint']
            self.deploy_model_to_endpoint(model, endpoint, deployment_config)
            
            # Return deployment info
            return {
                'model_id': model.resource_name,
                'endpoint_id': endpoint.resource_name,
                'endpoint_url': f"https://{self.config['region']}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict",
                'status': 'deployed'
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def test_endpoint(self, endpoint_id: str) -> bool:
        """Test deployed endpoint"""
        try:
            endpoint = Endpoint(endpoint_id)
            
            # Test prediction
            test_request = {
                "instances": [{
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.7
                }]
            }
            
            response = endpoint.predict(instances=test_request["instances"])
            
            if response.predictions:
                logger.info("Endpoint test successful")
                return True
            else:
                logger.error("Endpoint test failed: No predictions returned")
                return False
                
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            return False
    
    def cleanup_old_deployments(self, keep_count: int = 3):
        """Clean up old model deployments"""
        logger.info(f"Cleaning up old deployments, keeping {keep_count} latest...")
        
        try:
            # List all models
            models = Model.list(order_by="create_time desc")
            
            # Filter models by our naming pattern
            our_models = [m for m in models if self.config['model']['name'] in m.display_name]
            
            # Keep only the specified number of latest models
            if len(our_models) > keep_count:
                models_to_delete = our_models[keep_count:]
                
                for model in models_to_delete:
                    try:
                        # Undeploy from all endpoints first
                        for endpoint in Endpoint.list():
                            deployed_models = endpoint.list_models()
                            for deployed_model in deployed_models:
                                if deployed_model.model == model.resource_name:
                                    endpoint.undeploy(deployed_model_id=deployed_model.id)
                                    logger.info(f"Undeployed model {model.display_name} from endpoint")
                        
                        # Delete the model
                        model.delete()
                        logger.info(f"Deleted old model: {model.display_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to delete model {model.display_name}: {e}")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deploy model to Vertex AI')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='GCS path to the model artifacts')
    parser.add_argument('--test', action='store_true',
                       help='Test the endpoint after deployment')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old deployments')
    
    return parser.parse_args()


def main():
    """Main deployment function"""
    args = parse_args()
    
    # Setup logging
    setup_logging(level="INFO", format="standard")
    
    # Initialize deployer
    deployer = VertexAIDeployer(args.config)
    
    try:
        # Deploy model
        deployment_info = deployer.deploy(args.model_path)
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Model ID: {deployment_info['model_id']}")
        logger.info(f"Endpoint ID: {deployment_info['endpoint_id']}")
        logger.info(f"Endpoint URL: {deployment_info['endpoint_url']}")
        
        # Test endpoint if requested
        if args.test:
            logger.info("Testing endpoint...")
            if deployer.test_endpoint(deployment_info['endpoint_id']):
                logger.info("Endpoint test passed!")
            else:
                logger.error("Endpoint test failed!")
                return 1
        
        # Cleanup old deployments if requested
        if args.cleanup:
            deployer.cleanup_old_deployments()
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())