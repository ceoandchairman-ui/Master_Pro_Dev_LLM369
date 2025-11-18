"""
Google Cloud Storage utilities for MLOps workflows
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Union
from google.cloud import storage
from google.cloud.exceptions import NotFound
import tempfile
import shutil

logger = logging.getLogger(__name__)


class GCSManager:
    """Manager for Google Cloud Storage operations"""
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.client = storage.Client(project=self.project_id)
    
    def parse_gcs_path(self, gcs_path: str) -> tuple[str, str]:
        """Parse GCS path into bucket and blob name"""
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        path = gcs_path[5:]  # Remove 'gs://'
        parts = path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ''
        
        return bucket_name, blob_name
    
    def upload_file(self, local_path: Union[str, Path], gcs_path: str) -> None:
        """Upload a file to GCS"""
        local_path = Path(local_path)
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path} to {gcs_path}")
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {gcs_path}: {e}")
            raise
    
    def download_file(self, gcs_path: str, local_path: Union[str, Path]) -> None:
        """Download a file from GCS"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded {gcs_path} to {local_path}")
            
        except NotFound:
            logger.error(f"File not found: {gcs_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to download {gcs_path} to {local_path}: {e}")
            raise
    
    def upload_directory(self, local_dir: Union[str, Path], gcs_prefix: str) -> None:
        """Upload a directory to GCS"""
        local_dir = Path(local_dir)
        bucket_name, prefix = self.parse_gcs_path(gcs_prefix)
        
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
        
        try:
            bucket = self.client.bucket(bucket_name)
            
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    blob_name = f"{prefix}/{relative_path}".replace('\\', '/')
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    
            logger.info(f"Uploaded directory {local_dir} to gs://{bucket_name}/{prefix}")
            
        except Exception as e:
            logger.error(f"Failed to upload directory {local_dir}: {e}")
            raise
    
    def download_directory(self, gcs_prefix: str, local_dir: Union[str, Path]) -> None:
        """Download a directory from GCS"""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        bucket_name, prefix = self.parse_gcs_path(gcs_prefix)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directory markers
                    # Remove prefix from blob name
                    relative_path = blob.name[len(prefix):].lstrip('/')
                    local_file_path = local_dir / relative_path
                    
                    # Create parent directories
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    blob.download_to_filename(str(local_file_path))
            
            logger.info(f"Downloaded directory gs://{bucket_name}/{prefix} to {local_dir}")
            
        except Exception as e:
            logger.error(f"Failed to download directory gs://{bucket_name}/{prefix}: {e}")
            raise
    
    def list_files(self, gcs_prefix: str) -> List[str]:
        """List files in GCS with given prefix"""
        bucket_name, prefix = self.parse_gcs_path(gcs_prefix)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files with prefix {gcs_prefix}: {e}")
            raise
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if file exists in GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
            
        except Exception as e:
            logger.error(f"Error checking if file exists {gcs_path}: {e}")
            return False
    
    def delete_file(self, gcs_path: str) -> None:
        """Delete a file from GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            
            logger.info(f"Deleted {gcs_path}")
            
        except NotFound:
            logger.warning(f"File not found: {gcs_path}")
        except Exception as e:
            logger.error(f"Failed to delete {gcs_path}: {e}")
            raise
    
    def copy_file(self, source_gcs_path: str, dest_gcs_path: str) -> None:
        """Copy a file within GCS"""
        source_bucket_name, source_blob_name = self.parse_gcs_path(source_gcs_path)
        dest_bucket_name, dest_blob_name = self.parse_gcs_path(dest_gcs_path)
        
        try:
            source_bucket = self.client.bucket(source_bucket_name)
            source_blob = source_bucket.blob(source_blob_name)
            
            dest_bucket = self.client.bucket(dest_bucket_name)
            
            source_bucket.copy_blob(source_blob, dest_bucket, dest_blob_name)
            logger.info(f"Copied {source_gcs_path} to {dest_gcs_path}")
            
        except Exception as e:
            logger.error(f"Failed to copy {source_gcs_path} to {dest_gcs_path}: {e}")
            raise
    
    def get_file_metadata(self, gcs_path: str) -> dict:
        """Get metadata for a GCS file"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.reload()
            
            return {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created,
                'updated': blob.updated,
                'content_type': blob.content_type,
                'etag': blob.etag,
                'generation': blob.generation,
                'storage_class': blob.storage_class
            }
            
        except NotFound:
            logger.error(f"File not found: {gcs_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to get metadata for {gcs_path}: {e}")
            raise