"""
Data Analysis Script for OpenHermes 2.5 Dataset

This script analyzes the ingested dataset to understand:
- Data structure and format
- Conversation patterns
- Statistics and quality metrics
"""

import json
import pandas as pd
from google.cloud import storage
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenHermesDataExplorer:
    def __init__(self, bucket_name, dataset_path):
        self.bucket_name = bucket_name
        self.dataset_path = dataset_path
        self.client = storage.Client()
        
    def sample_data(self, num_samples=10):
        """
        Read first N samples from the dataset to understand structure
        """
        logger.info(f"📊 Sampling {num_samples} examples from dataset...")
        
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(self.dataset_path)
        
        samples = []
        with blob.open("r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {i}: {e}")
                    
        return samples
    
    def analyze_structure(self, samples):
        """
        Analyze the structure of conversation data
        """
        logger.info("🔍 Analyzing data structure...")
        
        if not samples:
            logger.error("No valid samples found!")
            return
            
        # Analyze first sample structure
        first_sample = samples[0]
        logger.info(f"📋 Sample keys: {list(first_sample.keys())}")
        
        # Print first sample for inspection
        logger.info("📝 First sample:")
        print(json.dumps(first_sample, indent=2, ensure_ascii=False)[:1000] + "...")
        
        # Analyze conversation structure
        if 'conversations' in first_sample:
            conv = first_sample['conversations']
            logger.info(f"💬 Conversation structure: {len(conv)} messages")
            for i, msg in enumerate(conv[:3]):  # Show first 3 messages
                msg_content = msg.get('value', '')
                logger.info(f"   Message {i}: {msg.get('from', 'unknown')} - {msg_content[:100]}...")
        
        return first_sample
    
    def conversation_statistics(self, num_samples=1000):
        """
        Analyze conversation patterns and statistics
        """
        logger.info(f"📈 Analyzing conversation statistics from {num_samples} samples...")
        
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(self.dataset_path)
        
        stats = {
            'total_conversations': 0,
            'conversation_lengths': [],
            'message_types': Counter(),
            'avg_message_length': [],
            'topics': Counter(),
        }
        
        with blob.open("r") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    stats['total_conversations'] += 1
                    
                    if 'conversations' in data:
                        conv = data['conversations']
                        stats['conversation_lengths'].append(len(conv))
                        
                        for msg in conv:
                            msg_type = msg.get('from', 'unknown')
                            stats['message_types'][msg_type] += 1
                            
                            msg_content = msg.get('value', '')
                            stats['avg_message_length'].append(len(msg_content))
                    
                    # Track topics/categories if available
                    if 'category' in data:
                        stats['topics'][data['category']] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        # Calculate averages
        if stats['conversation_lengths']:
            avg_conv_length = sum(stats['conversation_lengths']) / len(stats['conversation_lengths'])
            logger.info(f"📊 Average conversation length: {avg_conv_length:.1f} messages")
        
        if stats['avg_message_length']:
            avg_msg_length = sum(stats['avg_message_length']) / len(stats['avg_message_length'])
            logger.info(f"📊 Average message length: {avg_msg_length:.1f} characters")
        
        logger.info(f"📊 Message types: {dict(stats['message_types'])}")
        
        if stats['topics']:
            logger.info(f"📊 Top topics: {dict(stats['topics'].most_common(5))}")
        
        return stats
    
    def data_quality_check(self, num_samples=1000):
        """
        Check data quality and identify potential issues
        """
        logger.info(f"🔍 Performing data quality check on {num_samples} samples...")
        
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(self.dataset_path)
        
        quality_issues = {
            'empty_messages': 0,
            'malformed_json': 0,
            'missing_conversations': 0,
            'incomplete_conversations': 0,
            'total_checked': 0
        }
        
        with blob.open("r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                    
                quality_issues['total_checked'] += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    # Check for missing conversations
                    if 'conversations' not in data:
                        quality_issues['missing_conversations'] += 1
                        continue
                    
                    conv = data['conversations']
                    
                    # Check for incomplete conversations (should have human + assistant)
                    if len(conv) < 2:
                        quality_issues['incomplete_conversations'] += 1
                    
                    # Check for empty messages
                    for msg in conv:
                        if not msg.get('value', '').strip():
                            quality_issues['empty_messages'] += 1
                            
                except json.JSONDecodeError:
                    quality_issues['malformed_json'] += 1
        
        logger.info("🎯 Data Quality Results:")
        for issue, count in quality_issues.items():
            if issue != 'total_checked':
                percentage = (count / quality_issues['total_checked']) * 100
                logger.info(f"   {issue}: {count} ({percentage:.2f}%)")
        
        return quality_issues

def main():
    """
    Main analysis function
    """
    logger.info("=" * 60)
    logger.info("OpenHermes 2.5 Data Analysis")
    logger.info("=" * 60)
    
    # Configuration
    BUCKET_NAME = "newllm369-478400-llm-data"
    DATASET_PATH = "raw_data/openhermes-2.5-dataset.jsonl"
    
    # Initialize explorer
    explorer = OpenHermesDataExplorer(BUCKET_NAME, DATASET_PATH)
    
    # Step 1: Sample and analyze structure
    samples = explorer.sample_data(num_samples=5)
    structure = explorer.analyze_structure(samples)
    
    # Step 2: Conversation statistics
    stats = explorer.conversation_statistics(num_samples=1000)
    
    # Step 3: Data quality check
    quality = explorer.data_quality_check(num_samples=1000)
    
    logger.info("✅ Data analysis completed!")
    logger.info("🎯 Ready for preprocessing pipeline setup!")

if __name__ == "__main__":
    main()