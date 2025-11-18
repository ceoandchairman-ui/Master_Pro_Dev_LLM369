"""
Data preprocessing pipeline for chat completion model
Handles tokenization, formatting, and data validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset as HFDataset, load_dataset
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConversationExample:
    """Represents a single conversation example"""
    messages: List[Dict[str, str]]
    conversation_id: Optional[str] = None
    metadata: Optional[Dict] = None


class ChatDataProcessor:
    """Processes chat data for training and inference"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        chat_template: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template or self._default_chat_template()
        
        # Special tokens
        self.bos_token = tokenizer.bos_token or tokenizer.eos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token or tokenizer.eos_token
        
    def _default_chat_template(self) -> str:
        """Default chat template for conversation formatting"""
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|system|>\n{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "<|user|>\n{{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>\n{{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )
    
    def format_conversation(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """Format conversation using chat template"""
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
            else:
                # Fallback to manual formatting
                formatted = ""
                for message in messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    if role == 'system':
                        formatted += f"<|system|>\n{content}\n"
                    elif role == 'user':
                        formatted += f"<|user|>\n{content}\n"
                    elif role == 'assistant':
                        formatted += f"<|assistant|>\n{content}\n"
                
                if add_generation_prompt:
                    formatted += "<|assistant|>\n"
                
                return formatted
        except Exception as e:
            logger.error(f"Error formatting conversation: {e}")
            raise
    
    def tokenize_conversation(
        self,
        conversation: ConversationExample,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a conversation for training"""
        formatted_text = self.format_conversation(conversation.messages)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=return_tensors
        )
        
        # Create labels (same as input_ids for language modeling)
        encoding['labels'] = encoding['input_ids'].clone()
        
        return encoding
    
    def create_training_example(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Create a training example from messages"""
        conversation = ConversationExample(messages=messages)
        return self.tokenize_conversation(conversation)
    
    def prepare_inference_input(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Prepare input for inference"""
        formatted_text = self.format_conversation(messages, add_generation_prompt=True)
        
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return encoding


class ChatDataset(Dataset):
    """PyTorch Dataset for chat completion training"""
    
    def __init__(
        self,
        data: Union[str, Path, List[Dict], pd.DataFrame],
        processor: ChatDataProcessor,
        split: str = "train"
    ):
        self.processor = processor
        self.split = split
        self.examples = self._load_data(data)
        
        logger.info(f"Loaded {len(self.examples)} examples for {split} split")
    
    def _load_data(self, data: Union[str, Path, List[Dict], pd.DataFrame]) -> List[ConversationExample]:
        """Load data from various sources"""
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == '.jsonl':
                return self._load_jsonl(data_path)
            elif data_path.suffix == '.json':
                return self._load_json(data_path)
            elif data_path.suffix == '.csv':
                return self._load_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        elif isinstance(data, list):
            return [ConversationExample(messages=item['messages']) for item in data]
        elif isinstance(data, pd.DataFrame):
            return self._load_from_dataframe(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _load_jsonl(self, path: Path) -> List[ConversationExample]:
        """Load data from JSONL file"""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if 'messages' in data:
                        examples.append(ConversationExample(
                            messages=data['messages'],
                            conversation_id=data.get('id'),
                            metadata=data.get('metadata')
                        ))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON on line {line_num + 1}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num + 1}: {e}")
        
        return examples
    
    def _load_json(self, path: Path) -> List[ConversationExample]:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [ConversationExample(messages=item['messages']) for item in data]
        elif isinstance(data, dict) and 'conversations' in data:
            return [ConversationExample(messages=conv['messages']) for conv in data['conversations']]
        else:
            raise ValueError("Unsupported JSON structure")
    
    def _load_csv(self, path: Path) -> List[ConversationExample]:
        """Load data from CSV file"""
        df = pd.read_csv(path)
        return self._load_from_dataframe(df)
    
    def _load_from_dataframe(self, df: pd.DataFrame) -> List[ConversationExample]:
        """Load data from pandas DataFrame"""
        examples = []
        
        if 'messages' in df.columns:
            # Direct messages column
            for _, row in df.iterrows():
                messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
                examples.append(ConversationExample(messages=messages))
        
        elif all(col in df.columns for col in ['user_message', 'assistant_message']):
            # User-assistant pairs
            for _, row in df.iterrows():
                messages = [
                    {"role": "user", "content": row['user_message']},
                    {"role": "assistant", "content": row['assistant_message']}
                ]
                if 'system_message' in df.columns and pd.notna(row['system_message']):
                    messages.insert(0, {"role": "system", "content": row['system_message']})
                
                examples.append(ConversationExample(messages=messages))
        
        else:
            raise ValueError("DataFrame must contain either 'messages' column or 'user_message'/'assistant_message' columns")
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return self.processor.tokenize_conversation(example)


class DataCollator:
    """Custom data collator for chat completion training"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Determine max length in batch
        max_length = max(len(f['input_ids'][0]) for f in features)
        
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch = {}
        for key in features[0].keys():
            if key in ['input_ids', 'labels']:
                # Pad sequences
                batch[key] = []
                for f in features:
                    seq = f[key][0]  # Remove batch dimension
                    padding_length = max_length - len(seq)
                    
                    if key == 'labels':
                        # Pad labels with -100 (ignore index)
                        padded = torch.cat([seq, torch.full((padding_length,), -100, dtype=seq.dtype)])
                    else:
                        # Pad input_ids with pad_token_id
                        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                        padded = torch.cat([seq, torch.full((padding_length,), pad_token_id, dtype=seq.dtype)])
                    
                    batch[key].append(padded)
                
                batch[key] = torch.stack(batch[key])
            
            elif key == 'attention_mask':
                # Create attention masks
                batch[key] = []
                for f in features:
                    seq_len = len(f['input_ids'][0])
                    mask = torch.cat([
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(max_length - seq_len, dtype=torch.long)
                    ])
                    batch[key].append(mask)
                
                batch[key] = torch.stack(batch[key])
        
        # Create attention mask if not present
        if 'attention_mask' not in batch:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            batch['attention_mask'] = (batch['input_ids'] != pad_token_id).long()
        
        return batch


def load_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizer:
    """Load tokenizer with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set special tokens if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    
    return tokenizer


def validate_dataset(dataset: ChatDataset, max_samples: int = 100) -> Dict[str, any]:
    """Validate dataset and return statistics"""
    stats = {
        "total_examples": len(dataset),
        "avg_length": 0,
        "max_length": 0,
        "min_length": float('inf'),
        "empty_examples": 0,
        "truncated_examples": 0
    }
    
    lengths = []
    max_length = dataset.processor.max_length
    
    for i in range(min(max_samples, len(dataset))):
        try:
            example = dataset[i]
            seq_len = len(example['input_ids'][0])
            lengths.append(seq_len)
            
            if seq_len == 0:
                stats["empty_examples"] += 1
            
            if seq_len >= max_length:
                stats["truncated_examples"] += 1
        
        except Exception as e:
            logger.warning(f"Error validating example {i}: {e}")
    
    if lengths:
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["max_length"] = max(lengths)
        stats["min_length"] = min(lengths)
    
    logger.info(f"Dataset validation stats: {stats}")
    return stats