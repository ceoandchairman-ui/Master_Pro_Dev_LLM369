"""
Basic unit tests for the chat completion model
"""

import unittest
import torch
from src.models.chat_completion_model import ChatCompletionModel, ChatCompletionConfig


class TestChatCompletionModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ChatCompletionConfig(
            vocab_size=1000,
            max_length=512,
            num_layers=4,
            num_heads=4,
            hidden_size=256,
            intermediate_size=1024
        )
        self.model = ChatCompletionModel(self.config)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.config.vocab_size, 1000)
        self.assertEqual(self.model.config.num_layers, 4)
        self.assertIsNotNone(self.model.token_embedding)
        self.assertIsNotNone(self.model.position_embedding)
    
    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        outputs = self.model(input_ids=input_ids)
        
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (batch_size, seq_len, self.config.vocab_size))
    
    def test_generation(self):
        """Test text generation"""
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False
            )
        
        self.assertEqual(outputs.shape[0], 1)
        self.assertGreater(outputs.shape[1], 5)


if __name__ == '__main__':
    unittest.main()