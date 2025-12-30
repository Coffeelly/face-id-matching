# tests/test_basic.py
import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.face_engine import FaceVerifier

class TestSovereignID(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Runs ONCE before all tests. Loads the engine."""
        print("\n--- Setting up Test Suite ---")
        # cls.engine = FaceVerifier(use_quantized=False) # Test standard model
        cls.engine = FaceVerifier(use_quantized=True) # Test optimized model

    def test_engine_initialization(self):
        """Does the engine load without crashing?"""
        self.assertIsNotNone(self.engine.resnet)
        print("Engine Load Test Passed")

    def test_tensor_shape(self):
        """Does the AI output the correct 512-dim embedding?"""
        # Create a fake face tensor [3, 160, 160]
        fake_face = torch.randn(3, 160, 160)
        
        with torch.no_grad():
            embedding = self.engine.resnet(fake_face.unsqueeze(0).to(self.engine.device))
        
        # Expecting shape [1, 512]
        self.assertEqual(embedding.shape[1], 512)
        print("Output Shape Test Passed (512-dim)")

    def test_similarity_logic(self):
        """Does comparing the SAME face give a high score?"""
        fake_face = torch.randn(3, 160, 160)
        score, _ = self.engine.verify(fake_face, fake_face)
        
        # Identical noise should match 100% (approx 1.0)
        self.assertTrue(score > 0.99)
        print("Logic Test Passed (Self-Match > 0.99)")

if __name__ == '__main__':
    unittest.main()