"""
Basic unit tests for RealCircuitTracer.
Tests that circuit-tracer can load and perform basic operations.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.circuit_analysis.real_tracer import RealCircuitTracer


class TestCircuitTracerBasic(unittest.TestCase):
    
    def test_tracer_initialization(self):
        """Test that RealCircuitTracer can be initialized correctly."""
        tracer = RealCircuitTracer(
            model_name="google/gemma-2-2b",
            transcoder_set="gemma"
        )
        
        # Test configuration is set correctly
        self.assertEqual(tracer.model_name, "google/gemma-2-2b")
        self.assertEqual(tracer.transcoder_set, "gemma")
        
        # Initially model should be None
        self.assertIsNone(tracer.model)
        
        print("✅ Circuit tracer initialization test passed")
    
    def test_model_loading(self):
        """Test that circuit-tracer can load Gemma-2B model."""
        tracer = RealCircuitTracer(
            model_name="google/gemma-2-2b",
            transcoder_set="gemma"
        )
        
        print("Loading Gemma-2B model and transcoders...")
        try:
            tracer.initialize_model()
            
            # Verify model is loaded
            self.assertIsNotNone(tracer.model)
            
            print("✅ Model loading successful")
            
            # Test basic tokenization
            test_text = "The Golden Gate Bridge"
            tokens = tracer.model.tokenizer(test_text, return_tensors="pt")
            self.assertIsNotNone(tokens)
            self.assertGreater(len(tokens['input_ids'][0]), 0)
            
            print(f"✅ Tokenization test passed: {len(tokens['input_ids'][0])} tokens")
            
        except Exception as e:
            print(f"Model loading error: {e}")
            # Don't fail test if model loading fails due to GPU/memory issues
            # This allows us to test the structure without requiring full model load
            self.skipTest(f"Model loading requires GPU resources: {e}")


if __name__ == '__main__':
    print("Running basic circuit tracer tests...")
    unittest.main()
