"""
Unit tests for data structures using real transcoder components only.
Tests CircuitFeature layer_idx consistency with pure transcoder approach.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.data_structures import CircuitFeature


class TestDataStructures(unittest.TestCase):
    
    def test_circuit_feature_layer_idx_consistency(self):
        """Test that CircuitFeature layer_idx attribute works correctly."""
        # Create a real CircuitFeature - now always transcoder
        feature = CircuitFeature(
            feature_id=123,
            layer_idx=5,  # Use layer_idx consistently
            activation_strength=0.8,
            description="Test semantic feature",
            max_activation=1.0,
            examples=["The Golden Gate Bridge"],
            component_type="attention",
            semantic_description="Golden Gate Bridge feature",
            intervention_sites=["blocks.5.mlp.transcoder.123"]
        )
        
        # Test layer_idx attribute works
        self.assertEqual(feature.layer_idx, 5)
        
        # Test unique_id uses layer_idx correctly with transcoder prefix (always T now)
        expected_id = "L5T123"  # Layer 5, Transcoder, feature 123
        self.assertEqual(feature.unique_id, expected_id)
        
        # Test validation passes
        self.assertGreaterEqual(feature.layer_idx, 0)
        self.assertGreaterEqual(feature.activation_strength, 0)
        self.assertGreaterEqual(feature.max_activation, 0)
        
        print(f"✅ Feature unique_id: {feature.unique_id}")
    
    def test_all_features_are_transcoder(self):
        """Test that all features are now transcoder-based (no SAE)."""
        feature = CircuitFeature(
            feature_id=456,
            layer_idx=3,
            activation_strength=0.6,
            description="Transcoder feature",
            max_activation=1.0,
            examples=["Test"]
        )
        
        # Should always use T prefix now (no more SAE)
        expected_id = "L3T456"
        self.assertEqual(feature.unique_id, expected_id)
        
        # Should always be transcoder source
        self.assertEqual(feature.feature_source, "transcoder")
        
        print(f"✅ All features are transcoder: {feature.unique_id}")
    
    def test_transcoder_feature_semantic_description(self):
        """Test transcoder features get proper semantic descriptions."""
        feature = CircuitFeature(
            feature_id=789,
            layer_idx=8,
            activation_strength=0.9,
            description="Transcoder feature",
            max_activation=1.0,
            examples=["Test"],
            component_type="mlp"
        )
        
        # Should auto-generate semantic description for transcoder
        expected_desc = "Transcoder L8F789 (mlp)"
        self.assertEqual(feature.semantic_description, expected_desc)
        
        print(f"✅ Transcoder semantic description: {feature.semantic_description}")
    
    def test_feature_activation_check(self):
        """Test feature activation threshold checking."""
        feature = CircuitFeature(
            feature_id=100,
            layer_idx=2,
            activation_strength=0.3,
            description="Low activation feature",
            max_activation=1.0,
            examples=["Test"]
        )
        
        # Test activation threshold
        self.assertTrue(feature.is_active(threshold=0.1))   # 0.3 > 0.1
        self.assertTrue(feature.is_active(threshold=0.3))   # 0.3 >= 0.3
        self.assertFalse(feature.is_active(threshold=0.5))  # 0.3 < 0.5
        
        print(f"✅ Feature activation checks passed")
    
    def test_data_validation(self):
        """Test proper error handling for invalid data."""
        # Test invalid layer_idx
        with self.assertRaises(ValueError):
            CircuitFeature(
                feature_id=123,
                layer_idx=-1,  # Invalid negative layer
                activation_strength=0.8,
                description="Test feature",
                max_activation=1.0,
                examples=["Test"]
            )
        
        # Test invalid activation strength
        with self.assertRaises(ValueError):
            CircuitFeature(
                feature_id=123,
                layer_idx=5,
                activation_strength=-0.1,  # Invalid negative activation
                description="Test feature",
                max_activation=1.0,
                examples=["Test"]
            )
        
        print("✅ Data validation tests passed")


if __name__ == '__main__':
    print("Running CircuitFeature transcoder-only tests...")
    unittest.main()
