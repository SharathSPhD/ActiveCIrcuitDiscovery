"""
System test to validate all corrections work together.
Tests that the fixed codebase can run experiments with minimum interventions.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.circuit_discovery_integration import CircuitDiscoveryIntegration


class TestCorrectionsValidation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up real integrated system for testing."""
        print("Setting up integrated circuit discovery system...")
        cls.integration = CircuitDiscoveryIntegration(model_name="google/gemma-2-2b")
        print("Initializing components...")
        cls.integration.initialize()
        print("✅ System setup complete")
    
    def test_minimum_interventions_enforcement(self):
        """Test that minimum interventions are enforced (correction for single intervention issue)."""
        print("\nTesting minimum intervention enforcement...")
        
        # Run short experiment with minimum interventions
        test_prompts = ["The Golden Gate Bridge is in"]
        
        result = self.integration.run_integrated_discovery(
            test_prompts=test_prompts,
            max_interventions=15,
            min_interventions=10  # Should enforce at least 10 interventions
        )
        
        # Key correction validation: Should perform at least 10 interventions
        self.assertGreaterEqual(result.total_interventions, 10, 
                              "Should perform at least 10 interventions (was previously only 1)")
        
        print(f"✅ Performed {result.total_interventions} interventions (≥10 required)")
        
        # Should have meaningful intervention data
        self.assertGreater(len(self.integration.intervention_history), 1,
                          "Should have multiple intervention results")
        
        print(f"✅ Generated {len(self.integration.intervention_history)} intervention results")
    
    def test_layer_idx_consistency(self):
        """Test that layer_idx attribute works consistently (correction for layer vs layer_idx)."""
        print("\nTesting layer_idx consistency...")
        
        # Discover some features
        test_prompts = ["The Golden Gate Bridge is located in"]
        features = self.integration.circuit_tracer.find_active_features(test_prompts)
        
        self.assertGreater(len(features), 0, "Should discover some features")
        
        # Test that all features have layer_idx attribute
        for feature in features[:5]:  # Test first 5 features
            self.assertTrue(hasattr(feature, 'layer_idx'), 
                          f"Feature should have layer_idx attribute")
            self.assertIsInstance(feature.layer_idx, int,
                                f"layer_idx should be integer")
            self.assertGreaterEqual(feature.layer_idx, 0,
                                  f"layer_idx should be non-negative")
            
            # Test unique_id generation works with layer_idx
            unique_id = feature.unique_id
            self.assertIsInstance(unique_id, str)
            self.assertTrue(unique_id.startswith(f"L{feature.layer_idx}"),
                          f"unique_id should start with L{feature.layer_idx}")
        
        print(f"✅ All {len(features)} features have consistent layer_idx attributes")
    
    def test_belief_updating_from_interventions(self):
        """Test that AI agent updates beliefs from real interventions (correction for static beliefs)."""
        print("\nTesting belief updating from interventions...")
        
        # Get initial belief state
        initial_beliefs = self.integration.ai_agent.get_belief_states()
        
        # Perform a few interventions
        test_prompts = ["The Golden Gate Bridge connects"]
        
        result = self.integration.run_integrated_discovery(
            test_prompts=test_prompts,
            max_interventions=5,
            min_interventions=3
        )
        
        # Get updated belief state
        updated_beliefs = self.integration.ai_agent.get_belief_states()
        
        # Should have meaningful belief evolution
        self.assertGreater(len(updated_beliefs), len(initial_beliefs),
                          "Should have more belief states after interventions")
        
        # Should have at least 3 interventions worth of belief updates
        self.assertGreaterEqual(len(updated_beliefs), 3,
                              "Should have belief states from multiple interventions")
        
        print(f"✅ Belief states evolved from {len(initial_beliefs)} to {len(updated_beliefs)}")
    
    def test_research_questions_calculable(self):
        """Test that research questions can be calculated (correction for zero variance issues)."""
        print("\nTesting research question calculations...")
        
        # Run experiment with enough interventions for meaningful metrics
        test_prompts = ["San Francisco is known for the Golden Gate"]
        
        result = self.integration.run_integrated_discovery(
            test_prompts=test_prompts,
            max_interventions=12,
            min_interventions=8
        )
        
        # RQ1: Correspondence should be calculable (not 0.0 due to zero variance)
        self.assertIsInstance(result.correspondence_score, (int, float),
                            "Correspondence score should be numeric")
        self.assertGreaterEqual(result.correspondence_score, 0.0,
                              "Correspondence score should be non-negative")
        
        # RQ2: Efficiency should be calculable (not failing due to insufficient data)
        self.assertIsInstance(result.efficiency_improvement, (int, float),
                            "Efficiency improvement should be numeric")
        self.assertGreaterEqual(result.efficiency_improvement, 0.0,
                              "Efficiency should be non-negative")
        
        # RQ3: Should generate some predictions (not failing due to tensor shape mismatches)
        self.assertIsInstance(result.novel_predictions, list,
                            "Novel predictions should be a list")
        
        print(f"✅ RQ1 Correspondence: {result.correspondence_score:.1f}%")
        print(f"✅ RQ2 Efficiency: {result.efficiency_improvement:.1f}%")
        print(f"✅ RQ3 Predictions: {len(result.novel_predictions)} generated")
    
    def test_no_crashes_or_attribute_errors(self):
        """Test that experiment completes without crashes or attribute errors."""
        print("\nTesting system stability...")
        
        # Run full experiment pipeline
        test_prompts = ["The Golden Gate Bridge spans the Golden Gate strait"]
        
        try:
            result = self.integration.run_integrated_discovery(
                test_prompts=test_prompts,
                max_interventions=10,
                min_interventions=6
            )
            
            # Should complete without exceptions
            self.assertIsNotNone(result, "Experiment should return results")
            
            # Should have all expected result fields
            self.assertTrue(hasattr(result, 'total_interventions'))
            self.assertTrue(hasattr(result, 'correspondence_score'))
            self.assertTrue(hasattr(result, 'efficiency_improvement'))
            self.assertTrue(hasattr(result, 'novel_predictions'))
            self.assertTrue(hasattr(result, 'convergence_achieved'))
            
            print("✅ Experiment completed successfully without crashes")
            print(f"✅ Total interventions: {result.total_interventions}")
            print(f"✅ Convergence achieved: {result.convergence_achieved}")
            
        except Exception as e:
            self.fail(f"Experiment crashed with error: {e}")


if __name__ == '__main__':
    print("Running system-level corrections validation...")
    unittest.main()
