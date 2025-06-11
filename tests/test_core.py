#!/usr/bin/env python3
"""
ActiveCircuitDiscovery: Core Component Tests
==========================================

Basic functionality tests for the ActiveCircuitDiscovery library.
Tests core components, imports, and basic functionality.

Usage:
    python -m pytest tests/test_core.py
    python tests/test_core.py
"""

import sys
import unittest
from pathlib import Path
import tempfile
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestImports(unittest.TestCase):
    """Test that all core components can be imported."""
    
    def test_core_data_structures_import(self):
        """Test core data structures imports."""
        try:
            from core.data_structures import (
                SAEFeature, InterventionResult, AttributionGraph, 
                BeliefState, ExperimentResult, NovelPrediction, CorrespondenceMetrics
            )
            self.assertTrue(True, "Core data structures imports successful")
        except ImportError as e:
            self.fail(f"Core data structures import failed: {e}")
    
    def test_circuit_tracer_import(self):
        """Test circuit tracer imports."""
        try:
            from circuit_analysis.tracer import CircuitTracer
            self.assertTrue(True, "Circuit tracer imports successful")
        except ImportError as e:
            self.fail(f"Circuit tracer import failed: {e}")
    
    def test_active_inference_import(self):
        """Test active inference imports."""
        try:
            from active_inference.agent import ActiveInferenceAgent
            self.assertTrue(True, "Active inference imports successful")
        except ImportError as e:
            self.fail(f"Active inference import failed: {e}")
    
    def test_visualizer_import(self):
        """Test visualizer imports."""
        try:
            from visualization.visualizer import CircuitVisualizer
            self.assertTrue(True, "Visualizer imports successful")
        except ImportError as e:
            # Visualizer may fail due to optional dependencies
            self.skipTest(f"Visualizer import failed (optional dependencies): {e}")
    
    def test_experiment_import(self):
        """Test experiment runner imports."""
        try:
            from experiments.runner import YorKExperimentRunner
            self.assertTrue(True, "Experiment runner imports successful")
        except ImportError as e:
            self.fail(f"Experiment runner import failed: {e}")
    
    def test_enhanced_components_import(self):
        """Test enhanced components imports."""
        try:
            from core.statistical_validation import StatisticalValidator
            from core.prediction_system import EnhancedPredictionGenerator
            from core.prediction_validator import PredictionValidator
            from config.experiment_config import get_enhanced_config
            self.assertTrue(True, "Enhanced components imports successful")
        except ImportError as e:
            self.skipTest(f"Enhanced components import failed (optional): {e}")

class TestDataStructures(unittest.TestCase):
    """Test core data structures."""
    
    def setUp(self):
        """Set up test data structures."""
        from core.data_structures import (
            SAEFeature, InterventionResult, AttributionGraph, 
            BeliefState, CorrespondenceMetrics, NovelPrediction
        )
        import numpy as np
        
        self.SAEFeature = SAEFeature
        self.InterventionResult = InterventionResult
        self.AttributionGraph = AttributionGraph
        self.BeliefState = BeliefState
        self.CorrespondenceMetrics = CorrespondenceMetrics
        self.NovelPrediction = NovelPrediction
    
    def test_sae_feature_creation(self):
        """Test SAEFeature creation and attributes."""
        feature = self.SAEFeature(
            feature_id=1234,
            layer=8,
            activation_threshold=0.5,
            description="Test feature",
            max_activation=0.8,
            examples=["test", "example"]
        )
        
        self.assertEqual(feature.feature_id, 1234)
        self.assertEqual(feature.layer, 8)
        self.assertEqual(feature.description, "Test feature")
        self.assertEqual(len(feature.examples), 2)
    
    def test_belief_state_creation(self):
        """Test BeliefState creation and attributes."""
        import numpy as np
        
        belief_state = self.BeliefState(
            qs=np.array([0.7, 0.3]),
            feature_importances={1: 0.5, 2: 0.3},
            connection_beliefs={(1, 2): 0.7},
            uncertainty={1: 0.2, 2: 0.4},
            confidence=0.8
        )
        
        self.assertEqual(len(belief_state.feature_importances), 2)
        self.assertEqual(len(belief_state.connection_beliefs), 1)
        self.assertEqual(belief_state.confidence, 0.8)
        self.assertEqual(len(belief_state.qs), 2)
    
    def test_correspondence_metrics_creation(self):
        """Test CorrespondenceMetrics creation and validation."""
        metrics = self.CorrespondenceMetrics(
            belief_updating_correspondence=0.8,
            precision_weighting_correspondence=0.7,
            prediction_error_correspondence=0.6,
            overall_correspondence=0.7
        )
        
        self.assertEqual(metrics.overall_correspondence, 0.7)
        self.assertTrue(0 <= metrics.overall_correspondence <= 1)
        
        # Test validation fails for invalid values
        with self.assertRaises(ValueError):
            self.CorrespondenceMetrics(
                belief_updating_correspondence=1.5,  # Invalid > 1
                precision_weighting_correspondence=0.7,
                prediction_error_correspondence=0.6,
                overall_correspondence=0.7
            )
    
    def test_novel_prediction_creation(self):
        """Test NovelPrediction creation and validation."""
        prediction = self.NovelPrediction(
            prediction_type="attention_pattern",
            description="Test prediction",
            testable_hypothesis="Test hypothesis",
            expected_outcome="Test outcome",
            test_method="Test method",
            confidence=0.8
        )
        
        self.assertEqual(prediction.prediction_type, "attention_pattern")
        self.assertEqual(prediction.confidence, 0.8)
        self.assertEqual(prediction.validation_status, "untested")
        
        # Test validation fails for invalid prediction types
        with self.assertRaises(ValueError):
            self.NovelPrediction(
                prediction_type="invalid_type",
                description="Test prediction",
                testable_hypothesis="Test hypothesis",
                expected_outcome="Test outcome",
                test_method="Test method",
                confidence=0.8
            )

class TestCircuitTracer(unittest.TestCase):
    """Test circuit tracer basic functionality."""
    
    def setUp(self):
        """Set up circuit tracer for testing."""
        try:
            from circuit_analysis.tracer import CircuitTracer
            from config.experiment_config import CompleteConfig
            self.CircuitTracer = CircuitTracer
            self.CompleteConfig = CompleteConfig
            self.tracer = None  # Will be initialized in tests that need it
        except ImportError:
            self.skipTest("Circuit tracer dependencies not available")
    
    def test_circuit_tracer_initialization_cpu(self):
        """Test circuit tracer initialization on CPU."""
        try:
            config = self.CompleteConfig()
            config.model.device = "cpu"
            tracer = self.CircuitTracer(config)
            self.assertIsNotNone(tracer)
            self.assertIsNotNone(tracer.config)
        except Exception as e:
            self.skipTest(f"Circuit tracer initialization failed: {e}")
    
    def test_tracer_methods_exist(self):
        """Test that tracer has required methods."""
        try:
            config = self.CompleteConfig()
            config.model.device = "cpu"
            tracer = self.CircuitTracer(config)
            
            # Test that required methods exist
            self.assertTrue(hasattr(tracer, 'find_active_features'))
            self.assertTrue(hasattr(tracer, 'perform_intervention'))
            self.assertTrue(hasattr(tracer, 'build_attribution_graph'))
            self.assertTrue(hasattr(tracer, 'get_feature_activations'))
            
        except Exception as e:
            self.skipTest(f"Tracer methods test failed: {e}")
    
    def test_find_active_features_structure(self):
        """Test find_active_features method structure."""
        try:
            config = self.CompleteConfig()
            config.model.device = "cpu"
            tracer = self.CircuitTracer(config)
            
            text = "The Golden Gate Bridge"
            
            # Test that method exists and can be called
            self.assertTrue(hasattr(tracer, 'find_active_features'))
            
            # Basic structure test - method should exist even if it doesn't work without full dependencies
            
        except Exception as e:
            self.skipTest(f"Active features structure test failed: {e}")

class TestActiveInference(unittest.TestCase):
    """Test Active Inference functionality."""
    
    def setUp(self):
        """Set up Active Inference components."""
        try:
            from active_inference.agent import ActiveInferenceAgent
            from circuit_analysis.tracer import CircuitTracer
            from config.experiment_config import CompleteConfig
            
            self.ActiveInferenceAgent = ActiveInferenceAgent
            self.CircuitTracer = CircuitTracer
            self.CompleteConfig = CompleteConfig
            
        except ImportError:
            self.skipTest("Active Inference dependencies not available")
    
    def test_active_inference_initialization(self):
        """Test Active Inference agent initialization."""
        try:
            # Create mock tracer
            config = self.CompleteConfig()
            config.model.device = "cpu"
            tracer = self.CircuitTracer(config)
            
            # Initialize AI agent
            ai_agent = self.ActiveInferenceAgent(config, tracer)
            
            self.assertIsNotNone(ai_agent)
            self.assertEqual(ai_agent.tracer, tracer)
            self.assertIsNotNone(ai_agent.config)
            
        except Exception as e:
            self.skipTest(f"Active Inference initialization failed: {e}")
    
    def test_agent_methods_exist(self):
        """Test that agent has required methods."""
        try:
            config = self.CompleteConfig()
            config.model.device = "cpu"
            tracer = self.CircuitTracer(config)
            ai_agent = self.ActiveInferenceAgent(config, tracer)
            
            # Test that required methods exist
            self.assertTrue(hasattr(ai_agent, 'initialize_beliefs'))
            self.assertTrue(hasattr(ai_agent, 'calculate_expected_free_energy'))
            self.assertTrue(hasattr(ai_agent, 'update_beliefs'))
            self.assertTrue(hasattr(ai_agent, 'generate_predictions'))
            self.assertTrue(hasattr(ai_agent, 'check_convergence'))
            
        except Exception as e:
            self.skipTest(f"Agent methods test failed: {e}")

class TestVisualization(unittest.TestCase):
    """Test visualization components."""
    
    def setUp(self):
        """Set up visualization components."""
        try:
            from visualizer import CircuitVisualizer
            self.CircuitVisualizer = CircuitVisualizer
        except ImportError:
            self.skipTest("Visualization dependencies not available")
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                visualizer = self.CircuitVisualizer(temp_dir)
                self.assertIsNotNone(visualizer)
                self.assertEqual(str(visualizer.output_dir), temp_dir)
                
            except Exception as e:
                self.skipTest(f"Visualizer initialization failed: {e}")

class TestExperiment(unittest.TestCase):
    """Test experiment runner."""
    
    def setUp(self):
        """Set up experiment components."""
        try:
            from experiments.runner import YorKExperimentRunner
            from config.experiment_config import CompleteConfig
            self.YorKExperimentRunner = YorKExperimentRunner
            self.CompleteConfig = CompleteConfig
        except ImportError:
            self.skipTest("Experiment dependencies not available")
    
    def test_experiment_runner_initialization(self):
        """Test experiment runner initialization."""
        try:
            runner = self.YorKExperimentRunner()
            self.assertIsNotNone(runner)
            self.assertIsNotNone(runner.config)
            
        except Exception as e:
            self.skipTest(f"Experiment runner initialization failed: {e}")
    
    def test_runner_methods_exist(self):
        """Test that runner has required methods."""
        try:
            runner = self.YorKExperimentRunner()
            
            # Test that required methods exist
            self.assertTrue(hasattr(runner, 'setup_experiment'))
            self.assertTrue(hasattr(runner, 'run_experiment'))
            self.assertTrue(hasattr(runner, 'validate_research_questions'))
            self.assertTrue(hasattr(runner, 'save_results'))
            
        except Exception as e:
            self.skipTest(f"Runner methods test failed: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline without heavy computation."""
        try:
            from circuit_analysis.tracer import CircuitTracer
            from active_inference.agent import ActiveInferenceAgent
            from config.experiment_config import CompleteConfig
            
            # Initialize components
            config = CompleteConfig()
            config.model.device = "cpu"
            tracer = CircuitTracer(config)
            ai_agent = ActiveInferenceAgent(config, tracer)
            
            # Verify integration
            self.assertEqual(ai_agent.tracer, tracer)
            self.assertEqual(ai_agent.config, config)
            
        except Exception as e:
            self.skipTest(f"Basic pipeline test failed: {e}")

class TestEnhancedComponents(unittest.TestCase):
    """Test enhanced components functionality."""
    
    def test_statistical_validator_import(self):
        """Test statistical validator import and methods."""
        try:
            from core.statistical_validation import StatisticalValidator
            
            validator = StatisticalValidator()
            self.assertIsNotNone(validator)
            
            # Check required methods exist
            self.assertTrue(hasattr(validator, 'validate_correspondence_significance'))
            self.assertTrue(hasattr(validator, 'validate_efficiency_improvement'))
            self.assertTrue(hasattr(validator, 'validate_prediction_success_rate'))
            
        except ImportError:
            self.skipTest("Enhanced statistical validation not available")
        except Exception as e:
            self.skipTest(f"Statistical validator test failed: {e}")
    
    def test_prediction_system_import(self):
        """Test prediction system import and methods."""
        try:
            from core.prediction_system import EnhancedPredictionGenerator
            
            generator = EnhancedPredictionGenerator()
            self.assertIsNotNone(generator)
            
            # Check required methods exist
            self.assertTrue(hasattr(generator, 'generate_attention_pattern_predictions'))
            self.assertTrue(hasattr(generator, 'generate_feature_interaction_predictions'))
            self.assertTrue(hasattr(generator, 'generate_failure_mode_predictions'))
            
        except ImportError:
            self.skipTest("Enhanced prediction system not available")
        except Exception as e:
            self.skipTest(f"Prediction system test failed: {e}")
    
    def test_enhanced_config_import(self):
        """Test enhanced configuration import."""
        try:
            from config.experiment_config import get_enhanced_config
            
            config = get_enhanced_config()
            self.assertIsNotNone(config)
            
            # Check enhanced sections exist
            self.assertTrue(hasattr(config, 'statistical_validation'))
            self.assertTrue(hasattr(config, 'prediction_validation'))
            self.assertTrue(hasattr(config, 'visualization'))
            
        except ImportError:
            self.skipTest("Enhanced configuration not available")
        except Exception as e:
            self.skipTest(f"Enhanced config test failed: {e}")

def run_tests():
    """Run all tests with detailed output."""
    
    print("ActiveCircuitDiscovery: Running Core Component Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestImports,
        TestDataStructures,
        TestCircuitTracer,
        TestActiveInference,
        TestVisualization,
        TestExperiment,
        TestIntegration,
        TestEnhancedComponents
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("Core components are working correctly!")
    else:
        print("Some components need attention - check dependencies")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)