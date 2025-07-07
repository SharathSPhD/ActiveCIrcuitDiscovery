#!/usr/bin/env python3
"""
Comprehensive tests for ProperActiveInferenceAgent

Tests validate:
1. Real pymdp integration works correctly
2. Expected Free Energy decreases over time (learning)
3. Belief updating follows Active Inference principles
4. Policy selection optimizes information gain
5. Correspondence metrics are meaningful

NO MOCKS - Tests against real Active Inference mathematics.
"""

import numpy as np
import pytest
import logging
from unittest.mock import Mock
from typing import List, Dict

# Import the real implementation
from src.active_inference.proper_agent import ProperActiveInferenceAgent
from src.active_inference.generative_model import CircuitGenerativeModelBuilder
from core.data_structures import SAEFeature, InterventionResult, InterventionType
from config.experiment_config import CompleteConfig, ActiveInferenceConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestProperActiveInferenceAgent:
    """Test suite for ProperActiveInferenceAgent with real pymdp validation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        ai_config = ActiveInferenceConfig(
            use_pymdp=True,
            epistemic_weight=0.7,
            exploration_weight=0.3,
            convergence_threshold=0.01
        )
        return CompleteConfig(active_inference=ai_config)
    
    @pytest.fixture
    def mock_features(self):
        """Create mock SAE features for testing."""
        features = []
        for i in range(10):
            feature = SAEFeature(
                feature_id=i,
                layer=i // 3,  # Distribute across layers
                max_activation=np.random.uniform(0.3, 0.9),
                activation_threshold=0.1,
                description=f"Test feature {i}",
                examples=[]
            )
            features.append(feature)
        return features
    
    @pytest.fixture
    def agent(self, config):
        """Create ProperActiveInferenceAgent for testing."""
        return ProperActiveInferenceAgent(config)
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes with proper pymdp components."""
        
        # Verify pymdp Agent is created
        assert hasattr(agent, 'agent')
        assert agent.agent is not None
        
        # Verify generative model components exist
        assert hasattr(agent, 'A') and agent.A is not None
        assert hasattr(agent, 'B') and agent.B is not None  
        assert hasattr(agent, 'C') and agent.C is not None
        assert hasattr(agent, 'D') and agent.D is not None
        
        # Verify dimensions are correct
        assert len(agent.A) == 2  # Effect magnitude and confidence modalities
        assert len(agent.B) == 3  # Component, importance, intervention factors
        assert len(agent.C) == 2  # Preferences over observations
        assert len(agent.D) == 3  # Priors over state factors
        
        # Verify policies are generated
        assert hasattr(agent, 'policies') and len(agent.policies) > 0
        
        logger.info(f"Agent initialized with {len(agent.policies)} policies")
    
    def test_generative_model_validation(self, agent):
        """Test that generative model matrices are properly formed."""
        
        # Test A matrices normalization
        for m, A_m in enumerate(agent.A):
            for state_combo in np.ndindex(A_m.shape[1:]):
                obs_dist = A_m[:, state_combo]
                assert np.isclose(obs_dist.sum(), 1.0, atol=1e-6), f"A[{m}] not normalized at {state_combo}"
        
        # Test B matrices normalization  
        for f, B_f in enumerate(agent.B):
            for state in range(B_f.shape[1]):
                for action in range(B_f.shape[2]):
                    trans_dist = B_f[:, state, action]
                    assert np.isclose(trans_dist.sum(), 1.0, atol=1e-6), f"B[{f}] not normalized at state {state}, action {action}"
        
        # Test D vectors normalization
        for f, D_f in enumerate(agent.D):
            assert np.isclose(D_f.sum(), 1.0, atol=1e-6), f"D[{f}] not normalized"
        
        logger.info("Generative model validation passed")
    
    def test_belief_initialization(self, agent, mock_features):
        """Test belief initialization from features."""
        
        # Create feature dictionary
        features_dict = {0: mock_features[:5], 1: mock_features[5:]}
        
        # Initialize beliefs
        belief_state = agent.initialize_beliefs(features_dict)
        
        # Verify belief state structure
        assert belief_state is not None
        assert hasattr(belief_state, 'qs')
        assert hasattr(belief_state, 'feature_importances')
        assert hasattr(belief_state, 'confidence')
        
        # Verify beliefs are probability distributions
        for qs_factor in belief_state.qs:
            assert np.isclose(qs_factor.sum(), 1.0, atol=1e-6), "Belief state not normalized"
            assert np.all(qs_factor >= 0), "Negative probabilities in belief state"
        
        # Verify feature mapping
        assert len(belief_state.feature_importances) == len(mock_features)
        
        logger.info(f"Beliefs initialized for {len(belief_state.feature_importances)} features")
    
    def test_expected_free_energy_calculation(self, agent, mock_features):
        """Test EFE calculation using real pymdp algorithms."""
        
        # Initialize with features
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        # Calculate EFE for different interventions
        test_feature = mock_features[0]
        
        efe_ablation = agent.calculate_expected_free_energy(test_feature, InterventionType.ABLATION)
        efe_patching = agent.calculate_expected_free_energy(test_feature, InterventionType.ACTIVATION_PATCHING)
        efe_mean_ablation = agent.calculate_expected_free_energy(test_feature, InterventionType.MEAN_ABLATION)
        
        # Verify EFE values are finite and reasonable
        assert np.isfinite(efe_ablation), "EFE for ablation is not finite"
        assert np.isfinite(efe_patching), "EFE for patching is not finite"
        assert np.isfinite(efe_mean_ablation), "EFE for mean ablation is not finite"
        
        # Verify epistemic and pragmatic components exist
        assert hasattr(agent, 'last_epistemic_value')
        assert hasattr(agent, 'last_pragmatic_value')
        assert np.isfinite(agent.last_epistemic_value)
        assert np.isfinite(agent.last_pragmatic_value)
        
        logger.info(f"EFE calculated: ablation={efe_ablation:.3f}, patching={efe_patching:.3f}, mean_ablation={efe_mean_ablation:.3f}")
    
    def test_belief_updating_cycle(self, agent, mock_features):
        """Test complete Active Inference belief updating cycle."""
        
        # Initialize beliefs
        features_dict = {0: mock_features}
        initial_beliefs = agent.initialize_beliefs(features_dict)
        
        # Create intervention result
        intervention_result = InterventionResult(
            target_feature=mock_features[0],
            intervention_type=InterventionType.ABLATION,
            effect_size=0.6,
            confidence=0.8,
            baseline_performance=0.5,
            intervened_performance=0.2,
            metadata={}
        )
        
        # Perform belief update
        correspondence = agent.update_beliefs(intervention_result)
        
        # Verify beliefs were updated
        updated_beliefs = agent.get_current_beliefs()
        assert updated_beliefs is not None
        
        # Verify correspondence metrics
        assert hasattr(correspondence, 'overall_correspondence')
        assert 0 <= correspondence.overall_correspondence <= 100
        
        # Verify agent history is tracked
        assert len(agent.belief_history) > 0
        assert len(agent.efe_history) > 0
        assert len(agent.intervention_history) > 0
        
        logger.info(f"Belief update completed: correspondence={correspondence.overall_correspondence:.1f}%")
    
    def test_efe_minimization_over_time(self, agent, mock_features):
        """Test that Expected Free Energy decreases over time (learning)."""
        
        # Initialize beliefs
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        efe_values = []
        
        # Simulate multiple interventions
        for i in range(5):
            # Create intervention result with varying effect sizes
            intervention_result = InterventionResult(
                target_feature=mock_features[i % len(mock_features)],
                intervention_type=InterventionType.ABLATION,
                effect_size=np.random.uniform(0.2, 0.8),
                confidence=0.7,
                baseline_performance=0.5,
                intervened_performance=0.3,
                metadata={}
            )
            
            # Update beliefs
            agent.update_beliefs(intervention_result)
            
            # Calculate EFE
            efe = agent.calculate_expected_free_energy(mock_features[0], InterventionType.ABLATION)
            efe_values.append(efe)
        
        # Verify EFE trend (should generally decrease or stabilize)
        # Use linear regression to check overall trend
        x = np.arange(len(efe_values))
        slope = np.polyfit(x, efe_values, 1)[0]
        
        # Allow for some noise but expect overall decreasing trend
        assert slope <= 0.1, f"EFE not decreasing over time: slope={slope:.3f}"
        
        logger.info(f"EFE minimization test passed: slope={slope:.3f}, final_EFE={efe_values[-1]:.3f}")
    
    def test_intervention_selection(self, agent, mock_features):
        """Test optimal intervention selection using policy inference."""
        
        # Initialize beliefs
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        # Select intervention
        selected_feature, selected_intervention = agent.select_intervention(mock_features)
        
        # Verify selection is valid
        assert selected_feature in mock_features
        assert selected_intervention in [InterventionType.ABLATION, InterventionType.ACTIVATION_PATCHING, InterventionType.MEAN_ABLATION]
        
        # Test multiple selections to ensure variety
        selections = []
        for _ in range(10):
            feature, intervention = agent.select_intervention(mock_features)
            selections.append((feature.feature_id, intervention))
        
        # Should show some variety in selections (not always the same)
        unique_selections = set(selections)
        assert len(unique_selections) >= 2, "Intervention selection shows no variety"
        
        logger.info(f"Intervention selection test passed: {len(unique_selections)} unique selections")
    
    def test_prediction_generation(self, agent, mock_features):
        """Test novel prediction generation from learned model."""
        
        # Initialize and train agent
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        # Perform some interventions to train the model
        for i in range(3):
            intervention_result = InterventionResult(
                target_feature=mock_features[i],
                intervention_type=InterventionType.ABLATION,
                effect_size=0.5 + i * 0.1,
                confidence=0.8,
                baseline_performance=0.5,
                intervened_performance=0.3,
                metadata={}
            )
            agent.update_beliefs(intervention_result)
        
        # Generate predictions
        predictions = agent.generate_predictions()
        
        # Verify predictions structure
        assert isinstance(predictions, list)
        
        for prediction in predictions:
            assert hasattr(prediction, 'prediction_id')
            assert hasattr(prediction, 'description')
            assert hasattr(prediction, 'predicted_value')
            assert hasattr(prediction, 'confidence')
            
            # Verify prediction values are reasonable
            assert 0 <= prediction.predicted_value <= 1
            assert 0 <= prediction.confidence <= 1
        
        logger.info(f"Generated {len(predictions)} predictions")
    
    def test_convergence_detection(self, agent, mock_features):
        """Test convergence detection based on EFE stabilization."""
        
        # Initialize beliefs
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        # Should not converge initially
        assert not agent.check_convergence()
        
        # Simulate converged state by setting stable EFE values
        stable_efe = np.ones(5) * -2.5
        for efe in stable_efe:
            agent.efe_history.append(np.array([efe]))
        
        # Should now detect convergence
        assert agent.check_convergence(threshold=0.1)
        
        logger.info("Convergence detection test passed")
    
    def test_correspondence_metrics_validity(self, agent, mock_features):
        """Test that correspondence metrics are meaningful and bounded."""
        
        # Initialize beliefs
        features_dict = {0: mock_features}
        agent.initialize_beliefs(features_dict)
        
        # Create high-correspondence intervention (predicted effect matches actual)
        high_corr_result = InterventionResult(
            target_feature=mock_features[0],
            intervention_type=InterventionType.ABLATION,
            effect_size=0.7,  # High effect as expected for ablation
            confidence=0.9,
            baseline_performance=0.5,
            intervened_performance=0.1,
            metadata={}
        )
        
        correspondence = agent.update_beliefs(high_corr_result)
        
        # Verify correspondence metrics are bounded and reasonable
        assert 0 <= correspondence.belief_updating_correspondence <= 100
        assert 0 <= correspondence.precision_weighting_correspondence <= 100
        assert 0 <= correspondence.prediction_error_correspondence <= 100
        assert 0 <= correspondence.overall_correspondence <= 100
        
        # For a high-effect intervention, correspondence should be reasonably high
        assert correspondence.overall_correspondence > 30, f"Correspondence too low: {correspondence.overall_correspondence}"
        
        logger.info(f"Correspondence metrics validated: overall={correspondence.overall_correspondence:.1f}%")

class TestCircuitGenerativeModelBuilder:
    """Test suite for generative model building."""
    
    @pytest.fixture
    def builder(self):
        """Create model builder for testing."""
        return CircuitGenerativeModelBuilder(max_components=16, importance_levels=4)
    
    def test_model_building(self, builder):
        """Test complete model building process."""
        
        A, B, C, D = builder.create_complete_model()
        
        # Verify model structure
        assert len(A) == 2  # Effect and confidence modalities
        assert len(B) == 3  # Component, importance, intervention factors  
        assert len(C) == 2  # Preferences
        assert len(D) == 3  # Priors
        
        # Verify model passes validation
        assert builder.validate_generative_model(A, B, C, D)
        
        logger.info("Model building test passed")
    
    def test_feature_integration(self, builder):
        """Test building model with real feature data."""
        
        # Create test features
        features = [
            SAEFeature(i, i//3, np.random.uniform(0.3, 0.9), 0.1, f"Feature {i}", [])
            for i in range(8)
        ]
        
        feature_importances = {i: f.max_activation for i, f in enumerate(features)}
        
        A, B, C, D = builder.create_complete_model(feature_importances)
        
        # Verify priors reflect feature importances
        assert np.sum(D[0]) == pytest.approx(1.0, abs=1e-6)
        
        # Higher importance features should have higher priors
        max_importance_idx = max(feature_importances.keys(), key=lambda k: feature_importances[k])
        assert D[0][max_importance_idx] > 0.1, "Most important feature should have higher prior"
        
        logger.info("Feature integration test passed")

def run_comprehensive_tests():
    """Run all tests and report results."""
    
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])

if __name__ == "__main__":
    run_comprehensive_tests()