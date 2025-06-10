import pytest
import torch
import numpy as np
from unittest import mock
import logging # Import the logging module

# Adjust path to import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.experiment_config import CompleteConfig, ActiveInferenceConfig, ModelConfig, DeviceType, InterventionType
from src.active_inference.agent import ActiveInferenceAgent
from src.core.data_structures import SAEFeature, InterventionResult, BeliefState, CorrespondenceMetrics

# Default minimal config for agent tests
@pytest.fixture
def agent_config():
    config = CompleteConfig()
    config.model = ModelConfig(name="gpt2-test", device=DeviceType.CPU)
    config.active_inference = ActiveInferenceConfig(
        epistemic_weight=0.5, # Example value
        convergence_threshold=0.1
    )
    return config

@pytest.fixture
def mock_tracer():
    return mock.MagicMock()

@pytest.fixture
def agent_instance(agent_config, mock_tracer):
    # Temporarily disable pymdp for these specific tests if it's not the focus
    agent_config.active_inference.use_pymdp = False
    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)
    return agent

@pytest.fixture
def sample_sae_feature():
    return SAEFeature(
        feature_id=101,
        layer=3,
        activation_threshold=0.05,
        description="Test SAE Feature 101",
        max_activation=0.8,
        examples=["example text"],
        feature_vector=np.random.rand(768),
        decoder_weights=np.random.rand(768)
    )

@pytest.fixture
def sample_intervention_result(sample_sae_feature):
    return InterventionResult(
        intervention_type=InterventionType.ABLATION,
        target_feature=sample_sae_feature,
        original_logits=torch.randn(1, 10, 50257), # vocab_size for gpt2
        intervened_logits=torch.randn(1, 10, 50257),
        effect_size=0.5, # Positive effect
        target_token_change=0.2,
        intervention_layer=sample_sae_feature.layer
    )

def test_update_beliefs_initialization_check(agent_instance, sample_intervention_result, caplog):
    """Test that update_beliefs handles uninitialized belief_state."""
    agent_instance.belief_state = None # Ensure it's not initialized
    metrics = agent_instance.update_beliefs(sample_intervention_result)

    assert "Cannot update beliefs: BeliefState is not initialized." in caplog.text
    assert metrics.overall_correspondence == 0.0 # Should return default/empty metrics

def test_update_beliefs_updates_feature_metrics(agent_instance, sample_sae_feature, sample_intervention_result):
    """Test updates to feature_importances and uncertainty."""
    # Initialize a basic belief state
    feature_id = sample_sae_feature.feature_id
    initial_importance = 0.5
    initial_uncertainty = 0.6

    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]), # Dummy
        feature_importances={feature_id: initial_importance},
        connection_beliefs={},
        uncertainty={feature_id: initial_uncertainty},
        confidence=1.0 - initial_uncertainty
    )

    agent_instance.update_beliefs(sample_intervention_result)

    alpha = agent_instance.config.active_inference.epistemic_weight # Use configured alpha
    effect_size = sample_intervention_result.effect_size

    # expected_new_importance = initial_importance + effect_size * alpha
    expected_new_importance = np.clip(initial_importance + effect_size * alpha, 0.0, 1.0)
    # expected_new_uncertainty = initial_uncertainty * (1 - abs(effect_size) * alpha)
    expected_new_uncertainty = np.clip(initial_uncertainty * (1.0 - abs(effect_size) * alpha), 0.01, 1.0)

    assert agent_instance.belief_state.feature_importances[feature_id] == pytest.approx(expected_new_importance)
    assert agent_instance.belief_state.uncertainty[feature_id] == pytest.approx(expected_new_uncertainty)

def test_update_beliefs_updates_overall_confidence(agent_instance, sample_sae_feature, sample_intervention_result):
    """Test that overall confidence is updated."""
    feature_id = sample_sae_feature.feature_id
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]),
        feature_importances={feature_id: 0.5, 999: 0.3}, # Add another feature
        connection_beliefs={},
        uncertainty={feature_id: 0.6, 999: 0.4},
        confidence=0.5 # Initial dummy
    )

    initial_avg_uncertainty = agent_instance.belief_state.get_average_uncertainty()
    agent_instance.update_beliefs(sample_intervention_result)

    new_avg_uncertainty = agent_instance.belief_state.get_average_uncertainty()
    expected_new_confidence = np.clip(1.0 - new_avg_uncertainty, 0.0, 1.0)

    assert new_avg_uncertainty != initial_avg_uncertainty # Ensure uncertainty changed
    assert agent_instance.belief_state.confidence == pytest.approx(expected_new_confidence)

def test_update_beliefs_history_tracking(agent_instance, sample_intervention_result):
    """Test that intervention and correspondence history are tracked."""
    agent_instance.belief_state = BeliefState( # Basic init
        qs=np.array([1.0]), feature_importances={}, connection_beliefs={}, uncertainty={}, confidence=0.5
    )

    assert len(agent_instance.intervention_history) == 0
    assert len(agent_instance.correspondence_history) == 0

    agent_instance.update_beliefs(sample_intervention_result)

    assert len(agent_instance.intervention_history) == 1
    assert agent_instance.intervention_history[0] == sample_intervention_result
    assert len(agent_instance.correspondence_history) == 1
    assert isinstance(agent_instance.correspondence_history[0], CorrespondenceMetrics)

def test_update_beliefs_returns_correspondence_metrics(agent_instance, sample_intervention_result):
    """Test the structure and basic calculation of returned CorrespondenceMetrics."""
    agent_instance.belief_state = BeliefState( # Basic init
        qs=np.array([1.0]), feature_importances={}, connection_beliefs={}, uncertainty={}, confidence=0.5
    )

    metrics = agent_instance.update_beliefs(sample_intervention_result)

    effect_size = sample_intervention_result.effect_size # 0.5

    # belief_updating_corr = min(1.0, abs(effect_size * 2.0)) = min(1.0, abs(0.5*2.0)) = 1.0
    expected_belief_corr = np.clip(abs(effect_size * 2.0), 0.0, 1.0)
    # prediction_error_corr = min(1.0, abs(effect_size * 1.5)) = min(1.0, abs(0.5*1.5)) = 0.75
    expected_pred_err_corr = np.clip(abs(effect_size * 1.5), 0.0, 1.0)
    # precision_weighting_corr = 0.5 (placeholder)
    expected_prec_corr = 0.5

    expected_overall_corr = (expected_belief_corr + expected_pred_err_corr + expected_prec_corr) / 3.0
    expected_overall_corr = np.clip(expected_overall_corr, 0.0, 1.0)

    assert isinstance(metrics, CorrespondenceMetrics)
    assert metrics.belief_updating_correspondence == pytest.approx(expected_belief_corr)
    assert metrics.precision_weighting_correspondence == pytest.approx(expected_prec_corr)
    assert metrics.prediction_error_correspondence == pytest.approx(expected_pred_err_corr)
    assert metrics.overall_correspondence == pytest.approx(expected_overall_corr)

def test_update_beliefs_clips_importance_and_uncertainty(agent_instance, sample_sae_feature):
    """Test that importance and uncertainty are clipped to valid ranges [0,1] and [0.01,1] respectively."""
    feature_id = sample_sae_feature.feature_id
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]),
        feature_importances={feature_id: 0.99}, # High initial importance
        connection_beliefs={},
        uncertainty={feature_id: 0.02}, # Low initial uncertainty
        confidence=0.9
    )

    # Intervention with large positive effect size
    large_effect_intervention = InterventionResult(
        intervention_type=InterventionType.ABLATION, target_feature=sample_sae_feature,
        original_logits=torch.randn(1,1,1), intervened_logits=torch.randn(1,1,1),
        effect_size=0.5, target_token_change=0.1, intervention_layer=sample_sae_feature.layer
    )
    agent_instance.update_beliefs(large_effect_intervention)
    assert 0.0 <= agent_instance.belief_state.feature_importances[feature_id] <= 1.0
    assert 0.01 <= agent_instance.belief_state.uncertainty[feature_id] <= 1.0

# --- Tests for initialize_beliefs ---

@pytest.fixture
def multi_sae_features_dict():
    # d_model could be part of a shared fixture or config
    d_model = 768
    features = {
        0: [ # Layer 0
            SAEFeature(feature_id=1, layer=0, activation_threshold=0.1, description="L0F1", max_activation=0.9, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model)),
            SAEFeature(feature_id=2, layer=0, activation_threshold=0.1, description="L0F2", max_activation=0.7, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model))
        ],
        1: [ # Layer 1
            SAEFeature(feature_id=3, layer=1, activation_threshold=0.1, description="L1F3", max_activation=0.95, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model))
        ]
    }
    return features

def test_initialize_beliefs_no_features(agent_instance, caplog):
    """Tests initialize_beliefs with an empty feature dictionary."""
    logging.getLogger("src.active_inference.agent").propagate = True # Ensure logs are captured
    agent_instance.initialize_beliefs({}) # Empty features

    assert agent_instance.belief_state is not None
    # Based on _create_empty_belief_state: qs is empty, dicts are empty, confidence is 0.0
    assert len(agent_instance.belief_state.qs) == 0
    assert not agent_instance.belief_state.feature_importances
    assert not agent_instance.belief_state.uncertainty
    assert agent_instance.belief_state.confidence == 0.0
    assert "No features found for belief initialization" in caplog.text
    logging.getLogger("src.active_inference.agent").propagate = False # Reset


def test_initialize_beliefs_with_features(agent_instance, multi_sae_features_dict):
    """Tests initialize_beliefs with a sample set of features."""
    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())

    with mock.patch.object(agent_instance, '_initialize_connection_beliefs', return_value={}) as mock_init_conn, \
         mock.patch.object(agent_instance, '_calculate_initial_confidence') as mock_calc_confidence:
        # Set a known return value for confidence to simplify assertion if its internal logic is complex
        mock_calc_confidence.return_value = 0.75
        belief_state = agent_instance.initialize_beliefs(multi_sae_features_dict)

    assert belief_state is not None
    assert len(belief_state.qs) == num_total_features
    if num_total_features > 0:
        assert np.allclose(belief_state.qs, np.ones(num_total_features) / num_total_features) # Uniform distribution

    for layer_idx, sae_features_in_layer in multi_sae_features_dict.items():
        for sae_feature in sae_features_in_layer:
            fid = sae_feature.feature_id
            assert fid in belief_state.feature_importances
            assert belief_state.feature_importances[fid] == sae_feature.max_activation
            assert fid in belief_state.uncertainty
            # Check uncertainty calculation: clip(1.0 - max_activation, 0.01, 1.0)
            expected_uncertainty_val = np.clip(1.0 - sae_feature.max_activation, 0.01, 1.0)
            assert belief_state.uncertainty[fid] == pytest.approx(expected_uncertainty_val)

    mock_calc_confidence.assert_called_once() # Check it was called
    assert belief_state.confidence == 0.75 # Check it used the mocked value

    mock_init_conn.assert_called_once()


def test_initialize_beliefs_pymdp_logic_success(agent_config, mock_tracer, multi_sae_features_dict, monkeypatch):
    """Tests initialize_beliefs when use_pymdp is True and component init succeeds."""
    agent_config.active_inference.use_pymdp = True
    # Create agent first
    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)

    # If pymdp wasn't actually available during __init__, self.use_pymdp would be False.
    # For this test, we want to simulate it being available AND configured.
    monkeypatch.setattr(agent, 'pymdp_is_available', True)
    agent.use_pymdp = agent.config.active_inference.use_pymdp and agent.pymdp_is_available # Re-apply init logic for use_pymdp

    assert agent.use_pymdp is True # Pre-condition for the test's purpose

    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())
    dummy_gm = {"A": np.array([[0.5]*num_total_features]*3)} # Adjusted shape for num_total_features
    dummy_qs_correct_shape = np.ones(num_total_features) / num_total_features if num_total_features > 0 else np.array([])
    dummy_pm = np.eye(num_total_features) if num_total_features > 0 else np.array([[]])


    with mock.patch.object(agent, '_initialize_pymdp_components', return_value=(dummy_gm, dummy_qs_correct_shape, dummy_pm)) as mock_init_pymdp:
        belief_state = agent.initialize_beliefs(multi_sae_features_dict)

    mock_init_pymdp.assert_called_once_with(mock.ANY) # Check it was called with the list of features
    assert agent.use_pymdp is True
    assert belief_state.generative_model == dummy_gm
    if num_total_features > 0:
        assert np.allclose(belief_state.posterior_beliefs, dummy_qs_correct_shape)
        assert np.allclose(belief_state.precision_matrix, dummy_pm)
        assert np.allclose(belief_state.qs, dummy_qs_correct_shape)
    else:
        assert len(belief_state.posterior_beliefs) == 0
        assert belief_state.precision_matrix.size == 0
        assert len(belief_state.qs) == 0


def test_initialize_beliefs_pymdp_init_returns_none(agent_config, mock_tracer, multi_sae_features_dict, caplog, monkeypatch):
    """Tests initialize_beliefs when pymdp init returns None (e.g. no features)."""
    agent_config.active_inference.use_pymdp = True
    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)
    monkeypatch.setattr(agent, 'pymdp_is_available', True) # Assume pymdp could be imported
    agent.use_pymdp = True # Force attempt to use pymdp

    logging.getLogger("src.active_inference.agent").propagate = True
    with mock.patch.object(agent, '_initialize_pymdp_components', return_value=(None, None, None)) as mock_init_pymdp:
        belief_state = agent.initialize_beliefs(multi_sae_features_dict) # Use non-empty features

    mock_init_pymdp.assert_called_once()
    assert "PyMDP component initialization failed (no features?), falling back to non-pymdp mode for this session." in caplog.text
    assert agent.use_pymdp is False # Should have fallen back
    assert belief_state.generative_model is None
    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())
    assert np.allclose(belief_state.qs, np.ones(num_total_features) / num_total_features if num_total_features > 0 else np.array([]))
    logging.getLogger("src.active_inference.agent").propagate = False


def test_initialize_beliefs_pymdp_logic_exception(agent_config, mock_tracer, multi_sae_features_dict, caplog, monkeypatch):
    """Tests initialize_beliefs when use_pymdp is True but component init raises an exception."""
    agent_config.active_inference.use_pymdp = True
    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)
    monkeypatch.setattr(agent, 'pymdp_is_available', True)
    agent.use_pymdp = True

    logging.getLogger("src.active_inference.agent").propagate = True
    with mock.patch.object(agent, '_initialize_pymdp_components', side_effect=Exception("PyMDP Init Error")) as mock_init_pymdp:
        belief_state = agent.initialize_beliefs(multi_sae_features_dict)

    mock_init_pymdp.assert_called_once()
    assert "PyMDP components initialization failed: PyMDP Init Error" in caplog.text
    assert agent.use_pymdp is False
    assert belief_state.generative_model is None
    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())
    assert np.allclose(belief_state.qs, np.ones(num_total_features) / num_total_features if num_total_features > 0 else np.array([]))
    logging.getLogger("src.active_inference.agent").propagate = False


# --- Tests for calculate_expected_free_energy ---

def test_efe_no_belief_state(agent_instance, sample_sae_feature, caplog):
    """Tests EFE calculation when belief_state is None."""
    logging.getLogger("src.active_inference.agent").propagate = True
    agent_instance.belief_state = None
    efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)
    assert efe == 0.0
    assert "EFE calculation failed: BeliefState is not initialized." in caplog.text
    logging.getLogger("src.active_inference.agent").propagate = False


def test_efe_no_pymdp(agent_instance, sample_sae_feature):
    """Tests EFE calculation when use_pymdp is False."""
    agent_instance.use_pymdp = False
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]), feature_importances={sample_sae_feature.feature_id: 0.7},
        connection_beliefs={}, uncertainty={sample_sae_feature.feature_id: 0.3}, confidence=0.7
    )

    # Mock the helper methods directly on the instance since they are part of the class
    with mock.patch.object(agent_instance, '_calculate_epistemic_value', return_value=0.3) as mock_epistemic, \
         mock.patch.object(agent_instance, '_calculate_pragmatic_value', return_value=0.7) as mock_pragmatic:

        efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)

    mock_epistemic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_pragmatic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)

    expected_efe = agent_instance.config.active_inference.epistemic_weight * 0.3 + \
                   (1 - agent_instance.config.active_inference.epistemic_weight) * 0.7
    assert efe == pytest.approx(expected_efe)

def test_efe_with_pymdp(agent_instance, sample_sae_feature):
    """Tests EFE calculation when use_pymdp is True."""
    agent_instance.use_pymdp = True
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]), feature_importances={sample_sae_feature.feature_id: 0.7},
        connection_beliefs={}, uncertainty={sample_sae_feature.feature_id: 0.3}, confidence=0.7,
        generative_model={"A": np.array([])}
    )

    with mock.patch.object(agent_instance, '_calculate_epistemic_value', return_value=0.3) as mock_epistemic, \
         mock.patch.object(agent_instance, '_calculate_pragmatic_value', return_value=0.7) as mock_pragmatic, \
         mock.patch.object(agent_instance, '_calculate_model_uncertainty_reduction', return_value=0.05) as mock_model_uncert, \
         mock.patch.object(agent_instance, '_calculate_causal_information_gain', return_value=0.02) as mock_causal_gain:

        efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)

    mock_epistemic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_pragmatic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_model_uncert.assert_called_once_with(sample_sae_feature)
    mock_causal_gain.assert_called_once_with(sample_sae_feature)

    ew = agent_instance.config.active_inference.epistemic_weight
    # efe = ew * epistemic_val + (1 - ew) * pragmatic_val
    # efe += ew * model_uncertainty_reduction + (1-ew) * causal_info_gain
    expected_efe_base = ew * 0.3 + (1 - ew) * 0.7
    expected_efe_pymdp_bonus = ew * 0.05 + (1-ew) * 0.02
    expected_efe = expected_efe_base + expected_efe_pymdp_bonus
    assert efe == pytest.approx(expected_efe)

    # Intervention with large negative effect size (should still decrease uncertainty)
    agent_instance.belief_state.feature_importances[feature_id] = 0.01 # Low initial importance
    agent_instance.belief_state.uncertainty[feature_id] = 0.98 # High initial uncertainty

    large_neg_effect_intervention = InterventionResult(
        intervention_type=InterventionType.ABLATION, target_feature=sample_sae_feature,
        original_logits=torch.randn(1,1,1), intervened_logits=torch.randn(1,1,1),
        effect_size=-0.8, target_token_change=-0.1, intervention_layer=sample_sae_feature.layer
    )
    agent_instance.update_beliefs(large_neg_effect_intervention)
    assert 0.0 <= agent_instance.belief_state.feature_importances[feature_id] <= 1.0
    assert 0.01 <= agent_instance.belief_state.uncertainty[feature_id] <= 1.0

# --- Tests for initialize_beliefs ---

@pytest.fixture
def multi_sae_features_dict():
    # d_model could be part of a shared fixture or config
    d_model = 768
    features = {
        0: [ # Layer 0
            SAEFeature(feature_id=1, layer=0, activation_threshold=0.1, description="L0F1", max_activation=0.9, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model)),
            SAEFeature(feature_id=2, layer=0, activation_threshold=0.1, description="L0F2", max_activation=0.7, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model))
        ],
        1: [ # Layer 1
            SAEFeature(feature_id=3, layer=1, activation_threshold=0.1, description="L1F3", max_activation=0.95, examples=[], feature_vector=np.random.rand(d_model), decoder_weights=np.random.rand(d_model))
        ]
    }
    return features

def test_initialize_beliefs_no_features(agent_instance, caplog):
    """Tests initialize_beliefs with an empty feature dictionary."""
    agent_instance.initialize_beliefs({}) # Empty features

    assert agent_instance.belief_state is not None
    assert len(agent_instance.belief_state.qs) == 0 # Based on updated _create_empty_belief_state
    assert not agent_instance.belief_state.feature_importances
    assert not agent_instance.belief_state.uncertainty
    assert agent_instance.belief_state.confidence == 0.0
    assert "No features found for belief initialization" in caplog.text

def test_initialize_beliefs_with_features(agent_instance, multi_sae_features_dict):
    """Tests initialize_beliefs with a sample set of features."""
    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())

    with mock.patch.object(agent_instance, '_initialize_connection_beliefs', return_value={}) as mock_init_conn:
        belief_state = agent_instance.initialize_beliefs(multi_sae_features_dict)

    assert belief_state is not None
    assert len(belief_state.qs) == num_total_features
    assert np.allclose(belief_state.qs, np.ones(num_total_features) / num_total_features) # Uniform distribution

    for layer_idx, sae_features_in_layer in multi_sae_features_dict.items():
        for sae_feature in sae_features_in_layer:
            fid = sae_feature.feature_id
            assert fid in belief_state.feature_importances
            assert belief_state.feature_importances[fid] == sae_feature.max_activation
            assert fid in belief_state.uncertainty
            assert belief_state.uncertainty[fid] == pytest.approx(np.clip(1.0 - sae_feature.max_activation, 0.01, 1.0))

    # Test confidence calculation (relies on the actual uncertainty values)
    expected_avg_uncertainty = np.mean([np.clip(1.0 - f.max_activation, 0.01, 1.0) for layer_fs in multi_sae_features_dict.values() for f in layer_fs])
    expected_confidence = np.clip(1.0 - expected_avg_uncertainty, 0.0, 1.0)
    assert belief_state.confidence == pytest.approx(expected_confidence)

    mock_init_conn.assert_called_once()


def test_initialize_beliefs_pymdp_logic_success(agent_config, mock_tracer, multi_sae_features_dict, monkeypatch):
    """Tests initialize_beliefs when use_pymdp is True and component init succeeds."""
    agent_config.active_inference.use_pymdp = True
    # Create agent first
    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)

    # If pymdp wasn't actually available during __init__, self.use_pymdp would be False.
    # For this test, we want to simulate it being available AND configured.
    monkeypatch.setattr(agent, 'pymdp_is_available', True)
    agent.use_pymdp = agent.config.active_inference.use_pymdp and agent.pymdp_is_available # Re-apply init logic for use_pymdp

    assert agent.use_pymdp is True # Pre-condition for the test's purpose

    dummy_gm = {"A": np.array([[0.5,0.5],[0.5,0.5]])}
    dummy_qs = np.array([0.5, 0.5, 0.5]) # Adjusted to match num_total_features
    num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())
    dummy_qs_correct_shape = np.ones(num_total_features) / num_total_features
    dummy_pm = np.eye(num_total_features)

    with mock.patch.object(agent, '_initialize_pymdp_components', return_value=(dummy_gm, dummy_qs_correct_shape, dummy_pm)) as mock_init_pymdp:
        belief_state = agent.initialize_beliefs(multi_sae_features_dict)

    mock_init_pymdp.assert_called_once()
    assert agent.use_pymdp is True # Should remain true
    assert belief_state.generative_model == dummy_gm
    assert np.allclose(belief_state.posterior_beliefs, dummy_qs_correct_shape)
    assert np.allclose(belief_state.precision_matrix, dummy_pm)
    assert np.allclose(belief_state.qs, dummy_qs_correct_shape) # qs should be posterior_beliefs

def test_initialize_beliefs_pymdp_logic_failure(agent_config, mock_tracer, multi_sae_features_dict, caplog, monkeypatch):
    """Tests initialize_beliefs when use_pymdp is True but component init fails."""
    agent_config.active_inference.use_pymdp = True
    # Ensure pymdp is reported as available for the agent to try using it
    monkeypatch.setattr(agent_config.active_inference, 'use_pymdp', True)

    # We need to ensure the agent's self.pymdp_is_available is True for it to attempt pymdp logic
    # This is tricky because it's set in __init__. We re-create agent or monkeypatch.
    # For simplicity in this test, we'll assume the agent was created with pymdp_is_available = True
    # This can be done by ensuring pymdp *is* importable or by monkeypatching self.pymdp_is_available

    agent = ActiveInferenceAgent(config=agent_config, tracer=mock_tracer)
    # Force agent to think pymdp is available if it's not, to test the try-except block
    # Ensure logger will propagate to caplog for this test
    logger_to_test = logging.getLogger("src.active_inference.agent")
    original_propagate_status = logger_to_test.propagate
    logger_to_test.propagate = True

    try:
        if not agent.pymdp_is_available:
            monkeypatch.setattr(agent, 'pymdp_is_available', True) # Force attempt
            # Re-set use_pymdp based on this, as __init__ logic would have run
            agent.use_pymdp = agent.config.active_inference.use_pymdp and agent.pymdp_is_available


        with mock.patch.object(agent, '_initialize_pymdp_components', side_effect=Exception("PyMDP Init Error")) as mock_init_pymdp:
            belief_state = agent.initialize_beliefs(multi_sae_features_dict)

        mock_init_pymdp.assert_called_once()
        assert "PyMDP components initialization failed: PyMDP Init Error" in caplog.text
        assert agent.use_pymdp is False
        assert belief_state.generative_model is None
        # Check that qs is the default uniform, not from pymdp
        num_total_features = sum(len(f_list) for f_list in multi_sae_features_dict.values())
        assert np.allclose(belief_state.qs, np.ones(num_total_features) / num_total_features if num_total_features > 0 else np.array([]))
    finally:
        logger_to_test.propagate = original_propagate_status # Reset propagate status


# --- Tests for calculate_expected_free_energy ---

def test_efe_no_belief_state(agent_instance, sample_sae_feature):
    """Tests EFE calculation when belief_state is None."""
    agent_instance.belief_state = None
    efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)
    assert efe == 0.0

def test_efe_no_pymdp(agent_instance, sample_sae_feature):
    """Tests EFE calculation when use_pymdp is False."""
    agent_instance.use_pymdp = False # Ensure it's false for this agent instance
    # Initialize a basic belief state
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]), feature_importances={101: 0.7},
        connection_beliefs={}, uncertainty={101: 0.3}, confidence=0.7
    )

    with mock.patch.object(agent_instance, '_calculate_epistemic_value', return_value=0.3) as mock_epistemic, \
         mock.patch.object(agent_instance, '_calculate_pragmatic_value', return_value=0.7) as mock_pragmatic:

        efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)

    mock_epistemic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_pragmatic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)

    expected_efe = agent_instance.config.active_inference.epistemic_weight * 0.3 + \
                   (1 - agent_instance.config.active_inference.epistemic_weight) * 0.7
    assert efe == pytest.approx(expected_efe)

def test_efe_with_pymdp(agent_instance, sample_sae_feature):
    """Tests EFE calculation when use_pymdp is True."""
    agent_instance.use_pymdp = True
    # Initialize a belief state with a generative model structure
    agent_instance.belief_state = BeliefState(
        qs=np.array([1.0]), feature_importances={101: 0.7},
        connection_beliefs={}, uncertainty={101: 0.3}, confidence=0.7,
        generative_model={"A": np.array([])} # Needs generative_model to not be None
    )

    with mock.patch.object(agent_instance, '_calculate_epistemic_value', return_value=0.3) as mock_epistemic, \
         mock.patch.object(agent_instance, '_calculate_pragmatic_value', return_value=0.7) as mock_pragmatic, \
         mock.patch.object(agent_instance, '_calculate_model_uncertainty_reduction', return_value=0.05) as mock_model_uncert, \
         mock.patch.object(agent_instance, '_calculate_causal_information_gain', return_value=0.02) as mock_causal_gain:

        efe = agent_instance.calculate_expected_free_energy(sample_sae_feature, InterventionType.ABLATION)

    mock_epistemic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_pragmatic.assert_called_once_with(sample_sae_feature, InterventionType.ABLATION)
    mock_model_uncert.assert_called_once_with(sample_sae_feature)
    mock_causal_gain.assert_called_once_with(sample_sae_feature)

    ew = agent_instance.config.active_inference.epistemic_weight
    expected_efe = ew * (0.3 + 0.05) + (1 - ew) * (0.7 + 0.02)
    assert efe == pytest.approx(expected_efe)

# print("Test file created: tests/test_agent.py") # This was from create_file_with_block, remove for replace
