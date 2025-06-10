import pytest
import torch
import numpy as np
import json
from unittest import mock
from pathlib import Path
import logging # For caplog
import sys

# Adjust path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.runner import YorKExperimentRunner
from src.config.experiment_config import (
    CompleteConfig, ModelConfig, SAEConfig, ActiveInferenceConfig,
    ExperimentConfig, ResearchQuestionConfig, DeviceType, InterventionType, get_config
)
from src.core.data_structures import (
    ExperimentResult, CorrespondenceMetrics, NovelPrediction, SAEFeature, InterventionResult
)

# --- Fixtures ---

@pytest.fixture
def minimal_runner_config_dict_fixture(): # Renamed to avoid conflict with a var name
    """Provides a dictionary for a minimal CompleteConfig."""
    return {
        'model': {'name': 'test-gpt2', 'device': 'cpu'},
        'sae': {'enabled': True, 'target_layers': [0, 1], 'activation_threshold': 0.05},
        'active_inference': {'enabled': True, 'epistemic_weight': 0.5, 'max_interventions': 10, 'convergence_threshold': 0.1},
        'experiment': {'name': 'test_experiment', 'output_dir': 'test_results'}, # output_dir will be updated by other fixtures
        'research_questions': {
            'rq1_correspondence_target': 60.0,
            'rq2_efficiency_target': 25.0,
            'rq3_predictions_target': 2,
            'prediction_validation_confidence_threshold': 0.65
        }
    }

@pytest.fixture
def minimal_runner_config(minimal_runner_config_dict_fixture, tmp_path):
    """Provides a CompleteConfig instance with a temporary output directory."""
    config_dict = minimal_runner_config_dict_fixture.copy()
    config_dict['experiment']['output_dir'] = str(tmp_path / "experiment_results")

    model_config_data = config_dict['model'].copy()
    # Convert device string to DeviceType enum
    if isinstance(model_config_data.get('device'), str):
        try:
            model_config_data['device'] = DeviceType(model_config_data['device'])
        except ValueError:
            # Handle case where string is not a valid DeviceType member name
            model_config_data['device'] = DeviceType.CPU # Default or raise error
    elif not isinstance(model_config_data.get('device'), DeviceType):
        # If it's not str and not DeviceType (e.g. None or incorrect type), default
        model_config_data['device'] = DeviceType.CPU

    config = CompleteConfig(
        model=ModelConfig(**model_config_data), # Use updated model_config_data
        sae=SAEConfig(**config_dict['sae']),
        active_inference=ActiveInferenceConfig(**config_dict['active_inference']),
        experiment=ExperimentConfig(**config_dict['experiment']),
        research_questions=ResearchQuestionConfig(**config_dict['research_questions'])
    )
    return config

@pytest.fixture
def mock_experiment_result_fixture(minimal_runner_config_dict_fixture): # Use dict fixture
    """Creates a basic ExperimentResult object."""
    return ExperimentResult(
        experiment_name="test_experiment",
        timestamp="2023-01-01T12:00:00",
        config_used=minimal_runner_config_dict_fixture,
        correspondence_metrics=[],
        efficiency_metrics={},
        novel_predictions=[],
        rq1_passed=False,
        rq2_passed=False,
        rq3_passed=False,
        overall_success=False,
        intervention_results=[],
        metadata={'duration_seconds': 10.0}
    )

@pytest.fixture
def runner_instance(minimal_runner_config, tmp_path, mocker): # Added mocker
    """Fixture to create a YorKExperimentRunner instance."""
    minimal_runner_config.experiment.output_dir = str(tmp_path / "runner_test_results")

    # Mock get_config that is used inside YorKExperimentRunner.__init__
    mocker.patch('src.experiments.runner.get_config', return_value=minimal_runner_config)
    runner = YorKExperimentRunner()
    return runner

# --- Test Cases ---

def test_runner_initialization(minimal_runner_config, tmp_path): # Removed runner_instance fixture
    """Test YorKExperimentRunner initialization."""
    output_dir_path = Path(minimal_runner_config.experiment.output_dir)
    # Ensure the specific output_dir for this config is used
    minimal_runner_config.experiment.output_dir = str(output_dir_path)


    with mock.patch('src.experiments.runner.get_config', return_value=minimal_runner_config):
        runner = YorKExperimentRunner()

    assert runner.config == minimal_runner_config
    assert runner.output_dir == output_dir_path
    assert output_dir_path.exists()

def test_setup_experiment_success(runner_instance, mocker):
    mock_circuit_tracer_constructor = mocker.patch('src.experiments.runner.CircuitTracer')
    mock_ai_agent_constructor = mocker.patch('src.experiments.runner.ActiveInferenceAgent')
    mocker.patch.object(runner_instance, '_validate_configuration', return_value=[])

    runner_instance.setup_experiment()

    mock_circuit_tracer_constructor.assert_called_once_with(runner_instance.config)
    mock_ai_agent_constructor.assert_called_once_with(runner_instance.config, runner_instance.tracer)
    assert runner_instance.tracer is not None
    assert runner_instance.ai_agent is not None

def test_setup_experiment_config_validation_fails(runner_instance, mocker):
    mocker.patch.object(runner_instance, '_validate_configuration', return_value=["config error 1"])

    with pytest.raises(ValueError, match="Configuration validation failed: \\['config error 1'\\]"): # Escaped brackets
        runner_instance.setup_experiment()

def test_validate_research_questions(runner_instance):
    config_rq = runner_instance.config.research_questions

    metrics1 = [CorrespondenceMetrics(overall_correspondence=config_rq.rq1_correspondence_target / 100 + 0.1, belief_updating_correspondence=0, precision_weighting_correspondence=0, prediction_error_correspondence=0)]
    eff_metrics1 = {'overall_improvement': config_rq.rq2_efficiency_target + 5.0}
    preds1 = config_rq.rq3_predictions_target + 1

    result1 = runner_instance.validate_research_questions(metrics1, eff_metrics1, preds1)
    assert result1['rq1_passed']
    assert result1['rq2_passed']
    assert result1['rq3_passed']
    assert result1['overall_success'] # overall_success is True if 2 or more RQs pass. All 3 pass here.
    assert result1['rq1_achieved'] == pytest.approx(config_rq.rq1_correspondence_target + 10.0)

    metrics2 = [CorrespondenceMetrics(overall_correspondence=config_rq.rq1_correspondence_target / 100 - 0.1, belief_updating_correspondence=0, precision_weighting_correspondence=0, prediction_error_correspondence=0)]
    eff_metrics2 = {'overall_improvement': config_rq.rq2_efficiency_target + 5.0}
    preds2 = config_rq.rq3_predictions_target - 1

    result2 = runner_instance.validate_research_questions(metrics2, eff_metrics2, preds2)
    assert not result2['rq1_passed']
    assert result2['rq2_passed']
    assert not result2['rq3_passed']
    assert not result2['overall_success'] # Only 1 RQ passed, so overall_success should be False (needs >=2)

    result3 = runner_instance.validate_research_questions([], eff_metrics1, preds1)
    assert not result3['rq1_passed']
    assert result3['rq1_achieved'] == 0.0

def test_validate_configuration(runner_instance, mocker): # Changed monkeypatch to mocker
    assert runner_instance._validate_configuration() == []

    original_weight = runner_instance.config.active_inference.epistemic_weight
    runner_instance.config.active_inference.epistemic_weight = 1.5
    issues = runner_instance._validate_configuration()
    assert "epistemic_weight must be between 0 and 1" in issues
    runner_instance.config.active_inference.epistemic_weight = original_weight

    # Test TransformerLens import error
    # Mock the import within the runner module to make it fail
    mocker.patch.dict(sys.modules, {'transformer_lens': None})
    issues_import_error = runner_instance._validate_configuration()
    assert "TransformerLens not available - required for circuit analysis" in issues_import_error
    # Restore transformer_lens if it was there, or ensure it's cleaned up
    if 'transformer_lens' in sys.modules and sys.modules['transformer_lens'] is None:
        del sys.modules['transformer_lens']
    # No need to restore original_transformer_lens if we are deleting the key always after making it None


def test_calculate_efficiency_metrics(runner_instance):
    # This test assumes runner.py's _calculate_efficiency_metrics is corrected
    ai_interventions = 10
    baseline_counts = {
        'random': [20, 25, 30],
        'high_activation': [15, 20],
        'empty_baseline': []
    }
    metrics = runner_instance._calculate_efficiency_metrics(ai_interventions, baseline_counts)

    assert metrics['ai_interventions'] == ai_interventions
    assert metrics['random_improvement'] == pytest.approx(60.0)
    assert metrics['high_activation_improvement'] == pytest.approx(42.857, rel=1e-3)
    assert metrics['empty_baseline_improvement'] == 0.0
    assert 'overall_improvement' in metrics # overall_improvement is calculated from these

    metrics_no_baseline = runner_instance._calculate_efficiency_metrics(ai_interventions, {})
    assert metrics_no_baseline['overall_improvement'] == 0.0

def test_validate_predictions(runner_instance):
    predictions = [
        NovelPrediction("feature_interaction", "desc1", "hyp1", "out1", "meth1", confidence=0.8),
        NovelPrediction("attention_pattern", "desc2", "hyp2", "out2", "meth2", confidence=0.5),
        NovelPrediction("failure_mode", "desc3", "hyp3", "out3", "meth3", confidence=0.9),
    ]
    runner_instance.config.research_questions.prediction_validation_confidence_threshold = 0.7

    validated_count = runner_instance._validate_predictions(predictions, ["test input"])

    assert validated_count == 2
    assert predictions[0].validation_status == 'validated'
    assert predictions[1].validation_status == 'falsified'
    assert predictions[2].validation_status == 'validated'

def test_run_experiment_no_inputs(runner_instance, mocker):
    mocker.patch.object(runner_instance, 'setup_experiment')
    runner_instance.ai_agent = mock.MagicMock() # Assign mock ai_agent
    runner_instance.tracer = mock.MagicMock() # Assign mock tracer

    mocker.patch.object(runner_instance, '_discover_all_active_features', return_value={})
    mocker.patch.object(runner_instance, '_run_ai_interventions', return_value=([], []))
    mocker.patch.object(runner_instance, '_run_baseline_comparisons', return_value={})
    mocker.patch.object(runner_instance.ai_agent, 'generate_predictions', return_value=[])
    mocker.patch.object(runner_instance, '_validate_predictions', return_value=0)

    result = runner_instance.run_experiment([])

    assert isinstance(result, ExperimentResult)
    assert len(result.correspondence_metrics) == 0
    assert len(result.intervention_results) == 0
    assert result.novel_predictions == []
    assert result.metadata['duration_seconds'] >= 0

@mock.patch('json.dump')
def test_save_results(mock_json_dump, runner_instance, mock_experiment_result_fixture, tmp_path): # use fixture
    output_dir = tmp_path / "custom_output"
    # save_results method creates the directory

    runner_instance.save_results(mock_experiment_result_fixture, str(output_dir))

    assert output_dir.exists() # Check directory was actually created

    assert mock_json_dump.call_count == 1
    args, kwargs = mock_json_dump.call_args

    assert args[0]['experiment_name'] == mock_experiment_result_fixture.experiment_name

    file_object = args[1]
    expected_filename = f"experiment_results_{mock_experiment_result_fixture.timestamp}.json"
    assert file_object.name == str(output_dir / expected_filename)
