# ActiveCircuitDiscovery Enhanced API Reference

## Overview

ActiveCircuitDiscovery provides a comprehensive enhanced framework for circuit discovery in Large Language Models using Active Inference principles with statistical validation, prediction generation, and comprehensive analysis capabilities. This API reference covers all major enhanced components and their usage.

## 🏗️ Enhanced Architecture

The enhanced system is organized into modular components with clear separation of concerns:

```
src/
├── core/                     # Core data structures and enhanced systems
├── circuit_analysis/         # Circuit discovery and analysis
├── active_inference/         # Active Inference implementation
├── config/                   # Enhanced configuration management
├── experiments/              # Experiment orchestration and validation
└── visualization/           # Enhanced visualization and reporting
```

## 🔧 Core Enhanced Components

### core.data_structures

#### Enhanced Data Classes

All data structures include comprehensive validation and enhanced functionality:

**SAEFeature** - Enhanced sparse autoencoder feature representation
```python
from src.core.data_structures import SAEFeature

feature = SAEFeature(
    feature_id=1234,
    layer=8,
    activation_threshold=0.05,
    description="Golden Gate Bridge concept",
    max_activation=0.82,
    examples=["bridge", "suspension bridge", "San Francisco landmark"],
    feature_vector=np.array([...]),  # Optional: actual feature vector
    decoder_weights=np.array([...])  # Optional: decoder weights
)

# Enhanced validation - automatically validates ranges
assert 0 <= feature.max_activation <= 1
assert feature.layer >= 0
```

**BeliefState** - Enhanced Active Inference belief representation
```python
from src.core.data_structures import BeliefState

belief_state = BeliefState(
    qs=np.array([0.7, 0.3]),  # PyMDP posterior beliefs
    feature_importances={1: 0.8, 2: 0.6},
    connection_beliefs={(1, 2): 0.7},
    uncertainty={1: 0.2, 2: 0.3},
    confidence=0.85,
    generative_model={...},  # Optional: PyMDP model
    precision_matrix=np.array([...])  # Optional: precision weights
)

# Enhanced methods
entropy = belief_state.get_entropy()
avg_uncertainty = belief_state.get_average_uncertainty()
```

**CorrespondenceMetrics** - Enhanced correspondence measurement with validation
```python
from src.core.data_structures import CorrespondenceMetrics

metrics = CorrespondenceMetrics(
    belief_updating_correspondence=0.78,
    precision_weighting_correspondence=0.82,
    prediction_error_correspondence=0.75,
    overall_correspondence=0.78,
    # Supporting evidence arrays
    circuit_attention_patterns=[0.8, 0.9, 0.7],
    ai_precision_patterns=[0.85, 0.88, 0.72]
)

# Automatic validation ensures all values in [0,1]
```

**NovelPrediction** - Enhanced prediction with validation framework
```python
from src.core.data_structures import NovelPrediction

prediction = NovelPrediction(
    prediction_type="attention_pattern",  # validated enum
    description="High-precision features will receive more attention",
    testable_hypothesis="Features with uncertainty < 0.3 will have attention weights > 0.6",
    expected_outcome="Strong positive correlation (r > 0.7)",
    test_method="Correlation analysis between uncertainty and attention",
    confidence=0.85,
    validation_status="validated",  # 'untested', 'validated', 'falsified'
    validation_evidence={"correlation": 0.73, "p_value": 0.002}
)
```

**AttributionGraph** - Enhanced graph structure with proper node/edge classes
```python
from src.core.data_structures import AttributionGraph, GraphNode, GraphEdge

# Create enhanced attribution graph
nodes = [
    GraphNode("layer_8_feature_1234", layer=8, feature_id=1234, 
              importance=0.85, description="Bridge concept"),
    GraphNode("layer_10_feature_5678", layer=10, feature_id=5678,
              importance=0.72, description="Location concept")
]

edges = [
    GraphEdge("layer_8_feature_1234", "layer_10_feature_5678", 
              weight=0.67, confidence=0.89, edge_type="causal")
]

graph = AttributionGraph(
    input_text="The Golden Gate Bridge is located in",
    nodes=nodes,
    edges=edges,
    target_output="San Francisco",
    confidence=0.83
)

# Enhanced methods
node = graph.get_node_by_id("layer_8_feature_1234")
in_degree, out_degree = graph.get_node_degree("layer_8_feature_1234")
```

### core.statistical_validation

#### StatisticalValidator - Comprehensive Statistical Testing

```python
from src.core.statistical_validation import StatisticalValidator, perform_comprehensive_validation

# Initialize validator
validator = StatisticalValidator(alpha=0.05, bootstrap_samples=10000)

# Test correspondence significance
correspondence_test = validator.validate_correspondence_significance(
    correspondence_metrics, target_threshold=70.0
)
print(f"Correspondence: p={correspondence_test.p_value:.4f}, d={correspondence_test.effect_size:.3f}")

# Test efficiency improvement
efficiency_tests = validator.validate_efficiency_improvement(
    ai_interventions=[12, 15, 11],
    baseline_interventions={"random": [25, 30, 28], "sequential": [22, 26, 24]},
    target_improvement=30.0
)

# Test prediction success rate
prediction_test = validator.validate_prediction_success_rate(
    predictions, target_count=3
)

# Comprehensive validation
validation_results = perform_comprehensive_validation(experiment_result)
```

#### Statistical Test Results

```python
from src.core.statistical_validation import StatisticalTest

# Example test result
test_result = StatisticalTest(
    test_name="correspondence_significance",
    statistic=3.45,
    p_value=0.002,
    confidence_interval=(72.1, 84.7),
    effect_size=0.82,
    power=0.95,
    significant=True,
    interpretation="Correspondence (78.3%) significantly exceeds 70% threshold"
)
```

### core.prediction_system

#### Enhanced Prediction Generation

```python
from src.core.prediction_system import (
    EnhancedPredictionGenerator, 
    AttentionPatternPredictor,
    FeatureInteractionPredictor, 
    FailureModePredictor
)

# Initialize enhanced prediction system
generator = EnhancedPredictionGenerator()

# Generate comprehensive predictions
predictions = generator.generate_comprehensive_predictions(
    belief_state=belief_state,
    circuit_graph=attribution_graph,
    intervention_history=intervention_results
)

# Use individual predictors
attention_predictor = AttentionPatternPredictor()
attention_predictions = attention_predictor.generate_predictions(
    belief_state, {"feature_uncertainties": uncertainties}
)

feature_predictor = FeatureInteractionPredictor()
interaction_predictions = feature_predictor.generate_predictions(
    belief_state, attribution_graph
)

failure_predictor = FailureModePredictor()
failure_predictions = failure_predictor.generate_predictions(
    belief_state, intervention_results
)
```

### core.prediction_validator

#### Empirical Prediction Validation

```python
from src.core.prediction_validator import PredictionValidator, ValidationConfig

# Configure validation
config = ValidationConfig(
    significance_level=0.05,
    min_sample_size=15,
    bootstrap_samples=1000
)

validator = PredictionValidator(config)

# Validate predictions with empirical data
test_data = {
    'feature_uncertainties': np.random.beta(2, 2, 20),
    'attention_weights': np.random.beta(3, 2, 20)
}

validation_result = validator.validate_prediction(prediction, test_data)
print(f"Validation: {validation_result.validation_status}")
print(f"Evidence: {validation_result.evidence}")
```

### core.metrics

#### Enhanced Metrics Calculators

```python
from src.core.metrics import (
    CorrespondenceCalculator, 
    EfficiencyCalculator, 
    ValidationCalculator
)

# Initialize calculators
correspondence_calc = CorrespondenceCalculator()
efficiency_calc = EfficiencyCalculator()
validation_calc = ValidationCalculator(
    rq1_target=70.0, rq2_target=30.0, rq3_target=3
)

# Calculate correspondence with proper bounds [0,100%]
correspondence = correspondence_calc.calculate_correspondence(
    ai_beliefs=belief_state,
    circuit_behavior=intervention_results
)

# Calculate efficiency improvements
efficiency = efficiency_calc.calculate_efficiency_improvement(
    ai_interventions=12,
    baseline_results={"random": 28, "sequential": 24}
)

# Validate research questions
rq_validation = validation_calc.validate_research_questions(
    correspondence_metrics, efficiency_metrics, novel_predictions
)
```

## 🎯 Circuit Analysis Components

### circuit_analysis.tracer

#### CircuitTracer - Enhanced Circuit Discovery

```python
from src.circuit_analysis.tracer import CircuitTracer
from src.config.experiment_config import CompleteConfig

# Initialize with enhanced configuration
config = CompleteConfig()
config.sae.auto_discover_layers = True  # Auto-discover across all layers
config.sae.activation_threshold = 0.03
config.model.device = "auto"

tracer = CircuitTracer(config)

# Auto-discover active features across ALL layers
active_features = tracer.find_active_features(
    "The Golden Gate Bridge is located in",
    threshold=0.05
)
# Returns: Dict[int, List[SAEFeature]] - all layers with active features

# Perform comprehensive interventions
from src.config.experiment_config import InterventionType

result = tracer.perform_intervention(
    text="The Golden Gate Bridge is located in",
    feature=target_feature,
    intervention_type=InterventionType.ABLATION
)

# Build complete attribution graph
attribution_graph = tracer.build_attribution_graph(
    "The Golden Gate Bridge is located in"
)

# Get feature activations for specific layer
activations = tracer.get_feature_activations(
    "The Golden Gate Bridge is located in", layer=8
)
```

#### Enhanced SAE Integration

```python
# Automatic fallback system
# 1. Tries sae-lens if available
# 2. Falls back to custom analyzers
# 3. Graceful degradation with warnings

# Enhanced error handling
try:
    features = tracer.find_active_features(text)
except ImportError as e:
    logger.warning(f"Advanced SAE features unavailable: {e}")
    # Falls back to basic analysis
```

## 🧠 Active Inference Components

### active_inference.agent

#### ActiveInferenceAgent - Complete Belief Updating

```python
from src.active_inference.agent import ActiveInferenceAgent

# Initialize with enhanced configuration
agent = ActiveInferenceAgent(config, tracer)

# Initialize beliefs from discovered features
belief_state = agent.initialize_beliefs(active_features)

# Calculate Expected Free Energy for intervention selection
efe = agent.calculate_expected_free_energy(
    feature=candidate_feature,
    intervention_type=InterventionType.ABLATION
)

# Update beliefs with intervention results
correspondence_metrics = agent.update_beliefs(intervention_result)

# Generate novel predictions
predictions = agent.generate_predictions()

# Check convergence
converged = agent.check_convergence(threshold=0.15)
```

#### PyMDP Integration

```python
# Proper A/B matrices for generative model
agent._setup_generative_model(num_states=10, num_observations=5)

# Belief updating with prediction error minimization
agent._update_posterior_beliefs(observation, prediction_error)

# Policy selection using Expected Free Energy
policy = agent._select_policy(available_actions, current_beliefs)
```

## 🎛️ Enhanced Configuration

### config.experiment_config

#### Complete Configuration Management

```python
from src.config.experiment_config import (
    CompleteConfig, get_config, get_enhanced_config,
    StatisticalValidationConfig, PredictionValidationConfig
)

# Load enhanced configuration
config = get_enhanced_config()

# Access enhanced sections
config.statistical_validation.enabled = True
config.statistical_validation.bootstrap_samples = 10000
config.prediction_validation.min_sample_size = 15
config.visualization.publication_ready = True

# Save custom configuration
from src.config.experiment_config import ConfigManager
manager = ConfigManager("custom_config.yaml")
manager.save_config(config)

# Validate configuration
issues = manager.validate_config(config)
if issues:
    print(f"Configuration issues: {issues}")
```

#### Enhanced Configuration Options

```yaml
# enhanced_config.yaml
statistical_validation:
  enabled: true
  significance_level: 0.05
  bootstrap_samples: 10000
  confidence_level: 0.95
  multiple_comparisons_correction: "bonferroni"

prediction_validation:
  enabled: true
  min_sample_size: 15
  validation_threshold: 0.8
  cross_validation_folds: 5

visualization:
  enhanced_plots: true
  interactive_dashboards: true
  publication_ready: true
  save_formats: ["png", "pdf", "html"]
```

## 🧪 Experiment Framework

### experiments.runner

#### YorKExperimentRunner - Complete Experiment Orchestration

```python
from src.experiments.runner import YorKExperimentRunner

# Initialize enhanced runner
runner = YorKExperimentRunner()

# Setup with enhanced configuration
config = get_enhanced_config()
runner.setup_experiment(config)

# Run comprehensive experiment
test_inputs = [
    "The Golden Gate Bridge is located in",
    "San Francisco's most famous landmark is the"
]

results = runner.run_experiment(test_inputs)

# Enhanced results include:
# - Statistical validation results
# - Novel prediction validation
# - Comprehensive efficiency analysis
# - Research question validation

# Validate research questions with statistical rigor
rq_validation = runner.validate_research_questions(
    results.correspondence_metrics,
    results.efficiency_metrics, 
    results.novel_predictions
)

# Save comprehensive results
runner.save_results(results, "enhanced_experiment_results")
```

#### Experiment Result Structure

```python
from src.core.data_structures import ExperimentResult

# Enhanced experiment results
result = ExperimentResult(
    experiment_name="enhanced_golden_gate_discovery",
    timestamp=datetime.now().isoformat(),
    config_used=config_dict,
    correspondence_metrics=correspondence_list,
    efficiency_metrics=efficiency_dict,
    novel_predictions=prediction_list,
    rq1_passed=True,  # >70% correspondence achieved
    rq2_passed=True,  # >30% efficiency improvement
    rq3_passed=True,  # >3 validated predictions
    overall_success=True,
    intervention_results=intervention_list,
    belief_history=belief_history_list,
    metadata={'duration_seconds': 45.2, 'gpu_memory_used': '8.2GB'}
)

# Enhanced methods
summary = result.get_summary_stats()
success_rate = result.success_rate  # 1.0 for complete success
result_dict = result.to_dict()  # For serialization
```

## 📈 Enhanced Visualization

### visualization.visualizer

#### CircuitVisualizer - Publication-Ready Visualizations

```python
from src.visualization.visualizer import CircuitVisualizer

# Initialize enhanced visualizer
visualizer = CircuitVisualizer("enhanced_visualizations")

# Create comprehensive visualization suite
visualization_files = visualizer.generate_all_visualizations(
    experiment_result,
    attribution_graph=graph,
    belief_history=belief_history,
    statistical_validation=statistical_results
)

# Individual enhanced visualizations
circuit_file = visualizer.create_circuit_diagram(graph, "circuit")
metrics_file = visualizer.create_metrics_dashboard(result, "dashboard")
beliefs_file = visualizer.create_belief_evolution_plot(belief_history, "beliefs")
predictions_file = visualizer.create_prediction_validation_plot(predictions, "predictions")
stats_file = visualizer.create_statistical_validation_plot(stat_tests, "statistics")

# Comprehensive interactive dashboard
dashboard_file = visualizer.create_comprehensive_dashboard(
    result, statistical_validation, "comprehensive_dashboard"
)
```

#### Enhanced Visualization Features

- **Interactive Dashboards**: Multi-panel Plotly dashboards with drill-down
- **Statistical Validation Plots**: P-value distributions, effect sizes, power analysis
- **Circuit Diagrams**: NetworkX and CircuitsVis integration with fallbacks
- **Prediction Validation**: Confidence vs. success analysis
- **Publication Ready**: Multiple format output (PNG, PDF, HTML)

## 🔧 Advanced Usage Examples

### Complete Enhanced Pipeline

```python
from src.experiments.runner import YorKExperimentRunner
from src.config.experiment_config import get_enhanced_config
from src.core.statistical_validation import perform_comprehensive_validation
from src.visualization.visualizer import CircuitVisualizer

# 1. Setup enhanced experiment
config = get_enhanced_config()
runner = YorKExperimentRunner()
runner.setup_experiment(config)

# 2. Run experiment with statistical validation
test_inputs = ["The Golden Gate Bridge is located in"]
results = runner.run_experiment(test_inputs)

# 3. Perform comprehensive statistical validation
statistical_validation = perform_comprehensive_validation(results)

# 4. Generate publication-ready visualizations
visualizer = CircuitVisualizer("publication_figures")
visualization_files = visualizer.generate_all_visualizations(
    results, statistical_validation=statistical_validation
)

# 5. Print comprehensive results
print(f"🎯 Research Questions:")
print(f"   RQ1: {'✅ PASSED' if results.rq1_passed else '❌ FAILED'}")
print(f"   RQ2: {'✅ PASSED' if results.rq2_passed else '❌ FAILED'}")
print(f"   RQ3: {'✅ PASSED' if results.rq3_passed else '❌ FAILED'}")
print(f"   Success Rate: {results.success_rate:.1%}")

print(f"\n📊 Statistical Validation:")
stats = statistical_validation['statistical_summary']
print(f"   Tests: {stats['total_tests']} performed, {stats['significant_tests']} significant")
print(f"   Effect Size: {stats['average_effect_size']:.3f}")
print(f"   Power: {stats['average_power']:.3f}")

print(f"\n📈 Visualizations:")
for viz_type, file_path in visualization_files.items():
    print(f"   {viz_type}: {file_path}")
```

### Custom Statistical Validation

```python
from src.core.statistical_validation import StatisticalValidator

# Initialize custom validator
validator = StatisticalValidator(alpha=0.01, bootstrap_samples=20000)

# Custom correspondence validation
correspondence_test = validator.validate_correspondence_significance(
    correspondence_metrics, target_threshold=75.0  # Higher threshold
)

# Custom efficiency validation with multiple baselines
efficiency_tests = validator.validate_efficiency_improvement(
    ai_interventions=[10, 12, 11],
    baseline_interventions={
        "random": [25, 28, 26],
        "sequential": [22, 24, 23],
        "exhaustive": [45, 48, 46],
        "gradient_based": [18, 20, 19]
    },
    target_improvement=35.0  # Higher threshold
)

# Print detailed results
print(f"Correspondence Test:")
print(f"  Statistic: {correspondence_test.statistic:.3f}")
print(f"  P-value: {correspondence_test.p_value:.6f}")
print(f"  Effect size: {correspondence_test.effect_size:.3f}")
print(f"  Power: {correspondence_test.power:.3f}")
print(f"  Significant: {correspondence_test.significant}")
print(f"  Interpretation: {correspondence_test.interpretation}")
```

### Enhanced Prediction Generation

```python
from src.core.prediction_system import EnhancedPredictionGenerator
from src.core.prediction_validator import PredictionValidator

# Generate theory-grounded predictions
generator = EnhancedPredictionGenerator()
predictions = generator.generate_comprehensive_predictions(
    belief_state, attribution_graph, intervention_results
)

# Validate predictions empirically
validator = PredictionValidator()
validated_predictions = []

for prediction in predictions:
    # Generate test data based on prediction type
    test_data = generate_test_data_for_prediction(prediction)
    
    # Validate with statistical testing
    validation_result = validator.validate_prediction(prediction, test_data)
    
    # Update prediction with validation results
    prediction.validation_status = validation_result.validation_status
    prediction.validation_evidence = validation_result.evidence
    
    validated_predictions.append(prediction)
    
    print(f"Prediction: {prediction.description[:50]}...")
    print(f"  Type: {prediction.prediction_type}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Validation: {validation_result.validation_status}")
    print(f"  Evidence: {validation_result.evidence}")
    print()
```

## 🚀 Advanced Features

### Auto-Discovery Across All Layers

```python
# Enhanced auto-discovery configuration
config.sae.auto_discover_layers = True
config.sae.layer_search_range = [0, -1]  # All layers
config.sae.activation_threshold = 0.03  # Lower threshold for comprehensive discovery

# Auto-discovery finds active features across entire model
active_features = tracer.find_active_features(text)
print(f"Auto-discovered features in {len(active_features)} layers:")
for layer, features in active_features.items():
    print(f"  Layer {layer}: {len(features)} features")
```

### Enhanced Error Handling and Fallbacks

```python
# Graceful degradation system
try:
    # Try enhanced SAE analysis
    features = tracer.find_active_features_with_sae_lens(text)
except ImportError:
    logger.warning("SAE-lens not available, using fallback analyzer")
    features = tracer._fallback_feature_discovery(text)
except Exception as e:
    logger.error(f"Feature discovery failed: {e}")
    features = {}

# Enhanced logging and monitoring
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ActiveCircuitDiscovery')
logger.info(f"Discovered {len(features)} active features")
```

### Performance Optimization

```python
# Memory management for large experiments
import torch

# Clear GPU cache between experiments
torch.cuda.empty_cache()

# Batch processing for multiple inputs
def process_batch(inputs, batch_size=4):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        batch_results = runner.run_experiment(batch)
        results.extend(batch_results)
        torch.cuda.empty_cache()  # Clear between batches
    return results

# Parallel processing for independent analyses
from concurrent.futures import ThreadPoolExecutor

def analyze_input(text):
    return tracer.build_attribution_graph(text)

with ThreadPoolExecutor(max_workers=4) as executor:
    graphs = list(executor.map(analyze_input, test_inputs))
```

## 🔬 Research Extensions

### Custom Research Questions

```python
# Define custom validation targets
config.research_questions.rq1_correspondence_target = 80.0  # Higher threshold
config.research_questions.rq2_efficiency_target = 40.0     # More ambitious
config.research_questions.rq3_predictions_target = 5       # More predictions

# Custom validation logic
def validate_custom_research_question(results):
    # Example: RQ4 - Stability across multiple runs
    stability_scores = []
    for run in multiple_runs:
        stability = calculate_stability_metric(run)
        stability_scores.append(stability)
    
    avg_stability = np.mean(stability_scores)
    rq4_passed = avg_stability > 0.85
    
    return {
        'rq4_passed': rq4_passed,
        'stability_score': avg_stability,
        'interpretation': f"Stability: {avg_stability:.3f}"
    }
```

### Novel Intervention Types

```python
# Custom intervention implementation
class NoiseInjectionIntervention:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def apply(self, activations, feature):
        # Add controlled noise to feature activations
        noise = torch.randn_like(activations) * self.noise_level
        return activations + noise

# Register custom intervention
tracer.register_intervention_type("noise_injection", NoiseInjectionIntervention())

# Use in experiments
result = tracer.perform_intervention(
    text, feature, "noise_injection"
)
```

This enhanced API reference provides comprehensive coverage of all enhanced components in ActiveCircuitDiscovery v2.0, including statistical validation, prediction systems, and advanced analysis capabilities.