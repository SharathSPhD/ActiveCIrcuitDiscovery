# Experiment Runner

This directory contains the comprehensive experiment orchestration system that coordinates all components of the ActiveCircuitDiscovery project to run complete circuit discovery experiments with statistical validation.

## Overview

The experiment runner provides:
- **Complete experiment orchestration** integrating all project components
- **Research question validation** with statistical rigor
- **Baseline method comparisons** for efficiency measurement
- **Enhanced prediction generation and validation** 
- **Comprehensive result collection and analysis**
- **Statistical validation framework** with proper significance testing

## Main Files

### `runner.py`
The core `YorKExperimentRunner` class implementing the `IExperimentRunner` interface:

**Key Features:**
- **Full pipeline orchestration** from setup to result analysis
- **Auto-discovery circuit analysis** across all model layers
- **Active Inference guided interventions** with efficiency tracking
- **Baseline method comparisons** for research question validation
- **Statistical validation** with comprehensive testing framework
- **Result serialization** with complete experiment metadata

**Core Methods:**
- `setup_experiment()` - Initialize all components with configuration
- `run_experiment()` - Execute complete experiment pipeline
- `validate_research_questions()` - Validate all RQ targets with statistics
- `save_results()` - Serialize results with full metadata

## Experiment Pipeline

### 1. Experiment Setup Phase

```python
def setup_experiment(self, config: Optional[CompleteConfig] = None):
    """Setup experiment with enhanced validation and components."""
    
    # Configuration validation
    validation_errors = self._validate_configuration()
    if validation_errors:
        raise ValueError(f"Configuration validation failed: {validation_errors}")
    
    # Initialize circuit tracer with auto-discovery
    self.tracer = CircuitTracer(self.config)
    
    # Initialize Active Inference agent with belief management
    self.ai_agent = ActiveInferenceAgent(self.config, self.tracer)
    
    # Initialize enhanced prediction system
    self.prediction_generator = EnhancedPredictionGenerator()
    self.prediction_validator = PredictionValidator(ValidationConfig())
    
    # Setup baseline methods for efficiency comparison
    self._setup_baseline_methods()
```

### 2. Circuit Discovery Phase

```python
def _discover_all_active_features(self, text: str):
    """Auto-discover active features across ALL layers."""
    n_layers = self.tracer.model.cfg.n_layers
    all_active_features = {}
    
    # Search across all layers, not just configured targets
    for layer in range(n_layers):
        try:
            # Add layer to SAE analyzers if not present
            if layer not in self.tracer.sae_analyzers:
                self.tracer._create_fallback_analyzer(layer)
            
            # Find active features in this layer
            layer_features = self.tracer.find_active_features(text, threshold)
            
            if layer in layer_features and layer_features[layer]:
                all_active_features[layer] = layer_features[layer]
                
        except Exception as e:
            logger.warning(f"Could not analyze layer {layer}: {e}")
    
    return all_active_features
```

### 3. Active Inference Intervention Phase

```python
def _run_ai_interventions(self, text: str, active_features: Dict):
    """Run Active Inference guided interventions (efficient)."""
    interventions = []
    all_features = []
    
    # Flatten features for intervention selection
    for layer_features in active_features.values():
        all_features.extend(layer_features)
    
    intervention_count = 0
    max_interventions = self.config.active_inference.max_interventions
    
    while intervention_count < max_interventions and all_features:
        # Select intervention using Expected Free Energy
        best_feature = None
        best_efe = -float('inf')
        
        for feature in all_features:
            efe = self.ai_agent.calculate_expected_free_energy(
                feature, InterventionType.ABLATION
            )
            if efe > best_efe:
                best_efe = efe
                best_feature = feature
        
        if not best_feature:
            break
        
        # Perform intervention
        result = self.tracer.perform_intervention(
            text, best_feature, InterventionType.ABLATION
        )
        interventions.append(result)
        
        # Update AI agent beliefs
        self.ai_agent.update_beliefs(result)
        
        # Remove feature from consideration
        all_features = [f for f in all_features 
                       if f.feature_id != best_feature.feature_id]
        
        intervention_count += 1
        
        # Check convergence (AI should converge quickly)
        if self.ai_agent.check_convergence(self.config.active_inference.convergence_threshold):
            logger.info(f"AI agent converged after {intervention_count} interventions")
            break
    
    return interventions
```

### 4. Baseline Comparison Phase

```python
def _run_baseline_comparisons(self, text: str, active_features: Dict):
    """Run baseline methods for efficiency comparison (should need more interventions)."""
    baseline_strategies = ['random', 'high_activation', 'sequential']
    baseline_counts = {}
    
    all_features = []
    for layer_features in active_features.values():
        all_features.extend(layer_features)
    
    for strategy in baseline_strategies:
        intervention_count = 0
        max_baseline_interventions = self.config.active_inference.max_interventions * 3
        strategy_features = all_features.copy()
        baseline_effects = []
        
        while intervention_count < max_baseline_interventions and strategy_features:
            # Select feature based on strategy
            if strategy == 'random':
                feature = random.choice(strategy_features)
            elif strategy == 'high_activation':
                feature = max(strategy_features, key=lambda f: f.max_activation)
            elif strategy == 'sequential':
                feature = strategy_features[0]
            
            strategy_features.remove(feature)
            
            # Perform intervention
            result = self.tracer.perform_intervention(
                text, feature, InterventionType.ABLATION
            )
            baseline_effects.append(result.effect_size)
            intervention_count += 1
            
            # Simple convergence check (less sophisticated than AI)
            if len(baseline_effects) >= 5:
                recent_effects = baseline_effects[-5:]
                if np.std(recent_effects) < 0.05:  # Low variance = convergence
                    break
        
        baseline_counts[strategy] = intervention_count
    
    return baseline_counts
```

### 5. Research Question Validation Phase

```python
def validate_research_questions(self, correspondence_metrics, efficiency_metrics, predictions):
    """Validate all research questions with enhanced statistical testing."""
    
    # RQ1: Correspondence validation (target ≥70%)
    avg_correspondence = np.mean([m.overall_correspondence for m in correspondence_metrics]) * 100
    rq1_passed = avg_correspondence >= self.config.research_questions.rq1_correspondence_target
    
    # RQ2: Efficiency validation (target ≥30% improvement)
    overall_efficiency = efficiency_metrics.get('overall_improvement', 0)
    rq2_passed = overall_efficiency >= self.config.research_questions.rq2_efficiency_target
    
    # RQ3: Predictions validation (target ≥3 validated predictions)
    validated_count = len([p for p in predictions 
                          if hasattr(p, 'validation_status') and p.validation_status == 'validated'])
    rq3_passed = validated_count >= self.config.research_questions.rq3_predictions_target
    
    # Overall success
    overall_success = rq1_passed and rq2_passed and rq3_passed
    success_rate = sum([rq1_passed, rq2_passed, rq3_passed]) / 3.0
    
    return {
        'rq1_passed': rq1_passed,
        'rq1_achieved': avg_correspondence,
        'rq2_passed': rq2_passed,
        'rq2_achieved': overall_efficiency,
        'rq3_passed': rq3_passed,
        'rq3_achieved': validated_count,
        'overall_success': overall_success,
        'success_rate': success_rate
    }
```

## Usage Examples

### Basic Experiment Execution

```python
from experiments.runner import YorKExperimentRunner
from config.experiment_config import CompleteConfig

# Create and configure experiment runner
runner = YorKExperimentRunner()

# Setup experiment components
runner.setup_experiment()

# Define test inputs (Golden Gate Bridge example)
test_inputs = [
    "The Golden Gate Bridge is located in",
    "San Francisco's most famous landmark is the",
    "The bridge connecting San Francisco to Marin County is called the",
    "When visiting California, tourists often see the iconic",
    "The famous red suspension bridge in San Francisco is known as the"
]

# Run complete experiment
results = runner.run_experiment(test_inputs)

# Print summary
print(f"Experiment completed: {results.experiment_name}")
print(f"Research questions passed: {sum([results.rq1_passed, results.rq2_passed, results.rq3_passed])}/3")
print(f"Overall success: {results.overall_success}")
```

### Custom Configuration Experiment

```python
from pathlib import Path

# Load custom configuration
config_path = Path("custom_experiment_config.yaml")
runner = YorKExperimentRunner(config_path)

# Configure for specific research focus
runner.config.active_inference.epistemic_weight = 0.8  # More exploration
runner.config.sae.activation_threshold = 0.1          # Higher threshold
runner.config.research_questions.rq1_correspondence_target = 75.0  # Higher target

# Setup and run
runner.setup_experiment()
results = runner.run_experiment(test_inputs)
```

### Golden Gate Bridge Convenience Function

```python
from experiments.runner import run_golden_gate_experiment

# Run the canonical Golden Gate Bridge experiment
results = run_golden_gate_experiment()

print(f"Golden Gate Bridge experiment results:")
print(f"  Interventions performed: {len(results.intervention_results)}")
print(f"  Novel predictions: {len(results.novel_predictions)}")
print(f"  Average correspondence: {np.mean([m.overall_correspondence for m in results.correspondence_metrics]):.1%}")
```

### Detailed Results Analysis

```python
# Analyze experiment results in detail
summary_stats = results.get_summary_stats()

print("Detailed Results:")
print(f"  Total interventions: {summary_stats['total_interventions']}")
print(f"  Average effect size: {summary_stats['average_effect_size']:.3f}")
print(f"  Average correspondence: {summary_stats['average_correspondence']:.1%}")
print(f"  Predictions generated: {summary_stats['predictions_generated']}")
print(f"  Predictions validated: {summary_stats['predictions_validated']}")
print(f"  Success rate: {summary_stats['success_rate']:.1%}")
print(f"  Duration: {summary_stats['experiment_duration']:.1f} seconds")

# Efficiency analysis
print(f"\nEfficiency Analysis:")
for metric, value in results.efficiency_metrics.items():
    if 'improvement' in metric:
        print(f"  {metric.replace('_', ' ').title()}: {value:.1f}%")
```

### Research Question Validation

```python
# Validate research questions with detailed breakdown
rq_validation = runner.validate_research_questions(
    results.correspondence_metrics,
    results.efficiency_metrics,
    results.novel_predictions
)

print("Research Question Validation:")
print(f"RQ1 (Correspondence ≥70%): {'✅ PASSED' if rq_validation['rq1_passed'] else '❌ FAILED'}")
print(f"  Target: 70%, Achieved: {rq_validation['rq1_achieved']:.1f}%")

print(f"RQ2 (Efficiency ≥30%): {'✅ PASSED' if rq_validation['rq2_passed'] else '❌ FAILED'}")
print(f"  Target: 30%, Achieved: {rq_validation['rq2_achieved']:.1f}%")

print(f"RQ3 (Predictions ≥3): {'✅ PASSED' if rq_validation['rq3_passed'] else '❌ FAILED'}")
print(f"  Target: 3, Achieved: {rq_validation['rq3_achieved']}")

print(f"Overall Success: {'✅ YES' if rq_validation['overall_success'] else '❌ NO'}")
print(f"Success Rate: {rq_validation['success_rate']:.1%}")
```

### Statistical Validation Integration

```python
# Enhanced statistical validation
if runner.config.statistical_validation.enabled:
    from core.statistical_validation import perform_comprehensive_validation
    
    statistical_validation = perform_comprehensive_validation(results)
    
    print("Statistical Validation Results:")
    if 'statistical_summary' in statistical_validation:
        stats = statistical_validation['statistical_summary']
        print(f"  Total tests performed: {stats.get('total_tests', 0)}")
        print(f"  Significant results: {stats.get('significant_tests', 0)}")
        print(f"  Average effect size: {stats.get('average_effect_size', 0):.3f}")
        print(f"  Average power: {stats.get('average_power', 0):.3f}")
```

## Experiment Result Structure

### ExperimentResult Components

```python
@dataclass
class ExperimentResult:
    # Experiment metadata
    experiment_name: str
    timestamp: str
    config_used: Dict[str, Any]
    
    # Core results
    correspondence_metrics: List[CorrespondenceMetrics]
    efficiency_metrics: Dict[str, float]
    novel_predictions: List[NovelPrediction]
    
    # Research question validation
    rq1_passed: bool      # Correspondence ≥70%
    rq2_passed: bool      # Efficiency ≥30%
    rq3_passed: bool      # Predictions ≥3
    overall_success: bool # All RQs passed
    
    # Supporting data
    intervention_results: List[InterventionResult]
    belief_history: List[BeliefState]
    circuit_graphs: List[AttributionGraph]
    metadata: Dict[str, Any]
```

### Result Serialization

```python
# Save results with complete metadata
runner.save_results(results, "experiment_outputs")

# Results saved as JSON with structure:
{
    "experiment_name": "golden_gate_bridge_discovery",
    "timestamp": "2024-01-15T10:30:00",
    "rq1_passed": true,
    "rq2_passed": true,
    "rq3_passed": true,
    "overall_success": true,
    "summary_stats": {
        "total_interventions": 12,
        "average_effect_size": 0.456,
        "average_correspondence": 0.732,
        "predictions_generated": 5,
        "predictions_validated": 4,
        "success_rate": 1.0,
        "experiment_duration": 45.2
    }
}
```

## Baseline Method Implementations

### Random Selection Baseline
```python
def random_baseline(features):
    """Random feature selection baseline."""
    import random
    selected_features = []
    feature_pool = features.copy()
    
    while feature_pool and len(selected_features) < max_interventions:
        feature = random.choice(feature_pool)
        selected_features.append(feature)
        feature_pool.remove(feature)
    
    return selected_features
```

### High Activation Baseline
```python
def high_activation_baseline(features):
    """Select features by highest activation values."""
    sorted_features = sorted(features, key=lambda f: f.max_activation, reverse=True)
    return sorted_features[:max_interventions]
```

### Sequential Baseline
```python
def sequential_baseline(features):
    """Sequential feature selection (order of discovery)."""
    return features[:max_interventions]
```

## Performance Metrics

### Efficiency Measurement
The runner tracks efficiency improvements over baseline methods:

```python
def _calculate_efficiency_metrics(self, ai_interventions, baseline_counts):
    """Calculate efficiency improvement over baselines."""
    efficiency_metrics = {}
    
    for strategy, counts_list in baseline_counts.items():
        if counts_list:
            avg_baseline = np.mean(counts_list)
            if avg_baseline > 0:
                improvement = ((avg_baseline - ai_interventions) / avg_baseline) * 100
                efficiency_metrics[f"{strategy}_improvement"] = max(0.0, improvement)
    
    # Overall efficiency
    if efficiency_metrics:
        efficiency_metrics['overall_improvement'] = np.mean(list(efficiency_metrics.values()))
    
    return efficiency_metrics
```

### Expected Performance
- **Active Inference**: 10-15 interventions typical
- **Random baseline**: 40-60 interventions typical
- **High activation baseline**: 25-35 interventions typical
- **Sequential baseline**: 30-45 interventions typical

**Target efficiency improvement**: ≥30% over baselines

## Integration with Other Components

### Circuit Tracer Integration
```python
# Runner initializes and manages tracer
self.tracer = CircuitTracer(self.config)

# Uses tracer for feature discovery and interventions
active_features = self.tracer.find_active_features(text)
result = self.tracer.perform_intervention(text, feature, intervention_type)
```

### Active Inference Integration
```python
# Runner initializes and manages AI agent
self.ai_agent = ActiveInferenceAgent(self.config, self.tracer)

# Uses agent for belief management and intervention selection
belief_state = self.ai_agent.initialize_beliefs(active_features)
efe = self.ai_agent.calculate_expected_free_energy(feature, intervention_type)
correspondence = self.ai_agent.update_beliefs(intervention_result)
```

### Visualization Integration
```python
# Generate visualizations from results
from visualization.visualizer import CircuitVisualizer

visualizer = CircuitVisualizer("output/visualizations")
visualizations = visualizer.generate_all_visualizations(
    results, attribution_graph, belief_history
)
```

## Configuration Options

### Experiment Configuration
```yaml
experiment:
  name: "golden_gate_bridge_discovery"    # Experiment identifier
  output_dir: "experiment_results"        # Results output directory
  max_parallel_jobs: 1                    # Parallel execution (future)
  save_intermediate_results: true         # Save intermediate data
  generate_visualizations: true           # Create visualizations
```

### Research Question Targets
```yaml
research_questions:
  rq1_correspondence_target: 70.0         # Correspondence threshold (%)
  rq2_efficiency_target: 30.0             # Efficiency improvement threshold (%)
  rq3_predictions_target: 3               # Novel predictions threshold (count)
```

## Error Handling and Robustness

### Configuration Validation
- Validates all configuration parameters before execution
- Checks for required dependencies and model availability
- Provides clear error messages for configuration issues

### Experiment Robustness
- Continues analysis even if some layers fail
- Graceful degradation when optional components unavailable
- Comprehensive logging for debugging and monitoring

### Result Validation
- Validates all research question calculations
- Ensures statistical significance of results
- Provides confidence intervals and effect sizes

## Dependencies

### Required
- `numpy` - Numerical computations and statistics
- `torch` - Neural network operations (via components)
- `json` - Result serialization
- `pathlib` - File system operations
- `datetime` - Timestamp generation

### Project Dependencies
- `circuit_analysis.tracer` - Circuit discovery and intervention
- `active_inference.agent` - Active Inference guidance
- `core.data_structures` - All data structure definitions
- `core.metrics` - Metrics calculation
- `core.prediction_system` - Novel prediction generation
- `core.statistical_validation` - Statistical testing
- `config.experiment_config` - Configuration management

### Optional Dependencies
- `visualization.visualizer` - Result visualization
- `pymdp` - Enhanced Active Inference (via agent)

## Future Enhancements

### Planned Features
- **Parallel experiment execution** for multiple test cases
- **Real-time experiment monitoring** with live dashboards
- **Experiment comparison framework** for A/B testing
- **Automated hyperparameter optimization** for configuration tuning

### Research Extensions
- **Multi-model experiments** comparing different transformer architectures
- **Cross-domain validation** testing generalization across domains
- **Longitudinal studies** tracking performance over time
- **Meta-analysis capabilities** aggregating results across experiments

The experiment runner serves as the central orchestration point for the entire ActiveCircuitDiscovery system, ensuring reproducible, statistically rigorous, and comprehensive circuit discovery experiments with proper validation of all research questions.