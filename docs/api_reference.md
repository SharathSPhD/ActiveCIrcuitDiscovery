# ActiveCircuitDiscovery API Reference

## Overview

ActiveCircuitDiscovery provides a comprehensive framework for circuit discovery in Large Language Models using Active Inference principles. This API reference covers all major components and their usage.

## Core Components

### src.circuit_tracer

#### RealCircuitTracer

The main class for circuit discovery using Sparse Autoencoders (SAEs) and activation patching.

```python
from src.circuit_tracer import RealCircuitTracer

tracer = RealCircuitTracer(model_name="gpt2-small", device="cuda")
```

**Parameters:**
- `model_name` (str): Name of the transformer model ("gpt2-small", "gpt2-medium", etc.)
- `device` (str): Device to use ("cuda", "cpu", or "auto")

**Key Methods:**

##### `find_active_features(text: str, threshold: float = 0.1)`

Find interpretable features that activate above threshold for given text.

```python
active_features = tracer.find_active_features("The Golden Gate Bridge", threshold=0.05)
# Returns: Dict[int, List[SAEFeature]] - layer -> list of active features
```

##### `perform_ablation(text: str, target_feature: SAEFeature)`

Ablate a specific feature and measure the effect on model output.

```python
result = tracer.perform_ablation(text, feature)
# Returns: InterventionResult with effect_size, target_token_change, etc.
```

##### `perform_activation_patching(source_text: str, target_text: str, target_feature: SAEFeature)`

Patch activation from source to target context.

```python
result = tracer.perform_activation_patching(source_text, target_text, feature)
# Returns: InterventionResult
```

##### `build_attribution_graph(text: str)`

Build complete attribution graph showing causal relationships between features.

```python
graph = tracer.build_attribution_graph("The Golden Gate Bridge is located in")
# Returns: AttributionGraph with nodes, edges, confidence
```

### src.active_inference

#### ActiveInferenceGuide

Implements Active Inference principles for guided circuit discovery.

```python
from src.active_inference import ActiveInferenceGuide

ai_guide = ActiveInferenceGuide(tracer, exploration_weight=0.6, epistemic_weight=0.7)
```

**Parameters:**
- `tracer`: RealCircuitTracer instance
- `exploration_weight` (float): Weight for exploration vs exploitation
- `epistemic_weight` (float): Weight for epistemic vs pragmatic value

**Key Methods:**

##### `initialize_beliefs(text: str)`

Initialize belief state about circuit structure for given text.

```python
belief_state = ai_guide.initialize_beliefs("The Golden Gate Bridge")
# Returns: BeliefState with feature importance beliefs and uncertainties
```

##### `select_next_intervention(available_features: List[SAEFeature])`

Select next intervention using Expected Free Energy minimization.

```python
intervention_plan = ai_guide.select_next_intervention(active_features)
# Returns: InterventionPlan with target feature and expected free energy
```

##### `update_beliefs(intervention_result: InterventionResult)`

Update beliefs based on intervention results using Bayesian inference.

```python
ai_guide.update_beliefs(result)
# Updates internal belief state
```

##### `generate_circuit_hypothesis()`

Generate circuit hypothesis from current beliefs.

```python
hypothesis = ai_guide.generate_circuit_hypothesis()
# Returns: CircuitHypothesis with core/optional features and connections
```

#### Comparison Functions

##### `compare_intervention_strategies(tracer, text: str, max_interventions: int = 20)`

Compare Active Inference vs baseline intervention strategies.

```python
from src.active_inference import compare_intervention_strategies

results = compare_intervention_strategies(tracer, text, max_interventions=10)
# Returns: Dict with results for each strategy
```

### src.visualizer

#### CircuitVisualizer

Creates interactive visualizations of discovered circuits.

```python
from src.visualizer import CircuitVisualizer

visualizer = CircuitVisualizer(output_dir="visualizations")
```

**Key Methods:**

##### `visualize_feature_activations(text: str, active_features: Dict, save_name: str)`

Create feature activation heatmap.

```python
path = visualizer.visualize_feature_activations(text, active_features, "features")
# Returns: Path to HTML visualization
```

##### `visualize_attribution_graph(graph: AttributionGraph, save_name: str)`

Create network visualization of attribution graph.

```python
path = visualizer.visualize_attribution_graph(graph, "circuit_graph")
# Returns: Path to HTML visualization
```

##### `generate_complete_report(tracer, ai_guide, comparison_results, text: str)`

Generate comprehensive visualization report.

```python
report_path = visualizer.generate_complete_report(tracer, ai_guide, results, text)
# Returns: Path to complete HTML report with all visualizations
```

### src.experiment

#### CompleteExperimentRunner

Orchestrates complete experimental validation.

```python
from src.experiment import CompleteExperimentRunner

runner = CompleteExperimentRunner(output_dir="experiment_results")
```

**Key Methods:**

##### `setup_experiment()`

Initialize all experiment components.

```python
runner.setup_experiment()
```

##### `run_golden_gate_bridge_experiment()`

Run complete Golden Gate Bridge circuit discovery experiment.

```python
results = runner.run_golden_gate_bridge_experiment()
# Returns: Complete experimental results with research question validation
```

## Data Structures

### SAEFeature

Represents an interpretable feature from a Sparse Autoencoder.

```python
@dataclass
class SAEFeature:
    feature_id: int
    layer: int
    activation_threshold: float
    description: str
    max_activation: float
    examples: List[str]
```

### InterventionResult

Results from an intervention experiment (ablation or patching).

```python
@dataclass
class InterventionResult:
    intervention_type: str
    target_feature: SAEFeature
    original_logits: torch.Tensor
    intervened_logits: torch.Tensor
    effect_size: float
    target_token_change: float
    intervention_layer: int
```

### AttributionGraph

Complete graph of causal relationships between features.

```python
@dataclass
class AttributionGraph:
    input_text: str
    nodes: Dict[int, CircuitNode]
    edges: Dict[Tuple[int, int], float]
    target_output: str
    confidence: float
```

### BeliefState

Current beliefs about circuit structure in Active Inference.

```python
@dataclass
class BeliefState:
    feature_importances: Dict[int, float]
    connection_beliefs: Dict[Tuple[int, int], float]
    uncertainty: Dict[int, float]
    confidence: float
```

## Usage Examples

### Basic Circuit Discovery

```python
from src.circuit_tracer import RealCircuitTracer
from src.active_inference import ActiveInferenceGuide

# Initialize components
tracer = RealCircuitTracer(device="cuda")
ai_guide = ActiveInferenceGuide(tracer)

# Discover circuit
text = "The Golden Gate Bridge is located in"
ai_guide.initialize_beliefs(text)
graph = tracer.build_attribution_graph(text)

print(f"Discovered {len(graph.nodes)} nodes with confidence {graph.confidence:.3f}")
```

### Active Inference Guided Discovery

```python
# Initialize beliefs
ai_guide.initialize_beliefs(text)
active_features = tracer.find_active_features(text)

# Guided discovery loop
for iteration in range(10):
    # Select intervention using Expected Free Energy
    available = [f for layer_features in active_features.values() 
                 for f in layer_features]
    plan = ai_guide.select_next_intervention(available)
    
    if not plan:
        break
        
    # Execute intervention
    if plan.intervention_type == "ablation":
        result = tracer.perform_ablation(text, plan.target_feature)
    
    # Update beliefs
    ai_guide.update_beliefs(result)
    
    if ai_guide.check_convergence():
        break

# Generate final hypothesis
hypothesis = ai_guide.generate_circuit_hypothesis()
```

### Comparative Analysis

```python
from src.active_inference import compare_intervention_strategies

# Compare Active Inference vs baselines
results = compare_intervention_strategies(tracer, text, max_interventions=15)

print("Strategy Comparison:")
for strategy, metrics in results.items():
    print(f"{strategy}: {metrics['efficiency']:.3f} efficiency")
```

### Visualization

```python
from src.visualizer import CircuitVisualizer

visualizer = CircuitVisualizer("output_visualizations")

# Create individual visualizations
feature_viz = visualizer.visualize_feature_activations(text, active_features, "features")
graph_viz = visualizer.visualize_attribution_graph(graph, "circuit")

# Generate complete report
report = visualizer.generate_complete_report(tracer, ai_guide, results, text)
print(f"Complete report: {report}")
```

## Error Handling

The library provides graceful fallbacks for missing dependencies:

```python
# Check component availability
from src import COMPONENTS_AVAILABLE

if not COMPONENTS_AVAILABLE:
    print("Some optional dependencies missing")
    # Fallback to basic functionality
```

## Configuration

### Device Selection

```python
# Auto-select best available device
tracer = RealCircuitTracer(device="auto")

# Force specific device
tracer = RealCircuitTracer(device="cpu")  # For CPU-only environments
tracer = RealCircuitTracer(device="cuda") # For GPU acceleration
```

### Logging Configuration

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific logging
logger = logging.getLogger('src.circuit_tracer')
logger.setLevel(logging.DEBUG)
```

## Performance Considerations

### Memory Management

- Use `torch.cuda.empty_cache()` between interventions for GPU
- Consider batch processing for multiple inputs
- SAE loading is memory-intensive - monitor GPU memory

### Computational Efficiency

- Start with smaller thresholds for active feature discovery
- Limit max_interventions for faster experimentation
- Use CPU for development, GPU for production runs

## Extension Points

### Custom Features

```python
# Define custom interpretable features
custom_feature = SAEFeature(
    feature_id=9999,
    layer=8,
    activation_threshold=0.4,
    description="Custom bridge concept",
    max_activation=0.0,
    examples=["suspension bridge", "cable bridge"]
)

# Add to tracer's feature database
tracer.feature_database[9999] = custom_feature
```

### Custom Intervention Types

```python
# Extend intervention types in active_inference.py
def custom_intervention(self, text, feature):
    # Implement custom intervention logic
    return InterventionResult(...)
```

This API reference provides comprehensive coverage of all major components in ActiveCircuitDiscovery. For additional examples and tutorials, see the main README.md and experiment files.