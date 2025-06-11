# Circuit Analysis

This directory contains the circuit discovery and analysis components that form the foundation of the ActiveCircuitDiscovery system. It implements advanced circuit tracing techniques integrated with Sparse Autoencoder (SAE) analysis for mechanistic interpretability of transformer models.

## Overview

The circuit analysis module provides:
- **Automated circuit discovery** across transformer layers
- **SAE feature extraction** with auto-discovery capabilities
- **Intervention mechanisms** for causal circuit validation
- **Attribution graph construction** for circuit visualization
- **Integration with circuit-tracer patterns** for compatibility with existing tools

## Main Files

### `tracer.py`
The core `CircuitTracer` class implementing the `ICircuitTracer` interface with enhanced capabilities:

**Key Features:**
- **Auto-layer discovery**: Automatically identifies the most active layers for analysis
- **SAE integration**: Works with both real SAE models and fallback analyzers
- **Multiple intervention types**: Ablation, activation patching, and mean ablation
- **Hierarchical analysis**: Builds complete attribution graphs across layers
- **Device-aware**: Supports CPU/CUDA with automatic device resolution

**Core Methods:**
- `find_active_features()` - Discovers active SAE features above threshold
- `perform_intervention()` - Executes causal interventions on specific features
- `build_attribution_graph()` - Constructs complete circuit representation
- `get_feature_activations()` - Extracts feature activations for specific layers

## Architecture

### SAE Integration
The tracer integrates with Sparse Autoencoders through multiple pathways:

1. **SAE-Lens Integration**: Uses real SAE models when available
2. **Fallback SAE**: Creates synthetic analyzers when SAE-Lens unavailable
3. **Auto-discovery**: Automatically finds the most active layers for analysis

### Layer Discovery Algorithm
```python
def _auto_discover_active_layers(self):
    """Auto-discover layers with significant activity."""
    # Analyze activity across all layers
    for layer in range(n_layers):
        # Calculate variance in layer activations
        activity = torch.var(activations).item()
        layer_activities[layer] = activity
    
    # Select top 6-8 most active layers
    return sorted_top_layers
```

### Intervention Types

**Ablation (`InterventionType.ABLATION`)**
- Zeros out specific feature activations
- Measures direct causal impact
- Most common intervention type

**Activation Patching (`InterventionType.ACTIVATION_PATCHING`)**
- Replaces corrupted activations with clean ones
- Tests feature necessity
- Used for causal validation

**Mean Ablation (`InterventionType.MEAN_ABLATION`)**
- Replaces with dataset mean activation
- Controls for activation magnitude
- Currently approximated as ablation

## Usage Examples

### Basic Circuit Discovery

```python
from circuit_analysis.tracer import CircuitTracer
from config.experiment_config import CompleteConfig, InterventionType

# Initialize tracer with auto-discovery
config = CompleteConfig()
config.sae.auto_discover_layers = True
config.sae.activation_threshold = 0.05

tracer = CircuitTracer(config)

# Discover active features
text = "The Golden Gate Bridge is located in"
active_features = tracer.find_active_features(text)

print(f"Found features in {len(active_features)} layers:")
for layer, features in active_features.items():
    print(f"  Layer {layer}: {len(features)} features")
    for feature in features[:3]:  # Show top 3
        print(f"    Feature {feature.feature_id}: {feature.max_activation:.3f}")
```

### Performing Interventions

```python
# Select a feature for intervention
layer = list(active_features.keys())[0]
target_feature = active_features[layer][0]

# Perform ablation intervention
result = tracer.perform_intervention(
    text=text,
    feature=target_feature,
    intervention_type=InterventionType.ABLATION
)

print(f"Intervention Results:")
print(f"  Effect size: {result.effect_size:.3f}")
print(f"  Target token change: {result.target_token_change:.3f}")
print(f"  Logit difference L2: {result.logit_diff_l2:.3f}")
print(f"  KL divergence: {result.kl_divergence:.3f}")
```

### Building Attribution Graphs

```python
# Build complete attribution graph
attribution_graph = tracer.build_attribution_graph(text)

print(f"Attribution Graph:")
print(f"  Nodes: {len(attribution_graph.nodes)}")
print(f"  Edges: {len(attribution_graph.edges)}")
print(f"  Confidence: {attribution_graph.confidence:.3f}")
print(f"  Target output: '{attribution_graph.target_output}'")

# Analyze graph structure
for node_id, node in attribution_graph.nodes.items():
    print(f"  Node {node_id}: Feature {node.feature.feature_id} "
          f"(Layer {node.feature.layer}, Influence: {node.causal_influence:.3f})")
```

### Custom Configuration

```python
# Configure specific layers and thresholds
config = CompleteConfig()
config.sae.enabled = True
config.sae.auto_discover_layers = False  # Manual layer selection
config.sae.target_layers = [6, 8, 10]   # Specific layers
config.sae.activation_threshold = 0.1   # Higher threshold
config.sae.max_features_per_layer = 10  # Limit features

tracer = CircuitTracer(config)
```

## Advanced Features

### Device Management
The tracer automatically handles device placement:

```python
# Auto-detection
config.model.device = DeviceType.AUTO  # Chooses CUDA if available

# Manual specification
config.model.device = DeviceType.CPU   # Force CPU
config.model.device = DeviceType.CUDA  # Force CUDA
```

### SAE Fallback System
When SAE-Lens is unavailable, the tracer creates fallback analyzers:

```python
def _create_fallback_analyzer(self, layer: int):
    """Create fallback SAE for layer without real SAE."""
    d_model = self.model.cfg.d_model
    n_features = d_model * 4  # Expansion factor
    
    # Create random encoder/decoder matrices
    encoder = torch.randn(d_model, n_features) * 0.1
    decoder = torch.randn(n_features, d_model) * 0.1
    
    return {
        'type': 'fallback',
        'encoder': encoder,
        'decoder': decoder
    }
```

### Attribution Calculation
The system calculates feature attributions through correlation analysis:

```python
def _calculate_pairwise_attribution(self, source_feature, target_feature):
    """Calculate attribution between features in different layers."""
    # Get activations for both features
    source_acts = self.get_feature_activations(text, source_feature.layer)
    target_acts = self.get_feature_activations(text, target_feature.layer)
    
    # Calculate correlation-based attribution
    source_val = source_acts[source_feature.feature_id]
    target_val = target_acts[target_feature.feature_id]
    
    return source_val * target_val / (source_feature.max_activation + 1e-8)
```

## Integration with Other Components

### Active Inference Agent
The tracer works seamlessly with the Active Inference agent:

```python
# Agent uses tracer for interventions
ai_agent = ActiveInferenceAgent(config, tracer)
belief_state = ai_agent.initialize_beliefs(active_features)

# Agent selects interventions based on Expected Free Energy
best_feature = ai_agent.select_intervention(active_features)
result = tracer.perform_intervention(text, best_feature, InterventionType.ABLATION)

# Agent updates beliefs based on intervention results
correspondence = ai_agent.update_beliefs(result)
```

### Visualization System
The tracer outputs are directly compatible with the visualization system:

```python
from visualization.visualizer import CircuitVisualizer

visualizer = CircuitVisualizer("output/visualizations")
circuit_diagram = visualizer.create_circuit_diagram(
    attribution_graph, "golden_gate_circuit"
)
```

## Configuration Options

### SAE Configuration
```yaml
sae:
  enabled: true
  auto_discover_layers: true           # Auto-find active layers
  target_layers: []                    # Manual layer specification
  layer_search_range: [0, -1]         # Search range for auto-discovery
  activation_threshold: 0.05           # Feature activation threshold
  max_features_per_layer: 20           # Limit features per layer
  neuronpedia_source: true             # Use Neuronpedia SAEs if available
```

### Model Configuration
```yaml
model:
  name: "gpt2-small"                   # Model name for TransformerLens
  device: "auto"                       # Device selection
  max_context_length: 1024             # Maximum input length
  use_cache: true                      # Enable activation caching
```

## Performance Optimization

### Caching Strategy
- Activation caching reduces redundant forward passes
- Feature activation memoization for repeated access
- Device-optimized tensor operations

### Memory Management
- Automatic cleanup of large activation tensors
- Streaming analysis for long sequences
- Configurable batch sizes for memory control

### Parallel Processing
- GPU acceleration when available
- Vectorized operations for feature analysis
- Efficient sparse tensor handling for SAE operations

## Error Handling and Robustness

### Graceful Degradation
- Falls back to synthetic SAEs when real ones unavailable
- Continues analysis even if some layers fail
- Provides meaningful error messages and warnings

### Validation
- Input validation for all public methods
- Tensor shape and device consistency checks
- Configuration validation before initialization

### Logging
Comprehensive logging for debugging and monitoring:
- Layer discovery progress
- SAE loading status
- Intervention execution details
- Performance metrics

## Dependencies

### Required
- `torch` - Neural network operations and tensor handling
- `numpy` - Numerical computations
- `transformer_lens` - Transformer model loading and analysis

### Optional
- `sae_lens` - Real SAE model integration
- `circuitsvis` - Advanced circuit visualization

### Project Dependencies
- `core.interfaces` - Interface definitions
- `core.data_structures` - Data structure definitions
- `config.experiment_config` - Configuration management

## Testing

The circuit analysis module includes comprehensive tests:

```bash
# Run circuit analysis tests
python -m pytest tests/test_circuit_analysis.py

# Test with specific model
python -m pytest tests/test_circuit_analysis.py::test_gpt2_small

# Test intervention mechanisms
python -m pytest tests/test_circuit_analysis.py::test_interventions
```

## Future Enhancements

### Planned Features
- Multi-token sequence analysis
- Cross-attention circuit discovery
- Hierarchical feature decomposition
- Real-time intervention streaming

### Research Extensions
- Causal mediation analysis
- Information flow quantification
- Circuit robustness testing
- Transfer learning circuit analysis

The circuit analysis module provides the foundational capability for discovering and analyzing neural circuits in transformer models, enabling the Active Inference system to guide efficient and scientifically rigorous mechanistic interpretability research.