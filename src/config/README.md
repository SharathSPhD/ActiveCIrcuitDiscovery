# Configuration Management

This directory contains the comprehensive configuration system for the ActiveCircuitDiscovery project. It provides type-safe, validated configuration management with support for YAML files, environment variables, and programmatic configuration.

## Overview

The configuration system provides:
- **Type-safe configuration** using dataclasses and enums
- **Hierarchical configuration** with logical grouping
- **YAML file support** for easy configuration management
- **Validation system** to catch configuration errors early
- **Default configurations** for quick setup
- **Enhanced configurations** for advanced features

## Main Files

### `experiment_config.py`
The comprehensive configuration system with all configuration classes and management functionality:

**Configuration Classes:**
- `ModelConfig` - Transformer model configuration
- `SAEConfig` - Sparse Autoencoder analysis settings
- `ActiveInferenceConfig` - Active Inference agent parameters
- `ExperimentConfig` - Experiment execution settings
- `ResearchQuestionConfig` - Research question validation targets
- `StatisticalValidationConfig` - Statistical testing parameters
- `PredictionValidationConfig` - Prediction validation settings
- `VisualizationConfig` - Visualization generation options
- `LoggingConfig` - Logging configuration

**Management Classes:**
- `ConfigManager` - Configuration loading, saving, and validation
- `CompleteConfig` - Root configuration containing all subsections

### `default_config.yaml`
Default configuration file providing sensible defaults for all parameters:

```yaml
model:
  name: "gpt2-small"
  device: "auto"
  max_context_length: 1024
  use_cache: true

sae:
  enabled: true
  auto_discover_layers: true
  target_layers: []
  layer_search_range: [0, -1]
  activation_threshold: 0.05
  max_features_per_layer: 20
  neuronpedia_source: true

active_inference:
  enabled: true
  epistemic_weight: 0.7
  exploration_weight: 0.6
  convergence_threshold: 0.15
  max_interventions: 20
  use_pymdp: true

# ... additional configurations
```

## Configuration Sections

### Model Configuration

Controls transformer model loading and execution:

```python
@dataclass
class ModelConfig:
    name: str = "gpt2-small"                    # Model name for TransformerLens
    device: DeviceType = DeviceType.AUTO        # Device selection (auto/cpu/cuda)
    max_context_length: int = 1024              # Maximum input sequence length
    use_cache: bool = True                      # Enable activation caching
```

**Supported Models:**
- `gpt2-small` - GPT-2 Small (117M parameters)
- `gpt2-medium` - GPT-2 Medium (345M parameters)
- `gpt2-large` - GPT-2 Large (774M parameters)
- `gpt2-xl` - GPT-2 XL (1.5B parameters)

**Device Types:**
- `AUTO` - Automatically select CUDA if available, else CPU
- `CPU` - Force CPU execution
- `CUDA` - Force CUDA execution

### SAE Configuration

Controls Sparse Autoencoder analysis and feature discovery:

```python
@dataclass
class SAEConfig:
    enabled: bool = True                        # Enable SAE analysis
    auto_discover_layers: bool = True           # Auto-discover active layers
    target_layers: List[int] = []               # Manual layer specification
    layer_search_range: List[int] = [0, -1]     # Layer search range for auto-discovery
    activation_threshold: float = 0.05          # Feature activation threshold
    max_features_per_layer: int = 20            # Maximum features per layer
    neuronpedia_source: bool = True             # Use Neuronpedia SAEs when available
```

**Key Features:**
- **Auto-discovery**: Automatically finds the most active layers
- **Manual selection**: Specify exact layers to analyze
- **Threshold control**: Filter features by activation strength
- **Source flexibility**: Use real SAEs or fallback analyzers

### Active Inference Configuration

Controls the Active Inference agent behavior:

```python
@dataclass
class ActiveInferenceConfig:
    enabled: bool = True                        # Enable Active Inference guidance
    epistemic_weight: float = 0.7               # Weight for information gain (0-1)
    exploration_weight: float = 0.6             # Weight for exploration (0-1)
    convergence_threshold: float = 0.15         # Convergence detection threshold
    max_interventions: int = 20                 # Maximum interventions per experiment
    use_pymdp: bool = True                      # Enable pymdp integration
```

**Parameter Guidelines:**
- **Epistemic weight**: Higher values favor exploration and information gain
- **Exploration weight**: Controls exploration vs exploitation balance
- **Convergence threshold**: Lower values require more stable convergence
- **Max interventions**: Safety limit to prevent infinite loops

### Research Question Configuration

Sets targets for research question validation:

```python
@dataclass
class ResearchQuestionConfig:
    rq1_correspondence_target: float = 70.0     # RQ1: Correspondence target (%)
    rq2_efficiency_target: float = 30.0         # RQ2: Efficiency improvement target (%)
    rq3_predictions_target: int = 3             # RQ3: Novel predictions target (count)
```

**Research Questions:**
- **RQ1**: Does Active Inference correspond to circuit behavior? (Target: ≥70% correspondence)
- **RQ2**: Does Active Inference improve efficiency? (Target: ≥30% improvement)
- **RQ3**: Can Active Inference generate novel insights? (Target: ≥3 validated predictions)

### Statistical Validation Configuration

Controls enhanced statistical testing:

```python
@dataclass
class StatisticalValidationConfig:
    enabled: bool = True                        # Enable statistical validation
    significance_level: float = 0.05            # Alpha level for hypothesis testing
    bootstrap_samples: int = 10000              # Bootstrap samples for robust estimation
    confidence_level: float = 0.95              # Confidence interval level
    multiple_comparisons_correction: str = "bonferroni"  # Multiple comparison correction
```

**Correction Methods:**
- `bonferroni` - Conservative Bonferroni correction
- `holm` - Step-down Holm method
- `fdr_bh` - Benjamini-Hochberg FDR control
- `none` - No correction (not recommended)

## Usage Examples

### Basic Configuration Loading

```python
from config.experiment_config import get_config, CompleteConfig

# Load default configuration
config = get_config()

print(f"Model: {config.model.name}")
print(f"Device: {config.model.device.value}")
print(f"SAE enabled: {config.sae.enabled}")
print(f"Auto-discover layers: {config.sae.auto_discover_layers}")
print(f"Epistemic weight: {config.active_inference.epistemic_weight}")
```

### Custom Configuration Creation

```python
# Create custom configuration
config = CompleteConfig()

# Modify model settings
config.model.name = "gpt2-medium"
config.model.device = DeviceType.CUDA

# Modify SAE settings
config.sae.auto_discover_layers = False
config.sae.target_layers = [6, 8, 10, 12]
config.sae.activation_threshold = 0.1

# Modify Active Inference settings
config.active_inference.epistemic_weight = 0.8  # More exploration
config.active_inference.max_interventions = 15

# Modify research question targets
config.research_questions.rq1_correspondence_target = 75.0
config.research_questions.rq2_efficiency_target = 35.0

print("Custom configuration created")
```

### Configuration from YAML File

```python
from pathlib import Path
from config.experiment_config import ConfigManager

# Load from custom YAML file
config_path = Path("custom_config.yaml")
manager = ConfigManager(config_path)
config = manager.load_config()

print(f"Loaded configuration from: {config_path}")
```

### Configuration Validation

```python
from config.experiment_config import ConfigManager

# Validate configuration
manager = ConfigManager()
config = CompleteConfig()

# Modify with invalid values
config.active_inference.epistemic_weight = 1.5  # Invalid: must be ≤ 1.0
config.sae.activation_threshold = -0.1           # Invalid: must be > 0

# Validate
issues = manager.validate_config(config)

if issues:
    print("Configuration validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid")
```

### Enhanced Configuration for Advanced Features

```python
from config.experiment_config import get_enhanced_config

# Load enhanced configuration with all advanced features
enhanced_config = get_enhanced_config()

print("Enhanced features enabled:")
print(f"  Statistical validation: {enhanced_config.statistical_validation.enabled}")
print(f"  Prediction validation: {enhanced_config.prediction_validation.enabled}")
print(f"  Interactive dashboards: {enhanced_config.visualization.interactive_dashboards}")
print(f"  Publication ready plots: {enhanced_config.visualization.publication_ready}")
```

### Saving Configuration

```python
from config.experiment_config import ConfigManager

# Create and save configuration
config = CompleteConfig()
config.experiment.name = "my_custom_experiment"
config.model.name = "gpt2-large"

manager = ConfigManager(Path("my_config.yaml"))
manager.save_config(config)

print("Configuration saved to my_config.yaml")
```

## Configuration File Format

### Example YAML Configuration

```yaml
model:
  name: "gpt2-small"
  device: "auto"
  max_context_length: 1024
  use_cache: true

sae:
  enabled: true
  auto_discover_layers: true
  target_layers: []
  layer_search_range: [0, -1]
  activation_threshold: 0.05
  max_features_per_layer: 20
  neuronpedia_source: true

active_inference:
  enabled: true
  epistemic_weight: 0.7
  exploration_weight: 0.6
  convergence_threshold: 0.15
  max_interventions: 20
  use_pymdp: true

experiment:
  name: "golden_gate_bridge_discovery"
  output_dir: "experiment_results"
  max_parallel_jobs: 1
  save_intermediate_results: true
  generate_visualizations: true

research_questions:
  rq1_correspondence_target: 70.0
  rq2_efficiency_target: 30.0
  rq3_predictions_target: 3

logging:
  level: "INFO"
  file_output: true
  console_output: true
  max_file_size_mb: 10

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

## Environment Variable Support

Configuration values can be overridden using environment variables:

```bash
# Model configuration
export ACDISCOVERY_MODEL_NAME="gpt2-medium"
export ACDISCOVERY_MODEL_DEVICE="cuda"

# SAE configuration
export ACDISCOVERY_SAE_ACTIVATION_THRESHOLD="0.1"
export ACDISCOVERY_SAE_AUTO_DISCOVER_LAYERS="false"

# Active Inference configuration
export ACDISCOVERY_AI_EPISTEMIC_WEIGHT="0.8"
export ACDISCOVERY_AI_MAX_INTERVENTIONS="15"

# Run experiment with environment overrides
python experiments/golden_gate_bridge.py
```

## Validation Rules

The configuration system includes comprehensive validation:

### Model Validation
- Model name must be supported by TransformerLens
- Device type must be valid enum value
- Context length must be positive integer

### SAE Validation
- Activation threshold must be between 0 and 1
- Layer search range must be valid (start ≤ end)
- Target layers cannot be empty when auto-discovery disabled
- Max features per layer must be positive

### Active Inference Validation
- Epistemic weight must be between 0 and 1
- Exploration weight must be between 0 and 1
- Convergence threshold must be positive
- Max interventions must be positive

### Research Question Validation
- All targets must be positive values
- Correspondence target typically between 50-100%
- Efficiency target typically between 10-100%
- Predictions target typically between 1-10

## Best Practices

### Configuration Management
1. **Use version control**: Keep configuration files in version control
2. **Environment-specific configs**: Create separate configs for different environments
3. **Validation first**: Always validate configuration before running experiments
4. **Documentation**: Document custom configuration choices

### Parameter Selection
1. **Conservative defaults**: Start with provided defaults
2. **Gradual tuning**: Adjust parameters incrementally
3. **Validation feedback**: Use validation results to guide parameter selection
4. **Domain knowledge**: Incorporate knowledge about the specific research domain

### File Organization
```
config/
├── default_config.yaml           # Default configuration
├── enhanced_config.yaml          # Enhanced features configuration
├── gpt2_small_config.yaml        # GPT-2 Small specific settings
├── gpt2_large_config.yaml        # GPT-2 Large specific settings
└── custom_experiments/           # Custom experiment configurations
    ├── golden_gate_config.yaml
    └── attention_heads_config.yaml
```

## Integration with Other Components

### Circuit Tracer Integration
```python
# Configuration automatically passed to tracer
tracer = CircuitTracer(config)
# Tracer uses config.model and config.sae sections
```

### Active Inference Integration
```python
# Configuration controls agent behavior
agent = ActiveInferenceAgent(config, tracer)
# Agent uses config.active_inference section
```

### Experiment Runner Integration
```python
# Configuration controls entire experiment
runner = YorKExperimentRunner(config_path)
# Runner uses all configuration sections
```

## Troubleshooting

### Common Configuration Issues

1. **Invalid device specification**
   ```
   Error: Device 'invalid' not recognized
   Solution: Use 'auto', 'cpu', or 'cuda'
   ```

2. **SAE layer configuration conflict**
   ```
   Error: target_layers cannot be empty when auto_discover_layers is False
   Solution: Set auto_discover_layers=True or specify target_layers
   ```

3. **Parameter out of range**
   ```
   Error: epistemic_weight must be between 0 and 1
   Solution: Use values in valid range: epistemic_weight: 0.7
   ```

4. **YAML syntax errors**
   ```
   Error: YAML parsing failed
   Solution: Check indentation and syntax using YAML validator
   ```

### Debug Configuration Issues
```python
# Enable verbose logging for configuration
import logging
logging.getLogger('config').setLevel(logging.DEBUG)

# Load and validate configuration
config = get_config()
issues = validate_config(config)
```

## Dependencies

### Required
- `dataclasses` - Configuration structure definition
- `typing` - Type annotations and validation
- `pathlib` - File path handling
- `yaml` - YAML file parsing
- `enum` - Enumeration types

### Project Dependencies
All other project modules depend on the configuration system for their settings and parameters.

The configuration system provides the foundation for reproducible, validated, and flexible experiment setup across the entire ActiveCircuitDiscovery project.