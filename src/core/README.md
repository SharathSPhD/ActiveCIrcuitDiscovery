# Core Components

This directory contains the foundational components of the ActiveCircuitDiscovery project, providing essential data structures, interfaces, and core functionality for Active Inference-guided circuit discovery.

## Overview

The core module establishes the fundamental architecture for:
- **Data structures** for representing neural circuits, interventions, and experimental results
- **Abstract interfaces** defining contracts for all major components
- **Statistical validation** frameworks for research question verification
- **Prediction systems** for generating and validating novel hypotheses
- **Metrics calculation** for correspondence, efficiency, and validation

## Main Files

### `data_structures.py`
Contains all core data structures used throughout the project:

**Circuit Representation:**
- `SAEFeature` - Represents discovered Sparse Autoencoder features with metadata
- `InterventionResult` - Results from circuit intervention experiments
- `CircuitNode` - Individual nodes in discovered circuit graphs
- `AttributionGraph` - Complete circuit representation with nodes and edges

**Active Inference Components:**
- `BeliefState` - Active Inference agent belief state with pymdp integration
- `CorrespondenceMetrics` - Measures correspondence between AI and circuit behavior
- `NovelPrediction` - Novel predictions generated from Active Inference analysis

**Experimental Framework:**
- `ExperimentResult` - Complete experiment results with all metrics
- `ValidationResult` - Statistical validation results for predictions
- `EfficiencyMetrics` - Comprehensive efficiency measurements

### `interfaces.py`
Defines abstract interfaces following the Interface Segregation Principle:

**Core Interfaces:**
- `ICircuitTracer` - Circuit discovery and analysis operations
- `IActiveInferenceAgent` - Active Inference agent behavior
- `IInterventionStrategy` - Intervention selection strategies
- `IMetricsCalculator` - Research question metrics calculation

**Enhanced Interfaces:**
- `IStatisticalValidator` - Statistical validation operations
- `IPredictionGenerator` - Novel prediction generation
- `IPredictionValidator` - Prediction validation framework
- `IVisualizationGenerator` - Visualization creation

**Factory Interfaces:**
- `ICircuitTracerFactory` - Circuit tracer creation
- `IStrategyFactory` - Strategy pattern implementation
- `IAgentFactory` - Agent instantiation

### `metrics.py`
Implements comprehensive metrics calculation for all research questions:

**Correspondence Calculator:**
- Belief updating correspondence measurement
- Precision weighting alignment assessment
- Prediction error correspondence analysis
- Statistical significance testing

**Efficiency Calculator:**
- Intervention count comparisons
- Improvement percentage calculations
- Bootstrap confidence intervals
- Multiple comparison corrections

**Validation Calculator:**
- Research question validation framework
- Target threshold assessments
- Overall success determination

### `prediction_system.py`
Enhanced prediction generation system:

**Prediction Types:**
- Attention pattern predictions from belief uncertainties
- Feature interaction predictions from connection beliefs
- Failure mode predictions from intervention patterns

**Generation Methods:**
- Uncertainty-based prediction generation
- Connection strength analysis
- Statistical pattern recognition
- Hypothesis formulation

### `prediction_validator.py`
Comprehensive prediction validation framework:

**Validation Methods:**
- Statistical testing for predictions
- Cross-validation approaches
- Effect size calculations
- Confidence interval estimation

**Test Types:**
- Correlation tests for attention patterns
- Mann-Whitney U tests for interactions
- Bootstrap methods for failure modes

### `statistical_validation.py`
Advanced statistical validation system:

**Statistical Tests:**
- Correspondence significance testing
- Efficiency improvement validation
- Prediction success rate analysis
- Multiple comparison corrections

**Features:**
- Bootstrap sampling for robust estimation
- Effect size calculations (Cohen's d)
- Power analysis for adequate sample sizes
- Comprehensive reporting

## Usage Examples

### Creating and Using Data Structures

```python
from core.data_structures import SAEFeature, BeliefState, NovelPrediction
import numpy as np

# Create an SAE feature
feature = SAEFeature(
    feature_id=1234,
    layer=8,
    activation_threshold=0.05,
    description="Golden Gate Bridge feature",
    max_activation=0.78,
    examples=["San Francisco landmark", "red suspension bridge"]
)

# Create a belief state
belief_state = BeliefState(
    qs=np.array([0.7, 0.3]),
    feature_importances={1234: 0.78, 5678: 0.45},
    connection_beliefs={(1234, 5678): 0.65},
    uncertainty={1234: 0.22, 5678: 0.55},
    confidence=0.73
)

# Create a novel prediction
prediction = NovelPrediction(
    prediction_type="attention_pattern",
    description="High uncertainty features will show stronger attention",
    testable_hypothesis="Features with uncertainty > 0.5 have attention weights > 0.3",
    expected_outcome="Positive correlation between uncertainty and attention",
    test_method="Pearson correlation analysis",
    confidence=0.85
)
```

### Using the Metrics System

```python
from core.metrics import CorrespondenceCalculator, EfficiencyCalculator
from core.data_structures import InterventionResult

# Calculate correspondence metrics
calculator = CorrespondenceCalculator()
correspondence = calculator.calculate_correspondence(belief_state, [intervention_result])

print(f"Overall correspondence: {correspondence.overall_correspondence:.1%}")
print(f"Belief updating: {correspondence.belief_updating_correspondence:.1%}")

# Calculate efficiency improvements
efficiency_calc = EfficiencyCalculator()
efficiency = efficiency_calc.calculate_efficiency(
    ai_interventions=12,
    baseline_results={"random": 45, "exhaustive": 80}
)

print(f"Efficiency improvement: {efficiency['overall_improvement']:.1f}%")
```

### Statistical Validation

```python
from core.statistical_validation import StatisticalValidator

validator = StatisticalValidator()

# Validate correspondence significance
correspondence_validation = validator.validate_correspondence_significance(
    correspondence_metrics, target_threshold=70.0
)

print(f"Correspondence significant: {correspondence_validation['significant']}")
print(f"P-value: {correspondence_validation['p_value']:.4f}")
print(f"Effect size: {correspondence_validation['effect_size']:.3f}")
```

### Prediction Generation and Validation

```python
from core.prediction_system import EnhancedPredictionGenerator
from core.prediction_validator import PredictionValidator, ValidationConfig

# Generate predictions
generator = EnhancedPredictionGenerator()
predictions = generator.generate_circuit_predictions(belief_state, attribution_graph)

# Validate predictions
config = ValidationConfig(significance_level=0.05, min_sample_size=10)
validator = PredictionValidator(config)

for prediction in predictions:
    result = validator.validate_prediction(prediction, test_data)
    print(f"Prediction: {prediction.description}")
    print(f"Status: {result.validation_status}")
    print(f"Confidence: {result.confidence:.3f}")
```

## Design Principles

### Interface Segregation
Each interface focuses on a specific aspect of functionality, making the system modular and testable.

### Data Validation
All data structures include comprehensive validation to ensure data integrity and catch errors early.

### Statistical Rigor
Enhanced statistical validation ensures research conclusions are statistically sound with proper significance testing and effect size calculations.

### Extensibility
The modular design allows easy addition of new prediction types, validation methods, and statistical tests.

## Dependencies

### Required
- `numpy` - Numerical computations
- `scipy` - Statistical functions
- `dataclasses` - Data structure definitions
- `typing` - Type annotations

### Optional
- `pymdp` - Active Inference implementation
- `torch` - Neural network operations

## Configuration

The core components are configured through the main experiment configuration system. Key settings include:

- Statistical significance levels
- Bootstrap sample sizes
- Validation thresholds
- Confidence intervals

## Error Handling

The module includes comprehensive error handling with custom exception classes:

- `CircuitDiscoveryError` - Base exception
- `InterventionError` - Intervention-specific errors
- `ActiveInferenceError` - Active Inference errors
- `ValidationError` - Validation errors

## Testing

Core components include extensive unit tests covering:
- Data structure validation
- Interface contract verification
- Statistical method correctness
- Edge case handling

Run tests with:
```bash
python -m pytest tests/test_core.py
```

## Integration

The core module integrates seamlessly with other project components:
- Circuit tracers use core data structures
- Active Inference agents implement core interfaces
- Experiment runners use metrics calculators
- Visualizers display core data structures

This foundation ensures consistency, reliability, and statistical rigor across the entire ActiveCircuitDiscovery system.