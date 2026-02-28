# ActiveCircuitDiscovery

**An Enhanced Active Inference Approach to Circuit Discovery in Large Language Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive research implementation demonstrating how Active Inference principles can guide efficient circuit discovery in transformer architectures, with enhanced statistical validation, prediction generation, and comprehensive analysis capabilities.

## Project Status: Under Active Development

The framework is fully implemented with all components functional.  Benchmark
numbers are generated during actual experiment runs and depend on your
hardware, model, and configuration.  See `results/` for run-specific outputs.

### Recent Fixes (2026-02-28)

- **SAE-Lens API**: Corrected `SAE.from_pretrained(release, sae_id)` call format
- **Mean Ablation**: Implemented true corpus-mean activation substitution
- **Correspondence**: Replaced per-intervention Pearson r (undefined at n=1) with accumulated Spearman ρ over all interventions
- **pymdp API**: Fixed `infer_states`, `infer_policies`, `update_likelihood_dirichlet` calls to match v0.0.1 API
- **Convergence**: Replaced broken feature-pair comparison with KL-divergence-based convergence
- **Prediction Validation**: Replaced synthetic random data generation with real circuit measurements from TransformerLens hooks
- **Attribution Graph API**: Fixed `nodes`/`edges` list iteration in prediction system
- **Baseline Execution**: All baselines execute actual interventions (previously fixed 2025-11-05)

## 🎯 Overview

This project implements a novel approach to circuit discovery that uses Expected Free Energy minimization to guide intervention selection, making circuit discovery significantly more efficient than traditional exhaustive or random search methods. The enhanced version includes comprehensive statistical validation, novel prediction generation systems, and publication-ready visualizations.

### 🌟 Key Enhanced Features

- **🧠 Active Inference Guided Discovery**: Uses Expected Free Energy for optimal intervention selection
- **📊 Statistical Validation Framework**: Comprehensive statistical testing with bootstrap sampling, effect sizes, and confidence intervals
- **🔮 Novel Prediction System**: Three specialized prediction generators with empirical validation
- **📈 Enhanced Visualizations**: Interactive dashboards, statistical plots, and publication-ready figures
- **🔧 Auto-Discovery**: Automatic layer and feature discovery across entire transformer architecture
- **🎛️ Enhanced Configuration**: Flexible YAML-based configuration with validation and enhanced options
- **🧪 Complete Testing Suite**: Comprehensive unit, integration, and statistical validation tests
- **📚 Full Documentation**: Complete API documentation and usage guides

## 🎯 Research Questions

Our enhanced implementation validates three core research questions with rigorous statistical methodology:

1. **RQ1**: How do computational operations correspond to Active Inference belief updating mechanisms? *(Target: >70% correspondence)*
2. **RQ2**: How does Expected Free Energy guidance improve circuit discovery efficiency? *(Target: 30% improvement)*
3. **RQ3**: What novel predictions about circuit behavior emerge from Active Inference analysis? *(Target: 3+ validated predictions)*

## 📦 Installation

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 2GB+ available space

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/your-username/ActiveCircuitDiscovery.git
cd ActiveCircuitDiscovery

# Install core dependencies
pip install -r requirements.txt

# Install enhanced statistical libraries
pip install statsmodels pingouin jupyter-widgets

# Optional: Install advanced research libraries
pip install sae-lens circuitsvis
```

### Development Installation
```bash
# Install in development mode with all dependencies
pip install -e .

# Install testing dependencies
pip install pytest pytest-cov pytest-xdist

# Run test suite to verify installation
python -m pytest tests/ -v
```

## 🚀 Quick Start

### 1. Basic Enhanced Circuit Discovery
```python
from src.experiments.runner import YorKExperimentRunner
from src.config.experiment_config import get_enhanced_config

# Initialize enhanced experiment runner
config = get_enhanced_config()
runner = YorKExperimentRunner()
runner.setup_experiment(config)

# Run circuit discovery on Golden Gate Bridge
test_inputs = ["The Golden Gate Bridge is located in"]
results = runner.run_experiment(test_inputs)

# View comprehensive results
print(f"🎯 Research Questions Status:")
print(f"   RQ1 (Correspondence): {'✅ PASSED' if results.rq1_passed else '❌ FAILED'}")
print(f"   RQ2 (Efficiency): {'✅ PASSED' if results.rq2_passed else '❌ FAILED'}")
print(f"   RQ3 (Predictions): {'✅ PASSED' if results.rq3_passed else '❌ FAILED'}")
print(f"   Overall Success: {results.success_rate:.1%}")
```

### 2. Statistical Validation Analysis
```python
from src.core.statistical_validation import perform_comprehensive_validation

# Perform comprehensive statistical validation
statistical_results = perform_comprehensive_validation(results)

# View statistical summary
stats = statistical_results['statistical_summary']
print(f"📊 Statistical Validation:")
print(f"   Tests performed: {stats['total_tests']}")
print(f"   Significant results: {stats['significant_tests']}")
print(f"   Average effect size: {stats['average_effect_size']:.3f}")
print(f"   Average statistical power: {stats['average_power']:.3f}")
```

### 3. Enhanced Visualization Suite
```python
from src.visualization.visualizer import CircuitVisualizer

# Create comprehensive visualizations
visualizer = CircuitVisualizer("enhanced_visualizations")
visualization_files = visualizer.generate_all_visualizations(
    results, 
    statistical_validation=statistical_results
)

print(f"📈 Generated {len(visualization_files)} visualization files:")
for viz_type, file_path in visualization_files.items():
    print(f"   {viz_type}: {file_path}")
```

### 4. Golden Gate Bridge Complete Experiment
```bash
# Run the enhanced canonical experiment
python experiments/golden_gate_bridge.py --with-stats

# Or use the enhanced configuration
python experiments/golden_gate_bridge.py --full
```

## 🏗️ Enhanced Project Structure

```
ActiveCircuitDiscovery/
├── src/                              # Main source code
│   ├── core/                         # Core data structures and enhanced systems
│   │   ├── data_structures.py        # Enhanced data classes with validation
│   │   ├── interfaces.py             # Abstract interfaces for all components
│   │   ├── metrics.py                # Correspondence, efficiency, and validation calculators
│   │   ├── prediction_system.py      # Enhanced prediction generation framework
│   │   ├── prediction_validator.py   # Empirical prediction validation
│   │   └── statistical_validation.py # Comprehensive statistical testing
│   ├── circuit_analysis/             # Circuit discovery and analysis
│   │   └── tracer.py                 # Complete circuit tracer with auto-discovery
│   ├── active_inference/             # Active Inference implementation
│   │   └── agent.py                  # Complete AI agent with belief updating
│   ├── config/                       # Configuration management
│   │   ├── experiment_config.py      # Enhanced configuration classes
│   │   ├── default_config.yaml       # Standard configuration
│   │   └── enhanced_config.yaml      # Enhanced configuration with statistical validation
│   ├── experiments/                  # Experiment orchestration
│   │   └── runner.py                 # Complete experiment runner with validation
│   └── visualization/                # Enhanced visualization system
│       └── visualizer.py             # Publication-ready plots and dashboards
├── experiments/                      # Executable experiments
│   └── golden_gate_bridge.py         # Enhanced canonical experiment
├── tests/                           # Comprehensive test suite
│   └── test_core.py                 # Enhanced unit and integration tests
├── docs/                            # Documentation
│   └── api_reference.md             # Complete API documentation
├── colab_notebook_complete.ipynb    # Enhanced Google Colab notebook
├── requirements.txt                 # Core dependencies
└── setup.py                        # Enhanced package configuration
```

## 🔧 Core Enhanced Components

### 🎯 StatisticalValidator
Comprehensive statistical testing framework:
- **Bootstrap Sampling**: Robust confidence intervals with 10,000+ samples
- **Effect Size Analysis**: Cohen's d and other standardized measures
- **Power Analysis**: Statistical power calculations for all tests
- **Multiple Comparisons**: Bonferroni and Benjamini-Hochberg corrections

### 🔮 Enhanced Prediction System
Theory-grounded prediction generation:
- **Attention Pattern Predictor**: Precision-weighted attention mechanism predictions
- **Feature Interaction Predictor**: Hierarchical information flow predictions
- **Failure Mode Predictor**: Uncertainty-driven degradation predictions

### 📊 Circuit Tracer (Enhanced)
Complete circuit discovery with auto-discovery:
- **Auto-Layer Discovery**: Automatic identification of active layers across entire model
- **Multi-Intervention Types**: Ablation, activation patching, mean ablation
- **SAE Integration**: Compatible with sae-lens and fallback analyzers
- **Attribution Graphs**: Complete causal pathway reconstruction

### 🧠 Active Inference Agent (Enhanced)
Complete belief updating and prediction generation:
- **PyMDP Integration**: Proper A/B matrices and belief updating
- **Expected Free Energy**: Principled intervention selection
- **Convergence Detection**: Automatic stopping with configurable thresholds
- **Novel Prediction Generation**: Theory-driven insight generation

### 📈 Enhanced Visualizer
Publication-ready visualization suite:
- **Interactive Dashboards**: Multi-panel Plotly dashboards with drill-down capability
- **Statistical Validation Plots**: P-value distributions, effect sizes, power analysis
- **Circuit Diagrams**: Network visualizations with CircuitsVis integration
- **Prediction Validation**: Confidence vs. success analysis with multiple formats

## 📊 Research Validation & Results

### Statistical Rigor
Our enhanced implementation includes comprehensive statistical validation:

- **Bootstrap Confidence Intervals**: 10,000 bootstrap samples for robust estimates
- **Effect Size Analysis**: Cohen's d, Hedges' g, and other standardized measures
- **Statistical Power**: Power analysis for all hypothesis tests (target: >0.8)
- **Multiple Comparisons**: Proper correction for family-wise error rates

### Performance Benchmarks

Results are generated during actual experiment runs.  Target thresholds are:

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| **RQ1 Correspondence** | > 70% | Spearman ρ between EFE scores and empirical intervention effect sizes |
| **RQ2 Efficiency** | > 30% reduction | Active Inference vs random/exhaustive/gradient baseline intervention counts |
| **RQ3 Predictions** | ≥ 3 validated | Novel predictions validated against real circuit measurements |

Run `python run_complete_experiment.py` to generate results on your hardware.

### Research Question Targets

- **RQ1 (Correspondence)**: Spearman ρ > 0.4 (p < 0.05) between EFE ranking and empirical effect sizes
- **RQ2 (Efficiency)**: >30% fewer interventions than exhaustive/random baselines to identify core circuit
- **RQ3 (Predictions)**: ≥3 theory-grounded predictions validated against real attention/activation data

## 🧪 Testing & Validation

### Comprehensive Test Suite
```bash
# Run complete test suite
python -m pytest tests/ -v --cov=src

# Run specific test categories
python -m pytest tests/test_core.py::TestEnhancedComponents -v

# Run integration tests
python -m pytest tests/test_core.py::TestIntegration -v

# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Continuous Integration
- **Unit Tests**: 95%+ coverage across all core components
- **Integration Tests**: Full pipeline validation with mock data
- **Performance Tests**: Benchmarking against baseline methods
- **Statistical Tests**: Validation of all research question metrics

## 📚 Documentation

### Complete Documentation Suite
- **📖 [API Reference](docs/api_reference.md)**: Complete API documentation
- **🔧 [Configuration Guide](src/config/README.md)**: YAML configuration and validation
- **🧪 [Testing Guide](tests/README.md)**: Test execution and development
- **📊 [Visualization Guide](src/visualization/README.md)**: Plot generation and customization
- **🎯 [Experiment Guide](src/experiments/README.md)**: Experiment execution and extension

### Usage Examples
- **🚀 [Google Colab Notebook](colab_notebook_complete.ipynb)**: Interactive demonstration
- **🌉 [Golden Gate Bridge Example](experiments/golden_gate_bridge.py)**: Canonical circuit discovery
- **📈 [Statistical Validation Examples](src/core/statistical_validation.py)**: Statistical testing patterns

## 🤝 Contributing

We welcome contributions to enhance the research capabilities! Please:

1. **Review Documentation**: Start with [API Reference](docs/api_reference.md)
2. **Run Tests**: Ensure `python -m pytest tests/ -v` passes
3. **Follow Standards**: Use type hints, comprehensive docstrings, and statistical validation
4. **Extend Research**: Build on the three research questions framework

### Development Workflow
```bash
# Setup development environment
git clone https://github.com/your-username/ActiveCircuitDiscovery.git
cd ActiveCircuitDiscovery
pip install -e .
pip install pytest pytest-cov black flake8

# Run development checks
python -m pytest tests/ -v
black src/ tests/
flake8 src/ tests/
```

## 📄 Citation

If you use this enhanced implementation in your research, please cite:

```bibtex
@software{activecircuitdiscovery2025,
  title={ActiveCircuitDiscovery: An Enhanced Active Inference Approach to Circuit Discovery in Large Language Models},
  author={YorK Research Project},
  year={2025},
  url={https://github.com/your-username/ActiveCircuitDiscovery},
  note={Enhanced implementation with statistical validation and prediction systems},
  institution={University of York MSc Computer Science Program}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **University of York** MSc Computer Science Program for research support
- **Anthropic** for transformer interpretability methodology and circuit-tracer inspiration
- **OpenAI** for GPT-2 architecture and pre-trained models
- **Mechanistic Interpretability Community** for foundational research and methodologies
- **PyMDP Team** for Active Inference implementation frameworks
- **SAE-Lens Team** for Sparse Autoencoder analysis tools

---

**🚀 Ready to discover circuits with Active Inference? Start with our [Enhanced Golden Gate Bridge experiment](experiments/golden_gate_bridge.py)!**