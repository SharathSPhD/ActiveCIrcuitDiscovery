# ActiveCircuitDiscovery

**An Active Inference Approach to Circuit Discovery in Large Language Models**

A research implementation demonstrating how Active Inference principles can guide efficient circuit discovery in transformer architectures, specifically targeting mechanistic interpretability of Large Language Models.

## Overview

This project implements a novel approach to circuit discovery that uses Expected Free Energy minimization to guide intervention selection, making circuit discovery more efficient than traditional exhaustive or random search methods.

### Key Features

- **Active Inference Guided Discovery**: Uses Expected Free Energy to select the most informative interventions
- **Real SAE Integration**: Works with Sparse Autoencoders for interpretable feature extraction
- **Comprehensive Visualizations**: Neuronpedia-style interactive visualizations of discovered circuits
- **Golden Gate Bridge Circuit**: Demonstrates circuit discovery on a well-studied example

## Research Questions

1. **RQ1**: How do computational operations of induction heads correspond to Active Inference belief updating mechanisms?
2. **RQ2**: How does Expected Free Energy guidance improve circuit discovery efficiency compared to baseline methods?
3. **RQ3**: What novel predictions about circuit behavior emerge from Active Inference analysis?

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone or download the project
cd ActiveCircuitDiscovery

# Install dependencies
pip install -r requirements.txt

# Optional: Install SAE-lens for real SAE integration
pip install sae-lens

# Optional: Install circuit-tracer for advanced features
pip install circuit-tracer
```

## Quick Start

### Basic Circuit Discovery
```python
from src.circuit_tracer import RealCircuitTracer
from src.active_inference import ActiveInferenceGuide

# Initialize components
tracer = RealCircuitTracer(device="cuda")
ai_guide = ActiveInferenceGuide(tracer)

# Discover circuit for Golden Gate Bridge
text = "The Golden Gate Bridge is located in"
ai_guide.initialize_beliefs(text)
attribution_graph = tracer.build_attribution_graph(text)

print(f"Discovered {len(attribution_graph.nodes)} circuit nodes")
print(f"Graph confidence: {attribution_graph.confidence:.3f}")
```

### Run Complete Experiment
```python
from src.experiment import CompleteExperimentRunner

# Run full comparative experiment
runner = CompleteExperimentRunner()
runner.setup_experiment()
results = runner.run_golden_gate_bridge_experiment()

# Results include Active Inference vs baseline comparisons
print(f"Research Questions Validation:")
for rq, data in results['overall_results']['research_question_validation'].items():
    if rq.startswith('rq'):
        print(f"{rq.upper()}: {'PASSED' if data['passed'] else 'FAILED'}")
```

### Golden Gate Bridge Experiment
```python
# Run the signature experiment
python experiments/golden_gate_bridge.py

# This will generate:
# - Circuit discovery results
# - Comparative analysis (Active Inference vs baselines)
# - Interactive visualizations
# - Research question validation
```

## Project Structure

```
ActiveCircuitDiscovery/
├── src/
│   ├── circuit_tracer.py      # Core circuit discovery using SAEs
│   ├── active_inference.py    # Active Inference guidance system
│   ├── visualizer.py          # Interactive visualization system
│   └── experiment.py          # Complete experiment runner
├── experiments/
│   └── golden_gate_bridge.py  # Golden Gate Bridge circuit discovery
├── docs/
│   └── api_reference.md       # API documentation
└── tests/
    └── test_core.py           # Basic functionality tests
```

## Core Components

### CircuitTracer
Implements real circuit discovery using:
- Sparse Autoencoders (SAEs) for interpretable features
- Activation patching and ablation for causal analysis
- Attribution graph construction

### ActiveInference
Provides principled intervention selection using:
- Expected Free Energy minimization
- Bayesian belief updating
- Uncertainty-driven exploration

### Visualizer
Creates publication-ready visualizations:
- Feature activation heatmaps
- Circuit network diagrams
- Intervention effect analysis
- Active Inference process visualization

## Research Validation

The project validates three core research questions:

- **RQ1 Target**: >70% correspondence between Active Inference and circuit behavior
- **RQ2 Target**: 30% efficiency improvement over baseline methods
- **RQ3 Target**: 3+ validated novel predictions about circuit behavior

## Contributing

This is a research implementation. For contributions or questions:

1. Review the API documentation in `docs/api_reference.md`
2. Run tests with `python -m pytest tests/`
3. Follow the research methodology outlined in the original proposal

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{yorkrp2025,
  title={An Active Inference Approach to Circuit Discovery in Large Language Models},
  author={YorK_RP Research Project},
  year={2025},
  howpublished={University of York MSc Computer Science Research Project}
}
```

## License

This project is for research and educational purposes. See LICENSE for details.

## Acknowledgments

- University of York MSc Computer Science Program
- Anthropic for circuit-tracer methodology inspiration
- OpenAI for transformer architectures and GPT-2
- The mechanistic interpretability research community