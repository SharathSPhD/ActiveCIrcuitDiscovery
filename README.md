# ActiveCircuitDiscovery - Enhanced Active Inference

**Circuit discovery in Large Language Models using Active Inference + circuit-tracer**

## Overview

This project implements an Enhanced Active Inference approach to discover neural circuits in transformer architectures. It combines:

- **Real Circuit Discovery**: Using `circuit-tracer` library with Gemma-2-2B transcoders
- **Active Inference**: Principled intervention selection via `pymdp` Expected Free Energy
- **GPU Optimization**: L40S GPU deployment on DigitalOcean

## Research Questions

1. **RQ1**: AI-Circuit Correspondence ≥70%
2. **RQ2**: Intervention Efficiency ≥30% vs baseline
3. **RQ3**: Novel Predictions ≥3 from learned model

## Quick Start

### Local Development
```bash
git clone <repository>
cd ActiveCircuitDiscovery
pip install -r requirements.txt
python run_integrated_experiment.py  # CPU testing
```

### GPU Deployment (L40S)
```bash
# Deploy L40S GPU droplet
python create_droplet_fixed.py

# SSH to droplet
ssh ubuntu@<DROPLET_IP>
cd ActiveCircuitDiscovery

# Run full experiment  
python run_integrated_experiment.py
```

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐
│   Active Inference  │    │   Circuit Discovery  │
│                     │    │                      │
│ ProperActiveInference│◄──►│ RealCircuitTracer    │
│ Agent (pymdp)       │    │ (circuit-tracer)     │
│                     │    │                      │
│ - EFE calculation   │    │ - Gemma-2-2B model   │
│ - Belief updating   │    │ - Transcoder features │
│ - Policy inference  │    │ - Real interventions  │
└─────────────────────┘    └──────────────────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
           ┌─────────────────────┐
           │ Experiment Results  │
           │                     │
           │ - Circuit graphs    │
           │ - RQ validation     │
           │ - Novel predictions │
           └─────────────────────┘
```

## Key Features

### Circuit Discovery
- **Natural Feature Emergence**: Golden Gate Bridge → San Francisco features discovered automatically
- **Transcoder Analysis**: Real circuit components via sparse feature analysis
- **Intervention Framework**: Ablation, patching, scaling interventions

### Active Inference
- **Real pymdp Integration**: No fallbacks or approximations
- **Generative Models**: Circuit-specific A, B, C, D matrices
- **EFE-guided Selection**: Principled intervention choice
- **Belief Learning**: Bayesian updating from intervention results

### Research Validation
- **Statistical Testing**: Bootstrap confidence intervals, hypothesis testing
- **Efficiency Analysis**: Comparison with baseline intervention methods
- **Prediction Generation**: Novel theoretical predictions from learned models

## Project Structure

```
src/
├── active_inference/          # Real Active Inference implementation
│   ├── proper_agent.py       # Main pymdp.Agent integration
│   ├── generative_model.py   # Circuit-specific generative models
│   └── inference_cycle.py    # Complete perception-action cycle
├── circuit_analysis/          # Circuit discovery tools
│   └── real_tracer.py        # circuit-tracer + Gemma-2-2B integration
├── experiments/               # Experiment orchestration
│   └── circuit_discovery_integration.py  # Main integration class
├── core/                      # Core data structures and metrics
│   ├── data_structures.py    # Type-safe data classes
│   ├── metrics.py           # RQ calculators
│   └── statistical_validation.py  # Statistical testing
└── visualization/             # Result visualization
    └── visualizer.py         # Circuit and metrics visualization

tests/                         # Comprehensive test suite
└── test_proper_agent.py      # Active Inference validation

run_integrated_experiment.py  # Main experiment runner
requirements.txt              # All dependencies (circuit-tracer, pymdp, etc.)
```

## Dependencies

- **PyTorch**: 2.7.1 with CUDA 12.1 support
- **circuit-tracer**: ≥0.1.0 for real circuit discovery
- **pymdp**: 0.0.1 for Active Inference
- **transformers**: 4.52.4 for Gemma-2-2B
- **transformer-lens**: 2.16.0 for interpretability

## Results

Experiments generate:
- **Circuit Graphs**: Interactive visualization of discovered circuits
- **RQ Validation**: Statistical validation of all research questions
- **Novel Predictions**: Theoretical predictions with empirical validation
- **Comprehensive Metrics**: Correspondence, efficiency, statistical significance

## License

MIT License