# ActiveCircuitDiscovery - Complete Project Guide

**An Active Inference Approach to Circuit Discovery in Large Language Models**  
*YorK_RP Research Project - University of York MSc Computer Science*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Research Questions & Objectives](#research-questions--objectives) 
3. [Active Inference Theory](#active-inference-theory)
4. [Circuit Discovery Background](#circuit-discovery-background)
5. [Software Architecture](#software-architecture)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Research Methodology](#research-methodology)
9. [Expected Results](#expected-results)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## 🎯 Project Overview

### Core Innovation
This project introduces **Expected Free Energy guided circuit discovery** - a principled approach to understanding how Large Language Models (LLMs) implement specific algorithms in their weights. Unlike traditional exhaustive or random search methods, our approach uses Active Inference principles to select the most informative interventions, dramatically improving efficiency.

### Research Significance
- **AI Safety**: Better understanding of LLM internal mechanisms enables detection of problematic behaviors
- **Mechanistic Interpretability**: Provides theoretically grounded methods for circuit discovery
- **Neuroscience Bridge**: Connects transformer architectures with biological intelligence principles
- **Efficiency Gains**: Reduces computational cost of interpretability research

### Key Innovation Points
1. **Auto-Discovery**: Automatically identifies relevant layers instead of forcing target layers
2. **Theoretical Foundation**: Uses Expected Free Energy minimization for intervention selection
3. **Efficiency Proof**: Active Inference should need fewer interventions than baseline methods
4. **Novel Predictions**: Generates testable hypotheses about circuit behavior

---

## 🔬 Research Questions & Objectives

### Primary Research Questions

#### **RQ1: Correspondence Analysis (Target: >70%)**
*"How do computational operations of induction heads in GPT-2 Small correspond to active inference mechanisms of belief updating and precision weighting?"*

**Validation Criteria:**
- Measure functional similarity between Active Inference belief updates and transformer attention patterns
- Calculate correspondence between precision weighting and attention head behavior  
- Target: >70% correspondence in key operations

#### **RQ2: Efficiency Improvement (Target: 30%)**
*"How does an expected free energy-guided circuit discovery approach perform compared to existing interpretability methods in identifying and characterizing transformer circuits?"*

**Validation Criteria:**
- Compare intervention counts: Active Inference vs baseline methods
- Measure convergence speed to circuit identification
- Target: 30% reduction in required interventions

#### **RQ3: Novel Predictions (Target: 3+)**
*"What novel behavioral predictions about GPT-2's in-context learning emerge from analyzing circuits through active inference principles, and how do these predictions perform under empirical testing?"*

**Validation Criteria:**
- Generate predictions about circuit behavior using Active Inference theory
- Empirically test predictions with controlled experiments
- Target: 3+ validated novel predictions

### Research Objectives

1. **Develop Theoretical Mappings**: Create formal correspondences between Active Inference processes and transformer computations

2. **Implement Discovery Algorithm**: Build Expected Free Energy guided intervention selection system

3. **Validate Hierarchical Organization**: Test whether transformers exhibit Active Inference predicted hierarchical structures

4. **Demonstrate Efficiency**: Prove Active Inference guidance reduces required interventions vs baselines

5. **Generate Novel Insights**: Discover new understanding of transformer circuit organization

---

## 🧠 Active Inference Theory

### Theoretical Foundation

**Active Inference** is a comprehensive mathematical framework for understanding perception, learning, and action in intelligent systems, developed by Karl Friston and colleagues.

#### Core Principles

**1. Free Energy Minimization**
```
F = E_q[log q(x) - log p(o,x)] ≥ -log p(o)
```
Where:
- F = Variational Free Energy 
- q(x) = Approximate posterior beliefs
- p(o,x) = Generative model
- p(o) = Model evidence

**2. Prediction Error Minimization**
Intelligent agents minimize surprise by:
- **Perception**: Updating beliefs to reduce prediction error
- **Action**: Changing environment to match predictions

**3. Expected Free Energy (EFE)**
```
G = E_q[log q(x) - log p(o,x)] + KL[q(x)||p(x)]
```
Guides action/intervention selection by balancing:
- **Epistemic Value**: Information gain (exploration)
- **Pragmatic Value**: Goal achievement (exploitation)

#### Active Inference in Transformers

**Correspondence Hypothesis:**
- **Attention Mechanisms** ↔ **Precision Weighting**
- **Residual Stream Updates** ↔ **Belief Updating** 
- **Layer Hierarchies** ↔ **Hierarchical Message Passing**
- **In-Context Learning** ↔ **Online Belief Revision**

**Mathematical Mapping:**
```python
# Attention as Precision Weighting
attention_weights = softmax(Q @ K.T / sqrt(d_k))
precision_weights = softmax(log_precision_parameters)

# Residual Updates as Belief Updates
residual_update = x + attention_output + mlp_output
belief_update = prior_beliefs + prediction_error_correction

# Expected Free Energy for Intervention Selection
efe_score = epistemic_value + pragmatic_value
intervention_priority = argmax(efe_scores)
```

---

## ⚙️ Software Architecture

### Modular Design Overview

```
ActiveCircuitDiscovery/
├── src/
│   ├── config/                 # Configuration management
│   │   ├── experiment_config.py
│   │   ├── default_config.yaml
│   │   └── __init__.py
│   ├── core/                   # Core data structures & interfaces
│   │   ├── data_structures.py
│   │   ├── interfaces.py
│   │   └── __init__.py
│   ├── circuit_analysis/       # Circuit discovery & analysis
│   │   ├── tracer.py
│   │   └── __init__.py
│   ├── active_inference/       # Active Inference implementation
│   │   ├── agent.py
│   │   └── __init__.py
│   └── experiments/            # Experiment runners
│       ├── runner.py
│       └── __init__.py
├── experiments/                # Specific experiments
│   └── golden_gate_bridge.py
├── docs/                       # Documentation
├── tests/                      # Unit tests
└── requirements.txt            # Dependencies
```