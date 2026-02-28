# Active Circuit Discovery

[![Open In Colab - Circuit Discovery](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/01_circuit_discovery_gemma.ipynb)
[![Open In Colab - Active Inference Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/02_active_inference_demo.ipynb)
[![Open In Colab - Feature Steering](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/03_reproduce_biology_paper.ipynb)

**Active Inference-guided circuit discovery for mechanistic interpretability in Large Language Models.**

Combines attribution graph analysis with uncertainty-weighted exploration to efficiently identify causally important transcoder features in LLMs, using Anthropic's `circuit-tracer` library.

## Key Results

| Benchmark | AI vs Random | AI vs Greedy | Oracle Efficiency |
|-----------|-------------|-------------|-------------------|
| IOI (5 prompts) | **+36.1%** | +11.3% | **74.4%** |
| Multi-step Reasoning (3 prompts) | **+44.3%** | -1.2% | **78.4%** |

All results from real `feature_intervention` API calls on Gemma-2-2B with GemmaScope transcoders. No synthetic or fabricated data.

## Architecture

```
Prompt → Gemma-2-2B → circuit-tracer (EAP + GemmaScope Transcoders)
                                ↓
                        Attribution Graph
                    (active features, adjacency matrix)
                                ↓
                    Active Inference Selector
            (graph importance × layer prior + uncertainty bonus)
                                ↓
                     feature_intervention(layer, pos, fidx, value)
                                ↓
                  KL Divergence Measurement → Update Beliefs
```

## Research Questions

| RQ | Question | Target | Achieved |
|----|----------|--------|----------|
| RQ1 | Is AI selection more efficient than baselines? | ≥ 30% vs random | **+36-44%** |
| RQ2 | Can features be causally steered? | Prediction changes | **6/30 features** |
| RQ3 | Does AI approach oracle quality? | Oracle eff ≥ 70% | **74-78%** |

## Installation

### Local (DGX Spark / GPU machine)

```bash
git clone https://github.com/SharathSPhD/ActiveCIrcuitDiscovery.git
cd ActiveCIrcuitDiscovery
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/decoderesearch/circuit-tracer.git
```

### Docker (DGX Spark)

```bash
docker compose up active-circuit-discovery
```

### Google Colab

Click any Colab badge above -- dependencies install automatically.

## Quick Start

```python
import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph

model = ReplacementModel.from_pretrained(
    model_name="google/gemma-2-2b",
    transcoder_set="gemma",
    backend="transformerlens",
    device=torch.device("cuda"),
    dtype=torch.float32,
)

prompt = "When John and Mary went to the store, John gave the bag to"
raw_graph = attribute(prompt=prompt, model=model, max_n_logits=5,
                      desired_logit_prob=0.9, batch_size=256)

# Ablate a feature and measure causal effect
clean_logits, _ = model.feature_intervention(prompt, [])
clean_probs = torch.softmax(clean_logits[0, -1, :], -1)

iv_logits, _ = model.feature_intervention(
    prompt, [(25, 14, 4717, 0)]  # ablate L25 feature 4717
)
iv_probs = torch.softmax(iv_logits[0, -1, :], -1)
kl = torch.nn.functional.kl_div(
    torch.log(iv_probs + 1e-10), clean_probs, reduction='sum'
).item()
print(f"KL divergence from ablation: {kl:.6f}")
```

## Running Experiments

```bash
# Full experiment suite (IOI + steering + multi-step)
python -m src.experiments.run_real_experiments

# Results saved to results/ as JSON
ls results/*.json
```

## Project Structure

```
ActiveCIrcuitDiscovery/
├── src/
│   ├── experiments/
│   │   └── run_real_experiments.py    # Main experiment runner
│   ├── circuit_analysis/
│   │   └── circuit_tracer_backend.py  # circuit-tracer wrapper
│   ├── active_inference/
│   │   ├── pomdp_agent.py            # Legacy POMDP agent
│   │   └── agent.py                  # Legacy agent
│   ├── core/
│   │   ├── data_structures.py        # Core data types
│   │   └── metrics.py                # Statistical metrics
│   └── visualization/
│       ├── visualizer.py             # Result plotting
│       └── research_dashboard.py     # Interactive dashboard
├── notebooks/
│   ├── 01_circuit_discovery_gemma.ipynb   # Circuit discovery demo
│   ├── 02_active_inference_demo.ipynb     # AI selector comparison
│   └── 03_reproduce_biology_paper.ipynb   # Feature steering
├── paper/                                 # LaTeX paper source
├── results/                               # Experiment JSON outputs
├── Dockerfile.dgx-spark
├── docker-compose.yml
└── requirements.txt
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (T4) | 128 GB (DGX Spark) |
| RAM | 16 GB | 64 GB |
| Storage | 20 GB | 50 GB |

## Citation

```bibtex
@article{activecircuitdiscovery2026,
  title={Active Circuit Discovery: Uncertainty-Weighted Feature Selection
         for Mechanistic Interpretability in Large Language Models},
  author={Sharath S.},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
