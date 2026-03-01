# Active Circuit Discovery

[![Open In Colab - Circuit Discovery](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/01_circuit_discovery_gemma.ipynb)
[![Open In Colab - Active Inference Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/02_active_inference_demo.ipynb)
[![Open In Colab - Feature Steering](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/notebooks/03_reproduce_biology_paper.ipynb)

**Active Inference-guided circuit discovery for mechanistic interpretability in Large Language Models.**

Combines attribution graph analysis with a POMDP agent (powered by [pymdp](https://github.com/infer-actively/pymdp)) to efficiently identify causally important transcoder features in LLMs, using Anthropic's `circuit-tracer` library. Evaluated on **Gemma-2-2B** and **Llama-3.2-1B**.

## Key Results

| Benchmark | Model | POMDP vs Random | Oracle Efficiency |
|-----------|-------|-----------------|-------------------|
| IOI (5 prompts) | Gemma-2-2B | +16.3% | 58.3% |
| IOI (5 prompts) | Llama-3.2-1B | -25.4% | 37.5% |
| Multi-step (3 prompts) | Gemma-2-2B | +30.4% | 73.3% |
| Multi-step (3 prompts) | Llama-3.2-1B | -88.8% | 6.5% |
| Feature Steering | Gemma-2-2B | 8/50 prediction changes at 10x | --- |
| Feature Steering | Llama-3.2-1B | 9/50 prediction changes at 10x | --- |

All results from real `feature_intervention` API calls with GemmaScope / Llama transcoders. No synthetic or fabricated data.

## Architecture

```
Prompt → Gemma-2-2B / Llama-3.2-1B
           ↓
   circuit-tracer (EAP + Transcoders)
           ↓
   Attribution Graph (active features, adjacency matrix)
           ↓
   Active Inference POMDP Agent (pymdp)
   ├─ Variational state inference
   ├─ Expected Free Energy scoring
   └─ Dirichlet observation-model learning
           ↓
   feature_intervention(layer, pos, fidx, value)
           ↓
   KL Divergence Measurement → Update Beliefs
```

## Hypotheses and Outcomes

| ID | Hypothesis | Criterion | Outcome |
|----|-----------|-----------|---------|
| H1 | POMDP agent more efficient than random | Paired t-test, α=0.05 | Positive trend on Gemma; not significant (p=0.27) |
| H2 | Features causally control predictions | Binomial test > 1% baseline | **Accepted** (17/100, p < 10⁻¹⁵) |
| H3 | Oracle efficiency ≥ 50% | Point estimate | **Accepted** for Gemma (58.3%); not for Llama (37.5%) |
| H4 | Findings transfer across architectures | Qualitative replication | **Accepted** (all experiments replicated on Llama) |

## Installation

### Local (DGX Spark / GPU machine)

```bash
git clone https://github.com/SharathSPhD/ActiveCIrcuitDiscovery.git
cd ActiveCIrcuitDiscovery
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/safety-research/circuit-tracer.git
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
    model_name="google/gemma-2-2b",   # or "unsloth/Llama-3.2-1B"
    transcoder_set="gemma",            # or "llama"
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
# Full experiment suite on both models
python -m src.experiments.run_real_experiments --model both --experiment all

# Single model / single benchmark
python -m src.experiments.run_real_experiments --model gemma --experiment ioi
python -m src.experiments.run_real_experiments --model llama --experiment steering

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
│   │   └── pomdp_agent.py            # POMDP agent (pymdp)
│   ├── core/
│   │   ├── data_structures.py        # Core data types
│   │   └── metrics.py                # Statistical metrics
│   └── visualization/
│       ├── visualizer.py             # Result plotting
│       └── research_dashboard.py     # Interactive dashboard
├── scripts/
│   └── generate_figure_data.py       # LaTeX figure generation
├── notebooks/
│   ├── 01_circuit_discovery_gemma.ipynb   # Circuit discovery demo
│   ├── 02_active_inference_demo.ipynb     # POMDP agent comparison
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
  author={Sharath Sathish},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
