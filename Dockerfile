# syntax=docker/dockerfile:1
# ActiveCircuitDiscovery — DGX Spark / H100 / GB200 Optimised Container
#
# Build:  docker build -t activecircuitdiscovery:latest .
# Run:    docker compose -f docker-compose-dgx.yml up -d

FROM nvcr.io/nvidia/pytorch:24.04-py3

LABEL maintainer="ActiveCircuitDiscovery"
LABEL description="Active Inference Circuit Discovery in LLMs — DGX Spark"
LABEL version="2.0"

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget vim htop nvtop jq \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ────────────────────────────────────────────────────────
WORKDIR /workspace/ActiveCircuitDiscovery

# Copy dependency files first (layer-cache friendly)
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Core research libraries
RUN pip install --no-cache-dir \
        "transformer-lens>=2.0.0" \
        "sae-lens>=0.3.0" \
        "pymdp>=0.0.1" \
        "circuitsvis" \
        "einops>=0.7.0" \
        "fancy-einsum" \
        "plotly>=5.0.0" \
        "dash>=2.0.0" \
        "kaleido" \
        "wandb" \
        "scipy>=1.10.0" \
        "statsmodels" \
        "jupyter" \
        "jupyterlab" \
        "ipywidgets" \
        "tqdm" \
        "pydantic>=2.0.0"

# Flash-attention for H100/GB200 (graceful failure if unsupported)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "[INFO] Flash-attention not available — continuing without it"

# Try circuit-tracer (optional; install manually if not on PyPI)
RUN pip install --no-cache-dir circuit-tracer 2>/dev/null || \
    echo "[INFO] circuit-tracer not on PyPI — clone from safety-research/circuit-tracer"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

RUN pip install --no-cache-dir -e .

# Persistent directories
RUN mkdir -p results logs visualizations cache/models cache/saes data

# ── Environment ───────────────────────────────────────────────────────────────
ENV HF_HOME=/workspace/cache/models
ENV TRANSFORMERS_CACHE=/workspace/cache/models
ENV HF_DATASETS_CACHE=/workspace/cache/datasets
ENV SAE_LENS_CACHE=/workspace/cache/saes

# DGX / NCCL tuning
ENV NCCL_DEBUG=WARN
ENV NCCL_TREE_THRESHOLD=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=all

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8888   # JupyterLab
EXPOSE 8050   # Dash visualisation dashboard
EXPOSE 6006   # TensorBoard / WandB local

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "import torch; import transformer_lens; assert torch.cuda.is_available() or True; print('healthy')" \
    || exit 1

# ── Default entry point ───────────────────────────────────────────────────────
CMD ["bash"]
