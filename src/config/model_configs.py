"""
Supported Model Configurations
================================
Defines the models supported by ActiveCircuitDiscovery along with
their architecture parameters and corresponding SAE releases.

Usage:
    from src.config.model_configs import SUPPORTED_MODELS, get_model_config
    cfg = get_model_config("EleutherAI/pythia-1b")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Architecture and SAE metadata for a supported model."""
    n_layers: int
    d_model: int
    n_heads: int
    sae_release: Optional[str]         # SAE-Lens release name (None = no SAE)
    sae_hook_template: str             # e.g. "blocks.{layer}.hook_resid_post"
    vram_gb: float                     # Approximate GPU VRAM required
    notes: str = ""


SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    # ── GPT-2 family ──────────────────────────────────────────────────────
    "gpt2": ModelConfig(
        n_layers=12, d_model=768, n_heads=12,
        sae_release="gpt2-small-res-jb",
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=2.0,
        notes="GPT-2 Small (124M). Primary development target.",
    ),
    "gpt2-small": ModelConfig(
        n_layers=12, d_model=768, n_heads=12,
        sae_release="gpt2-small-res-jb",
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=2.0,
    ),
    "gpt2-medium": ModelConfig(
        n_layers=24, d_model=1024, n_heads=16,
        sae_release=None,  # No public SAE release yet for gpt2-medium
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=6.0,
    ),
    "gpt2-large": ModelConfig(
        n_layers=36, d_model=1280, n_heads=20,
        sae_release=None,
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=12.0,
    ),
    # ── Pythia family (EleutherAI) — mechanistic interpretability standard ─
    "EleutherAI/pythia-160m": ModelConfig(
        n_layers=12, d_model=768, n_heads=12,
        sae_release="pythia-160m-deduped-v0",
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=1.0,
    ),
    "EleutherAI/pythia-410m": ModelConfig(
        n_layers=24, d_model=1024, n_heads=16,
        sae_release="pythia-410m-deduped-v0",
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=2.0,
    ),
    "EleutherAI/pythia-1b": ModelConfig(
        n_layers=16, d_model=2048, n_heads=8,
        sae_release="pythia-1b-deduped-v0",
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=4.0,
    ),
    "EleutherAI/pythia-2.8b": ModelConfig(
        n_layers=32, d_model=2560, n_heads=32,
        sae_release=None,
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=12.0,
    ),
    "EleutherAI/pythia-6.9b": ModelConfig(
        n_layers=32, d_model=4096, n_heads=32,
        sae_release=None,
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=28.0,
        notes="Requires 40 GB GPU.  DGX Spark recommended.",
    ),
    # ── Gemma (Google) ────────────────────────────────────────────────────
    "google/gemma-2b": ModelConfig(
        n_layers=18, d_model=2048, n_heads=8,
        sae_release=None,
        sae_hook_template="blocks.{layer}.hook_resid_post",
        vram_gb=8.0,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Return ModelConfig for the given model name.

    Raises:
        KeyError: If the model is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise KeyError(
            f"Model '{model_name}' not in SUPPORTED_MODELS. "
            f"Available: {available}"
        )
    return SUPPORTED_MODELS[model_name]


def list_supported_models() -> List[str]:
    """Return list of all supported model names."""
    return list(SUPPORTED_MODELS.keys())


def get_sae_release(model_name: str) -> Optional[str]:
    """Return the SAE-Lens release name for the model, or None."""
    try:
        return get_model_config(model_name).sae_release
    except KeyError:
        return None
