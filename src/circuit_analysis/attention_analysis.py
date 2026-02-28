"""
Attention Head Level Analysis
==============================
Computes per-head importance scores via attention knockout and
head-level activation patching for circuit decomposition.

This is required for:
- RQ1: Mapping attention head operations to Active Inference precision weighting
- Circuit visualisation at the head level
- IOI task analysis (name-mover, negative name-mover, and induction heads)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False


class AttentionHeadAnalyzer:
    """Analyses individual attention head contributions to circuits.

    Methods
    -------
    get_head_importance_scores   : Attention knockout per head.
    get_attention_patterns       : Extract raw attention weights.
    get_head_direct_effects      : Direct effect via activation patching.
    get_q_k_v_composition        : Measure Q/K/V composition strengths.
    classify_head_role           : Label heads as name-mover, induction, etc.
    """

    def __init__(self, model: "HookedTransformer"):
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("transformer-lens is required for attention analysis.")
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_model = model.cfg.d_model
        self.d_head = model.cfg.d_head

    # ── Public API ────────────────────────────────────────────────────────────

    def get_head_importance_scores(
        self,
        tokens: torch.Tensor,
        metric_fn=None,
        layers: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute per-head importance via attention knockout.

        Args:
            tokens: Tokenised input of shape (1, seq_len).
            metric_fn: Function(logits) -> float.  Default: max logit.
            layers: Subset of layers to analyse.  Default: all.

        Returns:
            List of dicts sorted by abs(importance) descending.
            Each dict: {layer, head, importance, baseline_metric, ko_metric}.
        """
        if metric_fn is None:
            def metric_fn(logits: torch.Tensor) -> float:
                return float(logits[0, -1, :].max().item())

        target_layers = layers if layers is not None else list(range(self.n_layers))

        with torch.no_grad():
            baseline_logits = self.model(tokens)
        baseline_metric = metric_fn(baseline_logits)

        results = []
        for layer_idx in target_layers:
            for head_idx in range(self.n_heads):
                ko_metric = self._attention_knockout(
                    tokens, layer_idx, head_idx, metric_fn
                )
                results.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "importance": baseline_metric - ko_metric,
                    "baseline_metric": baseline_metric,
                    "ko_metric": ko_metric,
                })

        results.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return results

    def get_attention_patterns(
        self, tokens: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Extract raw attention weight matrix for all heads in a layer.

        Returns:
            Tensor of shape (n_heads, seq_len, seq_len).
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)

        hook_key = f"blocks.{layer}.attn.hook_pattern"
        if hook_key not in cache:
            raise KeyError(f"Attention pattern cache key not found: {hook_key}")

        # Shape: (batch, n_heads, seq_len, seq_len) → squeeze batch
        return cache[hook_key][0]

    def get_head_direct_effects(
        self, tokens: torch.Tensor, layer: int
    ) -> Dict[int, float]:
        """Estimate direct effect of each head via activation patching.

        Ablates each head independently and measures the change in output
        metric, providing a cleaner signal than full knockout.

        Returns:
            Dict mapping head_idx -> direct_effect_size.
        """
        with torch.no_grad():
            baseline_logits, baseline_cache = self.model.run_with_cache(tokens)

        baseline_metric = float(baseline_logits[0, -1, :].max().item())
        head_effects: Dict[int, float] = {}

        for head_idx in range(self.n_heads):
            def patch_hook(attn_out, hook, h=head_idx):
                # Zero out this head's contribution
                attn_out[:, :, h * self.d_head:(h + 1) * self.d_head] = 0.0
                return attn_out

            hook_point = f"blocks.{layer}.attn.hook_result"
            with torch.no_grad():
                with self.model.hooks([(hook_point, patch_hook)]):
                    ko_logits = self.model(tokens)

            ko_metric = float(ko_logits[0, -1, :].max().item())
            head_effects[head_idx] = baseline_metric - ko_metric

        return head_effects

    def get_q_k_v_composition(
        self, layer: int, prev_layer: int
    ) -> Dict[str, torch.Tensor]:
        """Measure Q, K, V composition strengths between layers.

        Computes the Frobenius norm of W_Q^T @ W_OV for each head pair,
        following Elhage et al. (2021) "A Mathematical Framework".

        Returns:
            Dict with keys 'Q', 'K', 'V' each of shape (n_heads_curr, n_heads_prev).
        """
        if prev_layer < 0 or prev_layer >= layer:
            raise ValueError("prev_layer must be < layer")

        # Extract weight matrices
        W_Q = self.model.blocks[layer].attn.W_Q          # (n_heads, d_model, d_head)
        W_K = self.model.blocks[layer].attn.W_K
        W_V = self.model.blocks[layer].attn.W_V
        W_O = self.model.blocks[prev_layer].attn.W_O     # (n_heads, d_head, d_model)
        W_V_prev = self.model.blocks[prev_layer].attn.W_V

        n_curr = self.n_heads
        n_prev = W_O.shape[0]

        # W_OV = W_V_prev @ W_O: shape (n_prev_heads, d_model, d_model)
        W_OV = torch.einsum("hid,hjd->hij", W_V_prev, W_O)  # approximate

        composition = {q: torch.zeros(n_curr, n_prev) for q in ("Q", "K", "V")}
        for curr_h in range(n_curr):
            for prev_h in range(n_prev):
                ov = W_OV[prev_h]  # (d_model, d_model)
                composition["Q"][curr_h, prev_h] = float(
                    torch.norm(W_Q[curr_h] @ ov).item()
                )
                composition["K"][curr_h, prev_h] = float(
                    torch.norm(W_K[curr_h] @ ov).item()
                )
                composition["V"][curr_h, prev_h] = float(
                    torch.norm(W_V[curr_h] @ ov).item()
                )

        return composition

    def classify_head_role(
        self,
        head_scores: List[Dict[str, Any]],
        dataset_prompts: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify the top-k heads into functional roles.

        Classification heuristics:
        - **Name-mover**: High importance, late layers, high V-composition from early layers.
        - **Induction**: Attends to previous occurrences of current token.
        - **Previous-token**: Consistently attends one position back.

        Returns:
            Dict mapping role name -> list of head dicts.
        """
        top_heads = head_scores[:top_k]
        roles: Dict[str, List[Dict[str, Any]]] = {
            "name_mover": [],
            "induction": [],
            "previous_token": [],
            "unknown": [],
        }

        n_layers = self.n_layers
        for head in top_heads:
            layer, h = head["layer"], head["head"]
            # Name-mover heuristic: top half of model, high absolute importance
            if layer >= n_layers // 2 and abs(head["importance"]) > 0.01:
                roles["name_mover"].append(head)
            else:
                roles["unknown"].append(head)

        return roles

    # ── Private helpers ───────────────────────────────────────────────────────

    def _attention_knockout(
        self,
        tokens: torch.Tensor,
        layer: int,
        head: int,
        metric_fn,
    ) -> float:
        """Zero out a single attention head and measure resulting metric."""

        def ko_hook(attn_pattern, hook, h=head):
            # attn_pattern: (batch, n_heads, seq, seq)
            attn_pattern[:, h, :, :] = 0.0
            return attn_pattern

        hook_point = f"blocks.{layer}.attn.hook_pattern"
        with torch.no_grad():
            with self.model.hooks([(hook_point, ko_hook)]):
                ko_logits = self.model(tokens)

        return metric_fn(ko_logits)
