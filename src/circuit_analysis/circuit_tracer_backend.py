"""
Circuit Tracer Backend
=======================
Production integration with Anthropic's circuit-tracer library for
attribution graph generation using Edge Attribution Patching (EAP)
with transcoders.

Supports:
  - Gemma-2-2B via GemmaScope transcoders (mwhanna/gemma-scope-transcoders)
  - Llama-3.2-1B via ReLU transcoders (mntss/transcoder-Llama-3.2-1B)

Uses the TransformerLens backend (no nnsight dependency required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from circuit_tracer import ReplacementModel, Graph, attribute
from circuit_tracer.graph import prune_graph, PruneResult

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_structures import AttributionGraph, GraphNode, GraphEdge
except ImportError:
    from src.core.data_structures import AttributionGraph, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


TRANSCODER_REGISTRY: Dict[str, str] = {
    "google/gemma-2-2b": "gemma",
    "gemma-2-2b": "gemma",
    "meta-llama/Llama-3.2-1B": "llama",
    "llama-3.2-1b": "llama",
}

MODEL_NAME_MAP: Dict[str, str] = {
    "google/gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b": "google/gemma-2-2b",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
}


@dataclass
class NodeInfo:
    """Parsed information about a node in the circuit-tracer graph."""
    node_id: str
    node_type: str  # "feature", "error", "embed", "logit"
    layer: int
    position: int
    feature_idx: int
    importance: float
    activation: float
    raw_index: int  # original index in adjacency matrix


class CircuitTracerBackend:
    """Production wrapper around Anthropic's circuit-tracer library.

    Loads a ReplacementModel with transcoders and provides methods to:
      1. Generate attribution graphs via EAP
      2. Prune graphs to desired size
      3. Convert to ActiveCircuitDiscovery's AttributionGraph format
      4. Extract feature-level information for the Active Inference agent
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        hf_token: Optional[str] = None,
    ):
        self.model_name = MODEL_NAME_MAP.get(model_name, model_name)
        self.transcoder_ref = TRANSCODER_REGISTRY.get(model_name)
        if self.transcoder_ref is None:
            raise ValueError(
                f"No transcoder registered for '{model_name}'. "
                f"Available: {list(TRANSCODER_REGISTRY.keys())}"
            )

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.hf_token = hf_token

        self._model: Optional[ReplacementModel] = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the ReplacementModel with transcoders from HuggingFace."""
        if self._loaded:
            return

        logger.info(
            f"Loading ReplacementModel for {self.model_name} "
            f"with transcoder set '{self.transcoder_ref}' on {self.device}"
        )

        self._model = ReplacementModel.from_pretrained(
            model_name=self.model_name,
            transcoder_set=self.transcoder_ref,
            backend="transformerlens",
            device=self.device,
            dtype=self.dtype,
        )
        self._loaded = True
        logger.info("ReplacementModel loaded successfully.")

    @property
    def model(self) -> ReplacementModel:
        if not self._loaded or self._model is None:
            self.load_model()
        return self._model

    def get_raw_graph(
        self,
        prompt: str,
        *,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        batch_size: int = 512,
        max_feature_nodes: Optional[int] = None,
        verbose: bool = False,
    ) -> Graph:
        """Generate an unpruned attribution graph via EAP."""
        return attribute(
            prompt=prompt,
            model=self.model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            verbose=verbose,
        )

    def get_attribution_graph(
        self,
        prompt: str,
        *,
        max_n_logits: int = 10,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
        batch_size: int = 512,
        max_feature_nodes: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[AttributionGraph, Graph]:
        """Generate a pruned attribution graph and convert to our format.

        Returns both the ActiveCircuitDiscovery AttributionGraph and the
        raw circuit-tracer Graph for downstream manipulation.
        """
        raw_graph = self.get_raw_graph(
            prompt,
            max_n_logits=max_n_logits,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            verbose=verbose,
        )

        prune_result = prune_graph(raw_graph, node_threshold, edge_threshold)

        attribution_graph = self._convert_to_attribution_graph(
            raw_graph, prune_result, prompt
        )

        return attribution_graph, raw_graph

    def _build_all_nodes(self, graph: Graph) -> List[NodeInfo]:
        """Build NodeInfo for every node in the adjacency matrix.

        Node ordering in circuit-tracer adjacency matrix:
          [selected_features(0..n_sel-1),
           error_nodes(n_sel..n_sel + n_layers*n_tokens - 1),
           embed_nodes(...+n_tokens-1),
           logit_nodes(...+n_logits-1)]
        """
        n_tokens = graph.n_pos
        n_logits = len(graph.logit_targets)
        n_selected = len(graph.selected_features)
        n_layers = graph.cfg.n_layers
        n_error = n_layers * n_tokens
        total_nodes = graph.adjacency_matrix.shape[0]

        adj = graph.adjacency_matrix.abs()
        influence = adj.sum(dim=0) + adj.sum(dim=1)

        nodes: List[NodeInfo] = []

        for i in range(n_selected):
            feat_tensor = graph.selected_features[i]
            layer = int(graph.active_features[feat_tensor, 0].item())
            pos = int(graph.active_features[feat_tensor, 1].item())
            feat_idx = int(graph.active_features[feat_tensor, 2].item())
            act_val = float(graph.activation_values[i].item()) if i < len(graph.activation_values) else 0.0
            imp = float(influence[i].item()) if i < influence.shape[0] else 0.0

            nodes.append(NodeInfo(
                node_id=f"feature_L{layer}_P{pos}_F{feat_idx}",
                node_type="feature",
                layer=layer,
                position=pos,
                feature_idx=feat_idx,
                importance=imp,
                activation=act_val,
                raw_index=i,
            ))

        error_start = n_selected
        for layer_idx in range(n_layers):
            for pos_idx in range(n_tokens):
                idx = error_start + layer_idx * n_tokens + pos_idx
                if idx < total_nodes:
                    nodes.append(NodeInfo(
                        node_id=f"error_L{layer_idx}_P{pos_idx}",
                        node_type="error",
                        layer=layer_idx,
                        position=pos_idx,
                        feature_idx=-1,
                        importance=float(influence[idx].item()) if idx < influence.shape[0] else 0.0,
                        activation=0.0,
                        raw_index=idx,
                    ))

        embed_start = error_start + n_error
        for pos_idx in range(n_tokens):
            idx = embed_start + pos_idx
            if idx < total_nodes:
                nodes.append(NodeInfo(
                    node_id=f"embed_P{pos_idx}",
                    node_type="embed",
                    layer=-1,
                    position=pos_idx,
                    feature_idx=-1,
                    importance=float(influence[idx].item()) if idx < influence.shape[0] else 0.0,
                    activation=0.0,
                    raw_index=idx,
                ))

        logit_start = embed_start + n_tokens
        for li, target in enumerate(graph.logit_targets):
            idx = logit_start + li
            if idx < total_nodes:
                nodes.append(NodeInfo(
                    node_id=f"logit_{target.token_str}",
                    node_type="logit",
                    layer=n_layers,
                    position=-1,
                    feature_idx=target.vocab_idx,
                    importance=float(graph.logit_probabilities[li].item()),
                    activation=0.0,
                    raw_index=idx,
                ))

        return nodes

    def _convert_to_attribution_graph(
        self,
        raw_graph: Graph,
        prune_result: PruneResult,
        prompt: str,
    ) -> AttributionGraph:
        """Convert a circuit-tracer Graph + PruneResult to our AttributionGraph."""
        node_mask = prune_result.node_mask  # bool tensor [total_nodes]
        edge_mask = prune_result.edge_mask  # bool tensor [total_nodes, total_nodes]

        all_nodes = self._build_all_nodes(raw_graph)

        kept_indices = set(torch.where(node_mask)[0].tolist())

        kept_nodes = [n for n in all_nodes if n.raw_index in kept_indices]

        max_imp = max((n.importance for n in kept_nodes), default=1.0) or 1.0
        graph_nodes = []
        raw_to_id: Dict[int, str] = {}

        for info in kept_nodes:
            normalized_imp = min(1.0, info.importance / max_imp)
            graph_nodes.append(GraphNode(
                node_id=info.node_id,
                layer=max(0, info.layer),
                feature_id=info.feature_idx,
                importance=normalized_imp,
                description=f"{info.node_type}(L{info.layer},P{info.position},F{info.feature_idx})",
            ))
            raw_to_id[info.raw_index] = info.node_id

        adj = raw_graph.adjacency_matrix
        masked_adj = adj * edge_mask.float()
        nonzero = torch.nonzero(masked_adj, as_tuple=False)

        graph_edges = []
        for idx in range(nonzero.shape[0]):
            row = int(nonzero[idx, 0].item())
            col = int(nonzero[idx, 1].item())
            if row not in raw_to_id or col not in raw_to_id:
                continue
            weight = float(masked_adj[row, col].item())
            if abs(weight) < 1e-8:
                continue
            graph_edges.append(GraphEdge(
                source_id=raw_to_id[col],
                target_id=raw_to_id[row],
                weight=weight,
                confidence=min(1.0, abs(weight)),
                edge_type="eap_attribution",
            ))

        target_tokens = ", ".join(
            t.token_str for t in raw_graph.logit_targets
        ) if raw_graph.logit_targets else ""

        total_abs_weight = sum(abs(e.weight) for e in graph_edges)
        graph_confidence = min(1.0, total_abs_weight / max(1.0, len(graph_edges))) if graph_edges else 0.0

        return AttributionGraph(
            input_text=prompt,
            nodes=graph_nodes,
            edges=graph_edges,
            target_output=target_tokens,
            confidence=graph_confidence,
            metadata={
                "source": "circuit-tracer",
                "model": self.model_name,
                "transcoder_set": self.transcoder_ref,
                "n_raw_features": int(raw_graph.active_features.shape[0]),
                "n_selected_features": int(raw_graph.selected_features.shape[0]),
                "n_total_nodes": int(raw_graph.adjacency_matrix.shape[0]),
                "n_kept_nodes": len(graph_nodes),
                "n_kept_edges": len(graph_edges),
            },
        )

    def get_feature_activations(
        self, prompt: str, *, max_n_logits: int = 10
    ) -> Dict[str, Any]:
        """Get feature activation data for a prompt without full graph computation."""
        graph = self.get_raw_graph(prompt, max_n_logits=max_n_logits)
        return {
            "active_features": graph.active_features.cpu(),
            "activation_values": graph.activation_values.cpu(),
            "selected_features": graph.selected_features.cpu(),
            "n_active": int(graph.active_features.shape[0]),
            "n_selected": int(graph.selected_features.shape[0]),
        }

    def get_feature_importance_ranking(
        self,
        prompt: str,
        *,
        node_threshold: float = 0.8,
    ) -> List[Tuple[int, int, int, float]]:
        """Rank features by importance for Active Inference agent.

        Returns list of (layer, position, feature_idx, importance_score)
        sorted by descending importance.
        """
        raw_graph = self.get_raw_graph(prompt)
        prune_result = prune_graph(raw_graph, node_threshold=node_threshold)
        node_mask = prune_result.node_mask

        n_selected = len(raw_graph.selected_features)
        adj = raw_graph.adjacency_matrix
        influence = adj.abs().sum(dim=0)[:n_selected] + adj.abs().sum(dim=1)[:n_selected]

        rankings = []
        for i in range(n_selected):
            if not node_mask[i]:
                continue
            feat_tensor = raw_graph.selected_features[i]
            layer = int(raw_graph.active_features[feat_tensor, 0].item())
            pos = int(raw_graph.active_features[feat_tensor, 1].item())
            feat_idx = int(raw_graph.active_features[feat_tensor, 2].item())
            imp = float(influence[i].item())
            rankings.append((layer, pos, feat_idx, imp))

        rankings.sort(key=lambda x: x[3], reverse=True)
        return rankings
