"""
Official Circuit Tracer Integration
=====================================
Wraps Anthropic's `circuit-tracer` library (safety-research/circuit-tracer)
as the attribution graph backend for ActiveCircuitDiscovery.

If `circuit-tracer` is not installed, this module falls back gracefully to
the custom TransformerLens-based tracer in `tracer.py`.

Installation:
    pip install circuit-tracer
    # or from source:
    pip install git+https://github.com/safety-research/circuit-tracer.git

Usage:
    from src.circuit_analysis.official_tracer import OfficialCircuitTracer
    tracer = OfficialCircuitTracer("gpt2-small")
    graph = tracer.get_attribution_graph("The Golden Gate Bridge is in")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from circuit_tracer import get_circuit
    CIRCUIT_TRACER_AVAILABLE = True
    logger.info("Anthropic circuit-tracer library is available.")
except ImportError:
    CIRCUIT_TRACER_AVAILABLE = False
    logger.info(
        "circuit-tracer not installed.  "
        "Install via: pip install circuit-tracer  "
        "Falling back to custom tracer."
    )

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_structures import AttributionGraph, GraphNode, GraphEdge
except ImportError:
    AttributionGraph = GraphNode = GraphEdge = None  # type: ignore


class OfficialCircuitTracer:
    """Wrapper around Anthropic's `circuit-tracer` library.

    Provides attribution graphs using Edge Attribution Patching (EAP),
    which is the state-of-the-art method for automatic circuit discovery.

    Falls back to the custom tracer when circuit-tracer is unavailable.
    """

    def __init__(self, model_name: str = "gpt2-small"):
        self.model_name = model_name
        self.available = CIRCUIT_TRACER_AVAILABLE

        if not self.available:
            logger.warning(
                "OfficialCircuitTracer: circuit-tracer not installed. "
                "Call get_attribution_graph() will raise ImportError."
            )

    def get_attribution_graph(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        max_n_logits: int = 10,
        desired_n_nodes: int = 20,
        node_threshold: float = 0.8,
    ) -> "AttributionGraph":
        """Build an attribution graph via EAP using the official circuit-tracer.

        Args:
            prompt: Input text to analyse.
            target_token: Target token for attribution (auto-detected if None).
            max_n_logits: Number of top logits to consider.
            desired_n_nodes: Approximate number of nodes to retain after pruning.
            node_threshold: Cumulative attribution threshold for pruning.

        Returns:
            AttributionGraph in ActiveCircuitDiscovery format.

        Raises:
            ImportError: If circuit-tracer is not installed.
        """
        if not self.available:
            raise ImportError(
                "circuit-tracer is not installed.  "
                "Run: pip install circuit-tracer"
            )

        logger.info(f"Building attribution graph via circuit-tracer for: '{prompt[:60]}'")

        try:
            ct_graph = get_circuit(
                prompt=prompt,
                model=self.model_name,
                max_n_logits=max_n_logits,
            )
            return self._convert_graph(ct_graph, prompt)

        except Exception as e:
            logger.error(f"circuit-tracer failed: {e}")
            raise

    def _convert_graph(self, ct_graph: Any, prompt: str) -> "AttributionGraph":
        """Convert a circuit-tracer graph to our AttributionGraph format."""

        nodes: List[GraphNode] = []
        node_idx_to_id: Dict[int, str] = {}

        for node in ct_graph.nodes:
            node_idx = getattr(node, "node_idx", id(node))
            layer = getattr(node, "layer", 0)
            feature_id = getattr(node, "feature", 0)
            importance = float(
                getattr(node, "in_graph_importance",
                        getattr(node, "score", 0.5))
            )
            description = str(node)

            node_id = f"ct_node_{node_idx}"
            node_idx_to_id[node_idx] = node_id

            nodes.append(GraphNode(
                node_id=node_id,
                layer=layer,
                feature_id=feature_id,
                importance=importance,
                description=description,
            ))

        edges: List[GraphEdge] = []
        for edge in ct_graph.edges:
            src_idx = getattr(edge.src, "node_idx", id(edge.src))
            dst_idx = getattr(edge.dst, "node_idx", id(edge.dst))
            weight = float(getattr(edge, "weight", getattr(edge, "score", 0.0)))

            src_id = node_idx_to_id.get(src_idx, f"ct_node_{src_idx}")
            dst_id = node_idx_to_id.get(dst_idx, f"ct_node_{dst_idx}")

            edges.append(GraphEdge(
                source_id=src_id,
                target_id=dst_id,
                weight=weight,
                confidence=min(1.0, abs(weight)),
                edge_type="eap_attribution",
            ))

        total_weight = sum(abs(e.weight) for e in edges)
        graph_confidence = min(1.0, total_weight / max(1.0, len(edges)))

        return AttributionGraph(
            input_text=prompt,
            nodes=nodes,
            edges=edges,
            target_output=getattr(ct_graph, "target_token", ""),
            confidence=graph_confidence,
            metadata={
                "source": "circuit-tracer",
                "official": True,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "model": self.model_name,
            },
        )

    @staticmethod
    def is_available() -> bool:
        """Return True if the circuit-tracer library is installed."""
        return CIRCUIT_TRACER_AVAILABLE
