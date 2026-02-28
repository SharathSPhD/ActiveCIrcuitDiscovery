"""
Golden Gate Bridge Benchmark
==============================
Feature steering demonstration inspired by Anthropic's "Golden Gate
Claude" experiment. Tests whether Active Inference can identify the
key features responsible for a concept and steer them.

Task: Identify features that activate for "Golden Gate Bridge" and
demonstrate that steering these features changes model behavior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.circuit_analysis.circuit_tracer_backend import CircuitTracerBackend
from src.active_inference.pomdp_agent import ActiveInferencePOMDPAgent
from src.experiments.intervention_engine import InterventionEngine
from src.core.metrics import compute_correspondence, CorrespondenceResult

logger = logging.getLogger(__name__)


STEERING_PROMPTS = [
    {
        "concept_prompt": "The Golden Gate Bridge is located in",
        "concept": "golden_gate_bridge",
        "test_prompts": [
            "I visited a famous landmark. It was the",
            "The most iconic bridge in the world is the",
            "San Francisco is known for its",
        ],
    },
    {
        "concept_prompt": "The Eiffel Tower in Paris is a",
        "concept": "eiffel_tower",
        "test_prompts": [
            "The tallest structure in Paris is the",
            "When visiting France, tourists often see the",
        ],
    },
    {
        "concept_prompt": "Python is a programming language used for",
        "concept": "python_programming",
        "test_prompts": [
            "The best language for data science is",
            "Machine learning code is often written in",
        ],
    },
]


@dataclass
class SteeringTrialResult:
    """Result from a single steering trial."""
    concept: str
    concept_prompt: str
    n_features_identified: int
    top_features: List[Tuple[str, float]]
    steering_effects: List[Dict[str, Any]]
    mean_steering_effect: float
    n_interventions: int
    converged: bool
    elapsed_seconds: float


@dataclass
class GoldenGateBenchmarkResult:
    """Aggregated Golden Gate benchmark results."""
    trials: List[SteeringTrialResult]
    mean_features_identified: float
    mean_steering_effect: float
    mean_n_interventions: float
    elapsed_total: float = 0.0


class GoldenGateBenchmark:
    """Golden Gate Bridge / feature steering benchmark.

    1. Discovers features for a concept via Active Inference
    2. Steers top features and measures effect on test prompts
    3. Validates that AI-identified features are causally relevant
    """

    def __init__(
        self,
        backend: CircuitTracerBackend,
        *,
        max_interventions: int = 30,
        n_concepts: Optional[int] = None,
        steering_multiplier: float = 5.0,
    ):
        self.backend = backend
        self.max_interventions = max_interventions
        self.concepts = STEERING_PROMPTS[:n_concepts] if n_concepts else STEERING_PROMPTS
        self.steering_multiplier = steering_multiplier

    def run(self, verbose: bool = True) -> GoldenGateBenchmarkResult:
        """Run the full Golden Gate benchmark."""
        start_time = time.time()
        trials = []

        for i, concept_data in enumerate(self.concepts):
            if verbose:
                logger.info(
                    f"Steering trial {i+1}/{len(self.concepts)}: "
                    f"{concept_data['concept']}"
                )
            trial = self._run_single_trial(concept_data, verbose=verbose)
            trials.append(trial)

        elapsed = time.time() - start_time

        result = GoldenGateBenchmarkResult(
            trials=trials,
            mean_features_identified=float(np.mean([t.n_features_identified for t in trials])),
            mean_steering_effect=float(np.mean([t.mean_steering_effect for t in trials])),
            mean_n_interventions=float(np.mean([t.n_interventions for t in trials])),
            elapsed_total=elapsed,
        )

        if verbose:
            logger.info(
                f"Golden Gate Benchmark complete: "
                f"mean_features={result.mean_features_identified:.1f}, "
                f"mean_steering_effect={result.mean_steering_effect:.4f}"
            )

        return result

    def _run_single_trial(
        self, concept_data: Dict[str, Any], verbose: bool = True,
    ) -> SteeringTrialResult:
        """Discover features for a concept and test steering."""
        start = time.time()

        agent = ActiveInferencePOMDPAgent()
        engine = InterventionEngine(
            self.backend, agent,
            steering_multiplier=self.steering_multiplier,
        )

        concept_prompt = concept_data["concept_prompt"]

        discovery_results = engine.run_active_discovery(
            concept_prompt,
            max_interventions=self.max_interventions,
            verbose=verbose,
        )

        importance_ranking = discovery_results.get("importance_ranking", [])
        top_features = importance_ranking[:5]

        steering_effects = []
        for test_prompt in concept_data.get("test_prompts", []):
            for fid, imp in top_features[:3]:
                parts = fid.split("_")
                layer = int(parts[1][1:]) if len(parts) > 1 else 0
                feat_idx = int(parts[3][1:]) if len(parts) > 3 else 0

                measurement = engine.perform_steering(
                    test_prompt, layer, feat_idx,
                    multiplier=self.steering_multiplier,
                )

                steering_effects.append({
                    "test_prompt": test_prompt,
                    "feature_id": fid,
                    "effect_size": measurement.effect_size,
                    "kl_divergence": measurement.kl_divergence,
                    "prob_change": measurement.target_token_prob_change,
                })

        mean_effect = float(np.mean([e["effect_size"] for e in steering_effects])) if steering_effects else 0.0

        elapsed = time.time() - start

        return SteeringTrialResult(
            concept=concept_data["concept"],
            concept_prompt=concept_prompt,
            n_features_identified=len(importance_ranking),
            top_features=top_features,
            steering_effects=steering_effects,
            mean_steering_effect=mean_effect,
            n_interventions=discovery_results.get("n_interventions", 0),
            converged=discovery_results.get("converged", False),
            elapsed_seconds=elapsed,
        )
