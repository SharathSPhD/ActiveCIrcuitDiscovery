"""
Multi-step Reasoning Benchmark
================================
Tests circuit discovery on multi-step reasoning tasks that require
information to flow through multiple processing stages.

Inspired by Anthropic's "Biology of LLMs" findings on multi-hop
factual reasoning circuits.

Task types:
  - Geographic: "Dallas is in Texas. The capital of Texas is" -> "Austin"
  - Compositional: "The mother of the president of the US is" -> requires composition
  - Temporal: "World War II ended in 1945. Two years later, in" -> "1947"
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


MULTISTEP_PROMPTS = [
    {
        "prompt": "Dallas is in Texas. The capital of Texas is",
        "answer": " Austin",
        "n_hops": 2,
        "category": "geographic",
    },
    {
        "prompt": "The Eiffel Tower is in Paris. Paris is the capital of",
        "answer": " France",
        "n_hops": 2,
        "category": "geographic",
    },
    {
        "prompt": "Water is H2O. The number of hydrogen atoms in water is",
        "answer": " two",
        "n_hops": 2,
        "category": "compositional",
    },
    {
        "prompt": "Einstein developed relativity. The theory of relativity describes",
        "answer": " gravity",
        "n_hops": 2,
        "category": "compositional",
    },
    {
        "prompt": "Python was created by Guido van Rossum. Guido van Rossum is from the",
        "answer": " Netherlands",
        "n_hops": 2,
        "category": "compositional",
    },
    {
        "prompt": "Tokyo is in Japan. Japan is in the continent of",
        "answer": " Asia",
        "n_hops": 2,
        "category": "geographic",
    },
]


@dataclass
class MultistepResult:
    """Result from a single multi-step reasoning trial."""
    prompt: str
    answer: str
    n_hops: int
    category: str
    correct: bool
    answer_prob: float
    n_interventions: int
    converged: bool
    n_intermediate_features: int
    n_input_features: int
    n_output_features: int
    elapsed_seconds: float
    correspondence: Optional[CorrespondenceResult] = None


@dataclass
class MultistepBenchmarkResult:
    """Aggregated multi-step reasoning benchmark results."""
    trials: List[MultistepResult]
    accuracy: float
    mean_n_interventions: float
    mean_intermediate_ratio: float
    correspondence: Optional[CorrespondenceResult] = None
    elapsed_total: float = 0.0


class MultistepReasoningBenchmark:
    """Multi-step reasoning benchmark for circuit discovery.

    Tests whether Active Inference can identify the multi-hop
    information flow patterns in LLM circuits.
    """

    def __init__(
        self,
        backend: CircuitTracerBackend,
        *,
        max_interventions: int = 40,
        n_prompts: Optional[int] = None,
    ):
        self.backend = backend
        self.max_interventions = max_interventions
        self.prompts = MULTISTEP_PROMPTS[:n_prompts] if n_prompts else MULTISTEP_PROMPTS

    def run(self, verbose: bool = True) -> MultistepBenchmarkResult:
        """Run the full multi-step reasoning benchmark."""
        start_time = time.time()
        trials = []

        for i, prompt_data in enumerate(self.prompts):
            if verbose:
                logger.info(
                    f"Multistep trial {i+1}/{len(self.prompts)}: "
                    f"{prompt_data['prompt'][:50]}..."
                )

            trial = self._run_single_trial(prompt_data, verbose=verbose)
            trials.append(trial)

        accuracy = float(np.mean([t.correct for t in trials]))
        mean_interv = float(np.mean([t.n_interventions for t in trials]))

        intermediate_ratios = []
        for t in trials:
            total = t.n_input_features + t.n_intermediate_features + t.n_output_features
            if total > 0:
                intermediate_ratios.append(t.n_intermediate_features / total)
        mean_intermediate_ratio = float(np.mean(intermediate_ratios)) if intermediate_ratios else 0.0

        elapsed = time.time() - start_time

        result = MultistepBenchmarkResult(
            trials=trials,
            accuracy=accuracy,
            mean_n_interventions=mean_interv,
            mean_intermediate_ratio=mean_intermediate_ratio,
            elapsed_total=elapsed,
        )

        if verbose:
            logger.info(
                f"Multistep Benchmark complete: "
                f"accuracy={accuracy:.3f}, "
                f"mean_interventions={mean_interv:.1f}, "
                f"intermediate_ratio={mean_intermediate_ratio:.3f}"
            )

        return result

    def _run_single_trial(
        self, prompt_data: Dict[str, Any], verbose: bool = True,
    ) -> MultistepResult:
        """Run a single multi-step reasoning trial."""
        start = time.time()

        agent = ActiveInferencePOMDPAgent()
        engine = InterventionEngine(self.backend, agent)

        prompt = prompt_data["prompt"]
        answer = prompt_data["answer"]

        model = engine.model
        tokenizer = model.tokenizer

        logits = model(prompt)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)

        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_prob = 0.0
        if answer_ids:
            answer_prob = float(probs[answer_ids[0]].item())
        correct = int(last_logits.argmax().item()) == answer_ids[0] if answer_ids else False

        results = engine.run_active_discovery(
            prompt,
            max_interventions=self.max_interventions,
            verbose=verbose,
        )

        role_predictions = results.get("role_predictions", {})
        n_input = sum(1 for r in role_predictions.values() if r == "input")
        n_intermediate = sum(1 for r in role_predictions.values() if r == "intermediate")
        n_output = sum(1 for r in role_predictions.values() if r == "output")

        efe_vals = np.array(results.get("efe_history", [0.0]))
        effect_vals = np.array([m.effect_size for m in results.get("measurements", [])])

        correspondence = None
        if len(efe_vals) >= 3 and len(effect_vals) >= 3:
            min_len = min(len(efe_vals), len(effect_vals))
            correspondence = compute_correspondence(
                efe_vals[:min_len], effect_vals[:min_len],
            )

        elapsed = time.time() - start

        return MultistepResult(
            prompt=prompt,
            answer=answer,
            n_hops=prompt_data["n_hops"],
            category=prompt_data["category"],
            correct=correct,
            answer_prob=answer_prob,
            n_interventions=results.get("n_interventions", 0),
            converged=results.get("converged", False),
            n_intermediate_features=n_intermediate,
            n_input_features=n_input,
            n_output_features=n_output,
            elapsed_seconds=elapsed,
            correspondence=correspondence,
        )
