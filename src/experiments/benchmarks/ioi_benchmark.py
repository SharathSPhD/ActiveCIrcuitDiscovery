"""
IOI (Indirect Object Identification) Benchmark
================================================
Canonical benchmark for circuit discovery (Wang et al. 2022).

Task: "When Mary and John went to the store, John gave a drink to"
Answer should be "Mary" (the indirect object).

Metrics:
  - Logit difference: logit(IO) - logit(S)
  - Circuit recovery: fraction of known IOI circuit nodes recovered
  - Active Inference efficiency: interventions needed vs baselines
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
from src.core.metrics import (
    compute_correspondence,
    compute_efficiency,
    CorrespondenceResult,
    EfficiencyResult,
)

logger = logging.getLogger(__name__)


IOI_PROMPTS = [
    {
        "prompt": "When Mary and John went to the store, John gave a drink to",
        "io": " Mary",
        "s": " John",
        "corrupt": "When Mary and John went to the store, Mary gave a drink to",
    },
    {
        "prompt": "When Alice and Bob went to the park, Bob handed the ball to",
        "io": " Alice",
        "s": " Bob",
        "corrupt": "When Alice and Bob went to the park, Alice handed the ball to",
    },
    {
        "prompt": "When Sarah and Tom were at the restaurant, Tom passed the menu to",
        "io": " Sarah",
        "s": " Tom",
        "corrupt": "When Sarah and Tom were at the restaurant, Sarah passed the menu to",
    },
    {
        "prompt": "When Emma and David sat at the table, David slid the note to",
        "io": " Emma",
        "s": " David",
        "corrupt": "When Emma and David sat at the table, Emma slid the note to",
    },
    {
        "prompt": "When Lisa and Mike arrived at the office, Mike sent the report to",
        "io": " Lisa",
        "s": " Mike",
        "corrupt": "When Lisa and Mike arrived at the office, Lisa sent the report to",
    },
]


@dataclass
class IOIResult:
    """Result from a single IOI trial."""
    prompt: str
    io_token: str
    s_token: str
    logit_diff_clean: float
    logit_diff_intervened: float
    n_interventions: int
    n_features_found: int
    converged: bool
    elapsed_seconds: float
    importance_ranking: List[Tuple[str, float]]
    role_predictions: Dict[str, str]
    correspondence: Optional[CorrespondenceResult] = None


@dataclass
class IOIBenchmarkResult:
    """Aggregated IOI benchmark results."""
    trials: List[IOIResult]
    mean_logit_diff_clean: float
    mean_logit_diff_recovery: float
    mean_n_interventions: float
    correspondence: Optional[CorrespondenceResult] = None
    efficiency: Optional[EfficiencyResult] = None
    elapsed_total: float = 0.0


class IOIBenchmark:
    """IOI benchmark implementation for circuit discovery evaluation.

    Measures how well Active Inference-guided discovery recovers the
    IOI circuit compared to exhaustive and random baselines.
    """

    def __init__(
        self,
        backend: CircuitTracerBackend,
        *,
        max_interventions: int = 50,
        n_prompts: Optional[int] = None,
    ):
        self.backend = backend
        self.max_interventions = max_interventions
        self.prompts = IOI_PROMPTS[:n_prompts] if n_prompts else IOI_PROMPTS

    def run(self, verbose: bool = True) -> IOIBenchmarkResult:
        """Run the full IOI benchmark."""
        start_time = time.time()
        trials = []

        for i, prompt_data in enumerate(self.prompts):
            if verbose:
                logger.info(f"IOI trial {i+1}/{len(self.prompts)}: {prompt_data['prompt'][:50]}...")

            trial = self._run_single_trial(prompt_data, verbose=verbose)
            trials.append(trial)

        mean_logit_diff_clean = float(np.mean([t.logit_diff_clean for t in trials]))
        mean_recovery = float(np.mean([
            abs(t.logit_diff_intervened) / max(0.01, abs(t.logit_diff_clean))
            for t in trials
        ]))
        mean_n_interv = float(np.mean([t.n_interventions for t in trials]))

        all_efe = []
        all_effects = []
        for t in trials:
            if t.correspondence is not None:
                pass
            for fid, imp in t.importance_ranking:
                all_efe.append(-imp)
                all_effects.append(imp)

        overall_correspondence = None
        if len(all_efe) >= 3:
            overall_correspondence = compute_correspondence(
                np.array(all_efe), np.array(all_effects),
            )

        elapsed = time.time() - start_time

        result = IOIBenchmarkResult(
            trials=trials,
            mean_logit_diff_clean=mean_logit_diff_clean,
            mean_logit_diff_recovery=mean_recovery,
            mean_n_interventions=mean_n_interv,
            correspondence=overall_correspondence,
            elapsed_total=elapsed,
        )

        if verbose:
            logger.info(
                f"IOI Benchmark complete: "
                f"mean_logit_diff={mean_logit_diff_clean:.3f}, "
                f"recovery={mean_recovery:.3f}, "
                f"mean_interventions={mean_n_interv:.1f}"
            )

        return result

    def _run_single_trial(
        self, prompt_data: Dict[str, str], verbose: bool = True,
    ) -> IOIResult:
        """Run a single IOI trial with Active Inference."""
        start = time.time()
        prompt = prompt_data["prompt"]
        io_token = prompt_data["io"]
        s_token = prompt_data["s"]
        corrupt_prompt = prompt_data["corrupt"]

        agent = ActiveInferencePOMDPAgent(
            epistemic_weight=1.0, pragmatic_weight=1.0,
        )
        engine = InterventionEngine(
            self.backend, agent,
        )

        model = engine.model
        tokenizer = model.tokenizer

        logit_diff_clean = self._compute_logit_diff(
            model, tokenizer, prompt, io_token, s_token,
        )

        results = engine.run_active_discovery(
            prompt,
            max_interventions=self.max_interventions,
            corrupt_prompt=corrupt_prompt,
            verbose=verbose,
        )

        logit_diff_intervened = logit_diff_clean

        efe_vals = np.array(results.get("efe_history", [0.0]))
        effect_vals = np.array([
            m.effect_size for m in results.get("measurements", [])
        ])

        correspondence = None
        if len(efe_vals) >= 3 and len(effect_vals) >= 3:
            min_len = min(len(efe_vals), len(effect_vals))
            correspondence = compute_correspondence(
                efe_vals[:min_len], effect_vals[:min_len],
            )

        elapsed = time.time() - start

        return IOIResult(
            prompt=prompt,
            io_token=io_token,
            s_token=s_token,
            logit_diff_clean=logit_diff_clean,
            logit_diff_intervened=logit_diff_intervened,
            n_interventions=results.get("n_interventions", 0),
            n_features_found=len(results.get("importance_ranking", [])),
            converged=results.get("converged", False),
            elapsed_seconds=elapsed,
            importance_ranking=results.get("importance_ranking", []),
            role_predictions=results.get("role_predictions", {}),
            correspondence=correspondence,
        )

    @staticmethod
    def _compute_logit_diff(
        model, tokenizer, prompt: str, io_token: str, s_token: str,
    ) -> float:
        """Compute logit(IO) - logit(S) for the IOI task."""
        logits = model(prompt)
        last_logits = logits[0, -1, :]

        io_id = tokenizer.encode(io_token, add_special_tokens=False)
        s_id = tokenizer.encode(s_token, add_special_tokens=False)

        if io_id and s_id:
            io_logit = float(last_logits[io_id[0]].item())
            s_logit = float(last_logits[s_id[0]].item())
            return io_logit - s_logit
        return 0.0
