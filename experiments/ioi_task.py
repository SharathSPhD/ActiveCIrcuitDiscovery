"""
Indirect Object Identification (IOI) Task
==========================================
Gold-standard circuit discovery benchmark from Wang et al. (2022).

The IOI task tests whether a model correctly identifies the indirect object
in sentences of the form:
    "When John and Mary went to the store, John gave the milk to [Mary]"

The target token is the IO name (e.g. "Mary"), not the S (subject) name ("John").

Usage:
    from experiments.ioi_task import IOIDataset, ioi_logit_diff_metric
    dataset = IOIDataset(n_samples=50, random_seed=42)
    runner.run_experiment(dataset.prompts)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IOI_TEMPLATES: List[str] = [
    "When {A} and {B} went to the store, {B} gave the milk to",
    "After {A} and {B} visited the park, {A} handed the ball to",
    "While {A} and {B} were at school, {B} showed the book to",
    "When {A} and {B} went to the restaurant, {A} passed the menu to",
    "After {A} and {B} arrived at the office, {B} sent the email to",
    "While {A} and {B} were at home, {A} gave the keys to",
    "When {A} and {B} travelled together, {B} offered the ticket to",
    "After {A} and {B} attended the meeting, {A} forwarded the report to",
]

NAME_PAIRS: List[Tuple[str, str]] = [
    ("John", "Mary"),
    ("Tom", "Alice"),
    ("Bob", "Sarah"),
    ("James", "Emma"),
    ("David", "Laura"),
    ("Michael", "Jessica"),
    ("Daniel", "Rachel"),
    ("Robert", "Hannah"),
]


@dataclass
class IOISample:
    """A single IOI task sample."""
    prompt: str
    io_name: str      # Indirect Object — the correct answer
    s_name: str       # Subject — the distractor
    template: str


@dataclass
class IOIDataset:
    """Dataset of IOI samples for circuit discovery evaluation."""

    n_samples: int = 50
    random_seed: int = 42
    samples: List[IOISample] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.random_seed)
        for _ in range(self.n_samples):
            template = rng.choice(IOI_TEMPLATES)
            a_name, b_name = rng.choice(NAME_PAIRS)
            # Randomly swap A/B so IO can be either name
            if rng.random() > 0.5:
                a_name, b_name = b_name, a_name
            prompt = template.format(A=a_name, B=b_name)
            # Determine IO: the name that appears after the last comma in the template
            # The filler words determine which name is IO
            if "{B}" in template.split(",")[-1]:
                io_name, s_name = b_name, a_name
            else:
                io_name, s_name = a_name, b_name
            self.samples.append(IOISample(
                prompt=prompt,
                io_name=io_name,
                s_name=s_name,
                template=template
            ))

    @property
    def prompts(self) -> List[str]:
        return [s.prompt for s in self.samples]

    @property
    def io_names(self) -> List[str]:
        return [s.io_name for s in self.samples]

    @property
    def s_names(self) -> List[str]:
        return [s.s_name for s in self.samples]


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def ioi_logit_diff_metric(
    model,
    prompt: str,
    io_name: str,
    s_name: str,
    device: str = "cpu"
) -> float:
    """Compute the IOI logit difference: logit(IO) - logit(S).

    Higher is better — the model should assign higher probability to the IO name.

    Args:
        model: A TransformerLens HookedTransformer instance.
        prompt: The IOI prompt string (without trailing space).
        io_name: The indirect object name (correct completion).
        s_name: The subject name (incorrect completion).
        device: Torch device string.

    Returns:
        Logit difference (float).  Positive = correct.
    """
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)

    last_logits = logits[0, -1, :]  # shape (vocab,)

    io_token_id = model.to_single_token(" " + io_name)
    s_token_id = model.to_single_token(" " + s_name)

    logit_diff = (last_logits[io_token_id] - last_logits[s_token_id]).item()
    return logit_diff


def ioi_accuracy(model, dataset: IOIDataset, device: str = "cpu") -> Dict[str, float]:
    """Evaluate model accuracy on the IOI task.

    Returns a dict with:
        mean_logit_diff  — average logit(IO) - logit(S)
        accuracy         — fraction of samples where IO > S
        std_logit_diff   — standard deviation of logit differences
    """
    diffs = []
    for sample in dataset.samples:
        try:
            diff = ioi_logit_diff_metric(model, sample.prompt, sample.io_name,
                                         sample.s_name, device=device)
            diffs.append(diff)
        except Exception:
            pass

    diffs_arr = np.array(diffs)
    return {
        "mean_logit_diff": float(diffs_arr.mean()) if len(diffs_arr) > 0 else 0.0,
        "std_logit_diff": float(diffs_arr.std()) if len(diffs_arr) > 0 else 0.0,
        "accuracy": float((diffs_arr > 0).mean()) if len(diffs_arr) > 0 else 0.0,
        "n_samples": len(diffs_arr),
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run_ioi_circuit_discovery(
    config_path: Optional[Path] = None,
    n_samples: int = 30,
    output_dir: str = "results/ioi_experiment",
) -> None:
    """Run circuit discovery on the IOI benchmark task."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from experiments.runner import YorKExperimentRunner

    dataset = IOIDataset(n_samples=n_samples)
    runner = YorKExperimentRunner(config_path)
    runner.setup_experiment()

    print(f"Running IOI circuit discovery on {n_samples} samples ...")
    result = runner.run_experiment(dataset.prompts)
    runner.save_results(result, output_dir)

    # Evaluate IOI accuracy
    accuracy_stats = ioi_accuracy(runner.tracer.model, dataset,
                                   device=runner.tracer.device)
    print("\nIOI Accuracy:")
    for k, v in accuracy_stats.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nRQ1 passed: {result.rq1_passed}")
    print(f"RQ2 passed: {result.rq2_passed}")
    print(f"RQ3 passed: {result.rq3_passed}")


if __name__ == "__main__":
    run_ioi_circuit_discovery()
