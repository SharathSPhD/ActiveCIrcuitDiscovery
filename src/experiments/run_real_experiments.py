"""
Real Experiment Runner for Active Circuit Discovery
=====================================================
Runs actual experiments on Gemma-2-2B using circuit-tracer attribution graphs
and active inference-guided feature selection.

No mocks. No fabrication. Real interventions via feature_intervention API.
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from circuit_tracer.utils.create_graph_files import create_graph_files
except ImportError:
    create_graph_files = None

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


IOI_PROMPTS = [
    "When John and Mary went to the store, John gave the bag to",
    "After Alice and Bob finished lunch, Alice handed the receipt to",
    "While Sarah and Tom were at the park, Sarah threw the ball to",
    "When Emma and David arrived at the office, Emma passed the keys to",
    "As Lisa and Mike left the restaurant, Lisa returned the coat to",
]

STEERING_PROMPTS = [
    "The Golden Gate Bridge is",
    "The Eiffel Tower is located in",
    "Mount Everest is the tallest",
    "The Great Wall of China was built",
    "The Statue of Liberty stands in",
]

MULTISTEP_PROMPTS = [
    "If Alice is taller than Bob, and Bob is taller than Carol, then the tallest person is",
    "The capital of France is Paris. Paris is in Europe. The continent containing Paris is",
    "All dogs are animals. Fido is a dog. Therefore Fido is",
]

# Multi-domain prompts from IRP (5 cognitive domains)
DOMAIN_PROMPTS: Dict[str, List[str]] = {
    "geography": [
        "The capital of France is",
        "The Golden Gate Bridge connects San Francisco to",
    ],
    "math": [
        "The square root of 64 is",
        "If 2 + 3 = 5 then 3 + 4 =",
    ],
    "science": [
        "Water is made of hydrogen and",
        "The speed of light is approximately",
    ],
    "logic": [
        "All mammals are warm-blooded. A whale is a mammal. Therefore a whale is",
        "All birds have wings. A penguin is a bird. Therefore a penguin has",
    ],
    "history": [
        "The year World War II ended was",
        "The first person to walk on the moon was",
    ],
}

# Model configurations: model_name, transcoder_set, n_layers (for layer bin boundaries)
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "gemma": {
        "model_name": "google/gemma-2-2b",
        "transcoder_set": "gemma",
        "n_layers": 26,
    },
    "llama": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "transcoder_set": "llama",
        "n_layers": 16,
    },
}


class ActiveInferenceSelector:
    """Feature selector combining graph importance with uncertainty-weighted exploration."""

    def __init__(self, candidates: List[Dict], exploration_weight: float = 2.0):
        self.candidates = candidates
        self.n = len(candidates)
        self.exploration_weight = exploration_weight
        self.observed: set = set()
        self.uncertainties = np.ones(self.n)
        self.observed_kls: Dict[int, float] = {}
        self.layer_kl_sum: Dict[int, float] = defaultdict(float)
        self.layer_kl_count: Dict[int, int] = defaultdict(int)
        self.layer_prior: Dict[int, float] = defaultdict(lambda: 1.0)
        self.selection_order: List[int] = []

    def select_next(self) -> Tuple[int, Dict]:
        scores = np.full(self.n, -np.inf)
        for i, feat in enumerate(self.candidates):
            if i in self.observed:
                continue
            pragmatic = feat['imp'] * self.layer_prior[feat['layer']]
            epistemic = self.uncertainties[i] * self.exploration_weight
            scores[i] = pragmatic + epistemic

        best_idx = int(np.argmax(scores))
        self.selection_order.append(best_idx)
        return best_idx, self.candidates[best_idx]

    def update(self, idx: int, kl_value: float) -> None:
        self.observed.add(idx)
        self.observed_kls[idx] = kl_value
        feat = self.candidates[idx]
        self.uncertainties[idx] = 0.0
        self.layer_kl_sum[feat['layer']] += kl_value
        self.layer_kl_count[feat['layer']] += 1
        layer_mean = self.layer_kl_sum[feat['layer']] / self.layer_kl_count[feat['layer']]
        global_mean = sum(self.observed_kls.values()) / len(self.observed_kls) if self.observed_kls else 0.001
        if global_mean > 0:
            self.layer_prior[feat['layer']] = 1.0 + (layer_mean / global_mean - 1.0) * 0.5
        for j, other in enumerate(self.candidates):
            if j in self.observed:
                continue
            if other['layer'] == feat['layer']:
                self.uncertainties[j] *= 0.7
            if abs(other['pos'] - feat['pos']) <= 1:
                self.uncertainties[j] *= 0.9


def extract_candidates(
    raw_graph, max_per_layer: int = 5, max_total: int = 60
) -> List[Dict[str, Any]]:
    """Extract diverse candidate features across all layers."""
    n_sel = len(raw_graph.selected_features)
    adj = raw_graph.adjacency_matrix
    infl = adj.abs().sum(0)[:n_sel] + adj.abs().sum(1)[:n_sel]
    mi = infl.max().item() or 1.0

    by_layer: Dict[int, list] = {}
    for i in range(n_sel):
        ft = raw_graph.selected_features[i]
        layer = int(raw_graph.active_features[ft, 0].item())
        pos = int(raw_graph.active_features[ft, 1].item())
        fidx = int(raw_graph.active_features[ft, 2].item())
        act = float(raw_graph.activation_values[i].item())
        imp = float(infl[i].item()) / mi
        entry = dict(
            layer=layer, pos=pos, fidx=fidx, act=act, imp=imp,
            fid=f'L{layer}_P{pos}_F{fidx}'
        )
        by_layer.setdefault(layer, []).append(entry)

    for l in by_layer:
        by_layer[l].sort(key=lambda x: x['imp'], reverse=True)

    candidates = []
    for l in sorted(by_layer.keys()):
        candidates.extend(by_layer[l][:max_per_layer])
    candidates.sort(key=lambda x: x['imp'], reverse=True)
    return candidates[:max_total]


def ablate_feature(
    model: ReplacementModel, prompt: str, feat: Dict,
    clean_probs: torch.Tensor, clean_last: torch.Tensor
) -> Tuple[float, float]:
    """Ablate a single transcoder feature and return (KL_divergence, logit_diff)."""
    iv, _ = model.feature_intervention(
        prompt, [(feat['layer'], feat['pos'], feat['fidx'], 0)],
        return_activations=False
    )
    iv_last = iv[0, -1, :]
    iv_probs = torch.softmax(iv_last, -1)
    kl = float(torch.nn.functional.kl_div(
        torch.log(iv_probs + 1e-10), clean_probs, reduction='sum'
    ).item())
    ld = float(torch.norm(iv_last - clean_last).item())
    return max(0, kl), ld


def steer_feature(
    model: ReplacementModel, prompt: str, feat: Dict,
    multiplier: float, clean_probs: torch.Tensor, clean_last: torch.Tensor
) -> Tuple[float, float, str]:
    """Steer a feature by multiplier and return (KL, logit_diff, new_top_token)."""
    target_val = feat['act'] * multiplier
    iv, _ = model.feature_intervention(
        prompt, [(feat['layer'], feat['pos'], feat['fidx'], target_val)],
        return_activations=False
    )
    iv_last = iv[0, -1, :]
    iv_probs = torch.softmax(iv_last, -1)
    kl = float(torch.nn.functional.kl_div(
        torch.log(iv_probs + 1e-10), clean_probs, reduction='sum'
    ).item())
    ld = float(torch.norm(iv_last - clean_last).item())
    new_top = model.tokenizer.decode([int(iv_probs.argmax().item())])
    return max(0, kl), ld, new_top


def run_ioi_experiment(
    model: ReplacementModel,
    prompts: List[str],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """Run full IOI experiment with AI, greedy, random, and oracle baselines."""
    all_results = []

    for pi, prompt in enumerate(prompts):
        logger.info(f"IOI prompt {pi+1}/{len(prompts)}: '{prompt[:50]}...'")

        raw = attribute(
            prompt=prompt, model=model, max_n_logits=5,
            desired_logit_prob=0.9, batch_size=256, verbose=False
        )
        candidates = extract_candidates(raw, max_per_layer, max_candidates)

        clean_logits, _ = model.feature_intervention(prompt, [], return_activations=False)
        clean_last = clean_logits[0, -1, :]
        clean_probs = torch.softmax(clean_last, -1)

        gt = {}
        for feat in candidates:
            kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
            gt[feat['fid']] = kl
        gt_sorted = sorted(gt.items(), key=lambda x: x[1], reverse=True)

        selector = ActiveInferenceSelector(candidates, exploration_weight=2.0)
        ai_kls = []
        for step in range(min(budget, len(candidates))):
            idx, feat = selector.select_next()
            kl = gt[feat['fid']]
            selector.update(idx, kl)
            ai_kls.append(kl)

        greedy_kls = [gt[candidates[i]['fid']] for i in range(min(budget, len(candidates)))]

        rand_trials = []
        for trial in range(10):
            sh = list(range(len(candidates)))
            np.random.shuffle(sh)
            rand_trials.append([gt[candidates[i]['fid']] for i in sh[:budget]])

        oracle_kls = [k for _, k in gt_sorted[:budget]]

        all_results.append({
            'prompt': prompt,
            'n_candidates': len(candidates),
            'n_features': int(raw.active_features.shape[0]),
            'ground_truth_top5': gt_sorted[:5],
            'ai_kls': ai_kls,
            'greedy_kls': greedy_kls,
            'ai_mean': float(np.mean(ai_kls)),
            'greedy_mean': float(np.mean(greedy_kls)),
            'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
            'ai_cumkl': float(np.cumsum(ai_kls)[-1]),
            'greedy_cumkl': float(np.cumsum(greedy_kls)[-1]),
            'random_cumkl': float(np.mean([np.cumsum(t)[-1] for t in rand_trials])),
            'oracle_cumkl': float(np.cumsum(oracle_kls)[-1]),
        })

    ai_means = [r['ai_mean'] for r in all_results]
    greedy_means = [r['greedy_mean'] for r in all_results]
    rand_means = [r['random_mean'] for r in all_results]

    return {
        'task': 'IOI',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': {
            'ai_mean_kl': float(np.mean(ai_means)),
            'ai_std_kl': float(np.std(ai_means)),
            'greedy_mean_kl': float(np.mean(greedy_means)),
            'greedy_std_kl': float(np.std(greedy_means)),
            'random_mean_kl': float(np.mean(rand_means)),
            'random_std_kl': float(np.std(rand_means)),
            'ai_vs_random_pct': float((np.mean(ai_means) - np.mean(rand_means)) / np.mean(rand_means) * 100),
            'ai_vs_greedy_pct': float((np.mean(ai_means) - np.mean(greedy_means)) / np.mean(greedy_means) * 100),
            'ai_oracle_efficiency': float(np.mean([r['ai_cumkl'] for r in all_results]) /
                                         np.mean([r['oracle_cumkl'] for r in all_results]) * 100),
            'greedy_oracle_efficiency': float(np.mean([r['greedy_cumkl'] for r in all_results]) /
                                             np.mean([r['oracle_cumkl'] for r in all_results]) * 100),
            'random_oracle_efficiency': float(np.mean([r['random_cumkl'] for r in all_results]) /
                                             np.mean([r['oracle_cumkl'] for r in all_results]) * 100),
        }
    }


def run_steering_experiment(
    model: ReplacementModel,
    prompts: List[str],
    multipliers: List[float] = [0.0, 2.0, 5.0, 10.0],
    n_features: int = 10,
) -> Dict[str, Any]:
    """Run feature steering experiment on multiple prompts."""
    all_results = []

    for pi, prompt in enumerate(prompts):
        logger.info(f"Steering prompt {pi+1}/{len(prompts)}: '{prompt[:50]}...'")

        raw = attribute(
            prompt=prompt, model=model, max_n_logits=5,
            desired_logit_prob=0.9, batch_size=256, verbose=False
        )
        candidates = extract_candidates(raw, max_per_layer=2, max_total=n_features)

        clean_logits, _ = model.feature_intervention(prompt, [], return_activations=False)
        clean_last = clean_logits[0, -1, :]
        clean_probs = torch.softmax(clean_last, -1)
        clean_top = model.tokenizer.decode([int(clean_probs.argmax().item())])

        steer_results = []
        for feat in candidates:
            feat_results = {'fid': feat['fid'], 'layer': feat['layer'], 'act': feat['act']}
            for mult in multipliers:
                kl, ld, new_top = steer_feature(
                    model, prompt, feat, mult, clean_probs, clean_last
                )
                feat_results[f'mult_{mult}'] = {
                    'kl': kl, 'logit_diff': ld, 'new_top': new_top,
                    'prediction_changed': new_top != clean_top,
                }
            steer_results.append(feat_results)

        all_results.append({
            'prompt': prompt,
            'clean_prediction': clean_top,
            'n_features_tested': len(candidates),
            'features': steer_results,
        })

    return {
        'task': 'Steering',
        'multipliers': multipliers,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
    }


def run_multistep_experiment(
    model: ReplacementModel,
    prompts: List[str],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """Run multi-step reasoning experiment with AI vs baselines.

    Measures whether the AI selector can efficiently find features that
    mediate multi-hop reasoning (intermediate features bridging input to output).
    """
    all_results = []

    for pi, prompt in enumerate(prompts):
        logger.info(f"Multi-step prompt {pi+1}/{len(prompts)}: '{prompt[:60]}...'")

        raw = attribute(
            prompt=prompt, model=model, max_n_logits=5,
            desired_logit_prob=0.9, batch_size=256, verbose=False
        )
        candidates = extract_candidates(raw, max_per_layer, max_candidates)

        clean_logits, _ = model.feature_intervention(prompt, [], return_activations=False)
        clean_last = clean_logits[0, -1, :]
        clean_probs = torch.softmax(clean_last, -1)
        clean_top_id = int(clean_probs.argmax().item())
        clean_top = model.tokenizer.decode([clean_top_id])

        gt = {}
        for feat in candidates:
            kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
            gt[feat['fid']] = {'kl': kl, 'ld': ld, 'layer': feat['layer']}
        gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)

        selector = ActiveInferenceSelector(candidates, exploration_weight=2.0)
        ai_kls = []
        for step in range(min(budget, len(candidates))):
            idx, feat = selector.select_next()
            kl = gt[feat['fid']]['kl']
            selector.update(idx, kl)
            ai_kls.append(kl)

        greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(min(budget, len(candidates)))]

        rand_trials = []
        for trial in range(10):
            sh = list(range(len(candidates)))
            np.random.shuffle(sh)
            rand_trials.append([gt[candidates[i]['fid']]['kl'] for i in sh[:budget]])

        oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

        layer_distribution = defaultdict(int)
        for fid, v in gt_sorted[:10]:
            layer_distribution[v['layer']] += 1

        n_layers = model.cfg.n_layers
        early = sum(1 for fid, v in gt_sorted[:10] if v['layer'] < n_layers // 3)
        mid = sum(1 for fid, v in gt_sorted[:10] if n_layers // 3 <= v['layer'] < 2 * n_layers // 3)
        late = sum(1 for fid, v in gt_sorted[:10] if v['layer'] >= 2 * n_layers // 3)

        all_results.append({
            'prompt': prompt,
            'clean_prediction': clean_top,
            'n_candidates': len(candidates),
            'top10_features': [(fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]],
            'ai_kls': ai_kls,
            'greedy_kls': greedy_kls,
            'oracle_kls': oracle_kls,
            'ai_mean': float(np.mean(ai_kls)),
            'greedy_mean': float(np.mean(greedy_kls)),
            'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
            'ai_cumkl': float(np.sum(ai_kls)),
            'greedy_cumkl': float(np.sum(greedy_kls)),
            'random_cumkl': float(np.mean([np.sum(t) for t in rand_trials])),
            'oracle_cumkl': float(np.sum(oracle_kls)),
            'layer_distribution': {'early': early, 'mid': mid, 'late': late},
        })

    ai_means = [r['ai_mean'] for r in all_results]
    greedy_means = [r['greedy_mean'] for r in all_results]
    rand_means = [r['random_mean'] for r in all_results]

    return {
        'task': 'MultiStep',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': {
            'ai_mean_kl': float(np.mean(ai_means)),
            'ai_std_kl': float(np.std(ai_means)),
            'greedy_mean_kl': float(np.mean(greedy_means)),
            'greedy_std_kl': float(np.std(greedy_means)),
            'random_mean_kl': float(np.mean(rand_means)),
            'random_std_kl': float(np.std(rand_means)),
            'ai_vs_random_pct': float((np.mean(ai_means) - np.mean(rand_means)) / max(np.mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((np.mean(ai_means) - np.mean(greedy_means)) / max(np.mean(greedy_means), 1e-10) * 100),
            'ai_oracle_efficiency': float(np.mean([r['ai_cumkl'] for r in all_results]) /
                                         max(np.mean([r['oracle_cumkl'] for r in all_results]), 1e-10) * 100),
        }
    }


def run_domain_experiment(
    model: ReplacementModel,
    domain_prompts: Dict[str, List[str]],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """Run multi-domain experiment across 5 cognitive domains (IRP).

    Categorizes results by domain and reports layer distribution per domain.
    """
    by_domain: Dict[str, Dict[str, Any]] = {}
    n_layers = model.cfg.n_layers

    for domain, prompts in domain_prompts.items():
        domain_results = []
        domain_early, domain_mid, domain_late = 0, 0, 0

        for pi, prompt in enumerate(prompts):
            logger.info(f"Domain [{domain}] prompt {pi+1}/{len(prompts)}: '{prompt[:50]}...'")

            raw = attribute(
                prompt=prompt, model=model, max_n_logits=5,
                desired_logit_prob=0.9, batch_size=256, verbose=False
            )
            candidates = extract_candidates(raw, max_per_layer, max_candidates)

            clean_logits, _ = model.feature_intervention(prompt, [], return_activations=False)
            clean_last = clean_logits[0, -1, :]
            clean_probs = torch.softmax(clean_last, -1)
            clean_top = model.tokenizer.decode([int(clean_probs.argmax().item())])

            gt = {}
            for feat in candidates:
                kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
                gt[feat['fid']] = {'kl': kl, 'ld': ld, 'layer': feat['layer']}
            gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)

            selector = ActiveInferenceSelector(candidates, exploration_weight=2.0)
            ai_kls = []
            for step in range(min(budget, len(candidates))):
                idx, feat = selector.select_next()
                kl = gt[feat['fid']]['kl']
                selector.update(idx, kl)
                ai_kls.append(kl)

            greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(min(budget, len(candidates)))]

            rand_trials = []
            for trial in range(10):
                sh = list(range(len(candidates)))
                np.random.shuffle(sh)
                rand_trials.append([gt[candidates[i]['fid']]['kl'] for i in sh[:budget]])

            oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

            early = sum(1 for fid, v in gt_sorted[:10] if v['layer'] < n_layers // 3)
            mid = sum(1 for fid, v in gt_sorted[:10] if n_layers // 3 <= v['layer'] < 2 * n_layers // 3)
            late = sum(1 for fid, v in gt_sorted[:10] if v['layer'] >= 2 * n_layers // 3)
            domain_early += early
            domain_mid += mid
            domain_late += late

            domain_results.append({
                'prompt': prompt,
                'clean_prediction': clean_top,
                'n_candidates': len(candidates),
                'top10_features': [(fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]],
                'ai_kls': ai_kls,
                'greedy_kls': greedy_kls,
                'ai_mean': float(np.mean(ai_kls)),
                'greedy_mean': float(np.mean(greedy_kls)),
                'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
                'ai_cumkl': float(np.sum(ai_kls)),
                'greedy_cumkl': float(np.sum(greedy_kls)),
                'random_cumkl': float(np.mean([np.sum(t) for t in rand_trials])),
                'oracle_cumkl': float(np.sum(oracle_kls)),
                'layer_distribution': {'early': early, 'mid': mid, 'late': late},
            })

        ai_means = [r['ai_mean'] for r in domain_results]
        greedy_means = [r['greedy_mean'] for r in domain_results]
        rand_means = [r['random_mean'] for r in domain_results]

        by_domain[domain] = {
            'per_prompt': domain_results,
            'layer_distribution': {'early': domain_early, 'mid': domain_mid, 'late': domain_late},
            'ai_mean_kl': float(np.mean(ai_means)),
            'greedy_mean_kl': float(np.mean(greedy_means)),
            'random_mean_kl': float(np.mean(rand_means)),
            'ai_vs_random_pct': float((np.mean(ai_means) - np.mean(rand_means)) / max(np.mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((np.mean(ai_means) - np.mean(greedy_means)) / max(np.mean(greedy_means), 1e-10) * 100),
        }

    all_ai = [d['ai_mean_kl'] for d in by_domain.values()]
    all_greedy = [d['greedy_mean_kl'] for d in by_domain.values()]
    all_rand = [d['random_mean_kl'] for d in by_domain.values()]

    return {
        'task': 'Domain',
        'budget': budget,
        'by_domain': by_domain,
        'aggregate': {
            'ai_mean_kl': float(np.mean(all_ai)),
            'greedy_mean_kl': float(np.mean(all_greedy)),
            'random_mean_kl': float(np.mean(all_rand)),
            'ai_vs_random_pct': float((np.mean(all_ai) - np.mean(all_rand)) / max(np.mean(all_rand), 1e-10) * 100),
            'ai_vs_greedy_pct': float((np.mean(all_ai) - np.mean(all_greedy)) / max(np.mean(all_greedy), 1e-10) * 100),
        }
    }


def _save_graph_for_prompt(
    model: ReplacementModel,
    prompt: str,
    slug: str,
    output_path: Path,
) -> None:
    """Run attribution for a prompt and save raw graph files. No-op if create_graph_files unavailable."""
    if create_graph_files is None:
        logger.warning("create_graph_files not available; skipping graph export")
        return
    try:
        raw_graph = attribute(
            prompt=prompt, model=model, max_n_logits=5,
            desired_logit_prob=0.9, batch_size=256, verbose=False
        )
        create_graph_files(raw_graph, slug, str(output_path))
        logger.info(f"Graph saved: {output_path / slug}")
    except Exception as e:
        logger.warning(f"Graph export failed for {slug}: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Active Circuit Discovery experiments on Gemma and/or Llama."
    )
    parser.add_argument(
        "--model",
        choices=["gemma", "llama", "both"],
        default="both",
        help="Model to run (default: both)",
    )
    parser.add_argument(
        "--experiment",
        choices=["ioi", "steering", "multistep", "domain", "all"],
        default="all",
        help="Experiment type to run (default: all)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    models_to_run = ["gemma", "llama"] if args.model == "both" else [args.model]
    experiments_to_run = (
        ["ioi", "steering", "multistep", "domain"]
        if args.experiment == "all"
        else [args.experiment]
    )

    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    graphs_dir = results_dir / "graphs"
    results_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_key in models_to_run:
        cfg = MODEL_CONFIG[model_key]
        logger.info(f"Loading {cfg['model_name']} with transcoders ({cfg['transcoder_set']})...")
        model = ReplacementModel.from_pretrained(
            model_name=cfg["model_name"],
            transcoder_set=cfg["transcoder_set"],
            backend="transformerlens",
            device=device,
            dtype=torch.float32,
        )
        logger.info("Model loaded.")

        if "ioi" in experiments_to_run:
            logger.info(f"Running IOI experiment [{model_key}]...")
            _save_graph_for_prompt(
                model, IOI_PROMPTS[0], f"ioi_{model_key}", graphs_dir
            )
            t0 = time.time()
            ioi_results = run_ioi_experiment(model, IOI_PROMPTS, budget=20)
            ioi_results["elapsed_seconds"] = time.time() - t0
            ioi_results["model"] = cfg["model_name"]
            out_path = results_dir / f"ioi_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(ioi_results, f, indent=2)
            logger.info(f"IOI results saved to {out_path}. AI vs Random: {ioi_results['aggregate']['ai_vs_random_pct']:+.1f}%")

        if "steering" in experiments_to_run:
            logger.info(f"Running steering experiment [{model_key}]...")
            _save_graph_for_prompt(
                model, STEERING_PROMPTS[0], f"steering_{model_key}", graphs_dir
            )
            t0 = time.time()
            steer_results = run_steering_experiment(model, STEERING_PROMPTS)
            steer_results["elapsed_seconds"] = time.time() - t0
            steer_results["model"] = cfg["model_name"]
            out_path = results_dir / f"steering_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(steer_results, f, indent=2)
            logger.info(f"Steering results saved to {out_path}.")

        if "multistep" in experiments_to_run:
            logger.info(f"Running multi-step experiment [{model_key}]...")
            _save_graph_for_prompt(
                model, MULTISTEP_PROMPTS[0], f"multistep_{model_key}", graphs_dir
            )
            t0 = time.time()
            ms_results = run_multistep_experiment(model, MULTISTEP_PROMPTS, budget=20)
            ms_results["elapsed_seconds"] = time.time() - t0
            ms_results["model"] = cfg["model_name"]
            out_path = results_dir / f"multistep_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(ms_results, f, indent=2)
            logger.info(f"Multi-step results saved to {out_path}.")

        if "domain" in experiments_to_run:
            first_domain_prompt = DOMAIN_PROMPTS["geography"][0]
            logger.info(f"Running domain experiment [{model_key}]...")
            _save_graph_for_prompt(
                model, first_domain_prompt, f"domain_{model_key}", graphs_dir
            )
            t0 = time.time()
            domain_results = run_domain_experiment(model, DOMAIN_PROMPTS, budget=20)
            domain_results["elapsed_seconds"] = time.time() - t0
            domain_results["model"] = cfg["model_name"]
            out_path = results_dir / f"domain_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(domain_results, f, indent=2)
            logger.info(f"Domain results saved to {out_path}.")

    # Print summary for last model run (or first if multiple)
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for model_key in models_to_run:
        cfg = MODEL_CONFIG[model_key]
        print(f"\n--- {cfg['model_name']} ---")
        if "ioi" in experiments_to_run:
            p = results_dir / f"ioi_results_{model_key}.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                agg = r["aggregate"]
                print(f"\nIOI (5 prompts, budget=20):")
                print(f"  AI mean KL:      {agg['ai_mean_kl']:.6f} +/- {agg['ai_std_kl']:.6f}")
                print(f"  Greedy mean KL:  {agg['greedy_mean_kl']:.6f} +/- {agg['greedy_std_kl']:.6f}")
                print(f"  Random mean KL:  {agg['random_mean_kl']:.6f} +/- {agg['random_std_kl']:.6f}")
                print(f"  AI vs Random:    {agg['ai_vs_random_pct']:+.1f}%")
                print(f"  AI vs Greedy:    {agg['ai_vs_greedy_pct']:+.1f}%")
                print(f"  Oracle efficiency: AI={agg['ai_oracle_efficiency']:.1f}% "
                      f"Greedy={agg['greedy_oracle_efficiency']:.1f}% "
                      f"Random={agg['random_oracle_efficiency']:.1f}%")
        if "steering" in experiments_to_run:
            p = results_dir / f"steering_results_{model_key}.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                n_changed = n_total = 0
                for pr in r["per_prompt"]:
                    for feat in pr["features"]:
                        for mult in [5.0, 10.0]:
                            key = f"mult_{mult}"
                            if key in feat:
                                n_total += 1
                                if feat[key]["prediction_changed"]:
                                    n_changed += 1
                print(f"\nSteering ({len(STEERING_PROMPTS)} prompts):")
                print(f"  Prediction changes: {n_changed}/{n_total} ({n_changed/max(n_total,1)*100:.0f}%)")
        if "multistep" in experiments_to_run:
            p = results_dir / f"multistep_results_{model_key}.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                ms_agg = r["aggregate"]
                print(f"\nMulti-step Reasoning ({len(MULTISTEP_PROMPTS)} prompts, budget=20):")
                print(f"  AI mean KL:      {ms_agg['ai_mean_kl']:.6f} +/- {ms_agg['ai_std_kl']:.6f}")
                print(f"  Greedy mean KL:  {ms_agg['greedy_mean_kl']:.6f} +/- {ms_agg['greedy_std_kl']:.6f}")
                print(f"  Random mean KL:  {ms_agg['random_mean_kl']:.6f} +/- {ms_agg['random_std_kl']:.6f}")
                print(f"  AI vs Random:    {ms_agg['ai_vs_random_pct']:+.1f}%")
                print(f"  AI vs Greedy:    {ms_agg['ai_vs_greedy_pct']:+.1f}%")
                print(f"  Oracle efficiency: AI={ms_agg['ai_oracle_efficiency']:.1f}%")
                for pr in r["per_prompt"]:
                    ld = pr["layer_distribution"]
                    print(f"    '{pr['prompt'][:50]}...' -> '{pr['clean_prediction']}' "
                          f"[early={ld['early']}, mid={ld['mid']}, late={ld['late']}]")
        if "domain" in experiments_to_run:
            p = results_dir / f"domain_results_{model_key}.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                agg = r["aggregate"]
                print(f"\nDomain (5 domains, 2 prompts each):")
                print(f"  AI mean KL:      {agg['ai_mean_kl']:.6f}")
                print(f"  Greedy mean KL:  {agg['greedy_mean_kl']:.6f}")
                print(f"  Random mean KL:  {agg['random_mean_kl']:.6f}")
                print(f"  AI vs Random:    {agg['ai_vs_random_pct']:+.1f}%")
                print(f"  AI vs Greedy:    {agg['ai_vs_greedy_pct']:+.1f}%")
                for domain, d in r["by_domain"].items():
                    ld = d["layer_distribution"]
                    print(f"    {domain}: [early={ld['early']}, mid={ld['mid']}, late={ld['late']}]")


if __name__ == "__main__":
    main()
