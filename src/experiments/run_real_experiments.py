"""
Real Experiment Runner for Active Circuit Discovery
=====================================================
Runs experiments on Gemma-2-2B and Llama-3.2-1B using circuit-tracer
attribution graphs and active inference (pymdp POMDP) for feature
selection.

Selectors compared:
  - ActiveInferencePOMDPAgent  (pymdp, EFE-based)
  - BanditSelector             (UCB-style heuristic baseline)
  - Greedy                     (descending graph importance)
  - Random                     (uniform shuffle, 10 trials)
  - Oracle                     (true descending KL)
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
from src.active_inference.pomdp_agent import ActiveInferencePOMDPAgent

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
        "local_path": os.path.expanduser("~/.cache/huggingface/llama-3.2-1b-local"),
    },
}


# ======================================================================
# Baseline selector (UCB-style bandit -- kept for comparison)
# ======================================================================

class BanditSelector:
    """UCB-style feature selector (heuristic baseline)."""

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


# ======================================================================
# Candidate extraction
# ======================================================================

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
        pos   = int(raw_graph.active_features[ft, 1].item())
        fidx  = int(raw_graph.active_features[ft, 2].item())
        act   = float(raw_graph.activation_values[i].item())
        imp   = float(infl[i].item()) / mi

        in_deg  = int((adj[:, i].abs() > 0).sum().item())
        out_deg = int((adj[i, :].abs() > 0).sum().item())

        entry = dict(
            layer=layer, pos=pos, fidx=fidx, act=act, imp=imp,
            fid=f'L{layer}_P{pos}_F{fidx}',
            in_degree=in_deg, out_degree=out_deg,
        )
        by_layer.setdefault(layer, []).append(entry)

    for l in by_layer:
        by_layer[l].sort(key=lambda x: x['imp'], reverse=True)

    candidates = []
    for l in sorted(by_layer.keys()):
        candidates.extend(by_layer[l][:max_per_layer])
    candidates.sort(key=lambda x: x['imp'], reverse=True)
    return candidates[:max_total]


# ======================================================================
# Intervention helpers
# ======================================================================

def ablate_feature(
    model: ReplacementModel, prompt: str, feat: Dict,
    clean_probs: torch.Tensor, clean_last: torch.Tensor
) -> Tuple[float, float]:
    """Ablate a single transcoder feature and return (KL, logit_diff)."""
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


# ======================================================================
# IOI experiment
# ======================================================================

def run_ioi_experiment(
    model: ReplacementModel,
    prompts: List[str],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """IOI experiment comparing pymdp agent, bandit, greedy, random, oracle."""
    n_layers = model.cfg.n_layers
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

        # Ground-truth KL for every candidate
        gt: Dict[str, Dict[str, Any]] = {}
        for feat in candidates:
            kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
            gt[feat['fid']] = {'kl': kl, 'ld': ld, 'layer': feat['layer']}
        gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)

        # --- pymdp agent ---
        agent = ActiveInferencePOMDPAgent(n_layers=n_layers)
        agent.initialize()
        observed_fids: set = set()
        ai_kls = []
        for step in range(min(budget, len(candidates))):
            unobserved = [c for c in candidates if c['fid'] not in observed_fids]
            if not unobserved:
                break
            feat, action, efe = agent.select_intervention(unobserved)
            kl = gt[feat['fid']]['kl']
            agent.update_beliefs(
                feat,
                kl_divergence=kl,
                activation_value=feat['act'],
                graph_connectivity=feat.get('in_degree', 0) + feat.get('out_degree', 0),
            )
            observed_fids.add(feat['fid'])
            ai_kls.append(kl)

        # --- bandit selector ---
        bandit = BanditSelector(candidates, exploration_weight=2.0)
        bandit_kls = []
        for step in range(min(budget, len(candidates))):
            idx, feat = bandit.select_next()
            kl = gt[feat['fid']]['kl']
            bandit.update(idx, kl)
            bandit_kls.append(kl)

        # --- greedy (descending graph importance) ---
        greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(min(budget, len(candidates)))]

        # --- random (10 trials) ---
        rand_trials = []
        for _ in range(10):
            sh = list(range(len(candidates)))
            np.random.shuffle(sh)
            rand_trials.append([gt[candidates[i]['fid']]['kl'] for i in sh[:budget]])

        # --- oracle ---
        oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

        all_results.append({
            'prompt': prompt,
            'n_candidates': len(candidates),
            'n_features': int(raw.active_features.shape[0]),
            'ground_truth_top5': [(fid, v['kl']) for fid, v in gt_sorted[:5]],
            'ai_kls': ai_kls,
            'bandit_kls': bandit_kls,
            'greedy_kls': greedy_kls,
            'oracle_kls': oracle_kls,
            'ai_mean': float(np.mean(ai_kls)) if ai_kls else 0,
            'bandit_mean': float(np.mean(bandit_kls)) if bandit_kls else 0,
            'greedy_mean': float(np.mean(greedy_kls)),
            'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
            'ai_cumkl': float(np.sum(ai_kls)),
            'bandit_cumkl': float(np.sum(bandit_kls)),
            'greedy_cumkl': float(np.sum(greedy_kls)),
            'random_cumkl': float(np.mean([np.sum(t) for t in rand_trials])),
            'oracle_cumkl': float(np.sum(oracle_kls)),
            'agent_entropy_history': agent.get_belief_entropy_history(),
            'agent_efe_history': agent.get_efe_history(),
            'agent_converged': agent.is_converged,
        })

    # Aggregate
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    def safe_std(lst):
        return float(np.std(lst)) if lst else 0.0

    ai_means     = [r['ai_mean'] for r in all_results]
    bandit_means = [r['bandit_mean'] for r in all_results]
    greedy_means = [r['greedy_mean'] for r in all_results]
    rand_means   = [r['random_mean'] for r in all_results]

    ai_cum     = safe_mean([r['ai_cumkl'] for r in all_results])
    bandit_cum = safe_mean([r['bandit_cumkl'] for r in all_results])
    greedy_cum = safe_mean([r['greedy_cumkl'] for r in all_results])
    rand_cum   = safe_mean([r['random_cumkl'] for r in all_results])
    oracle_cum = safe_mean([r['oracle_cumkl'] for r in all_results])

    return {
        'task': 'IOI',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': {
            'ai_mean_kl': safe_mean(ai_means),
            'ai_std_kl': safe_std(ai_means),
            'bandit_mean_kl': safe_mean(bandit_means),
            'bandit_std_kl': safe_std(bandit_means),
            'greedy_mean_kl': safe_mean(greedy_means),
            'greedy_std_kl': safe_std(greedy_means),
            'random_mean_kl': safe_mean(rand_means),
            'random_std_kl': safe_std(rand_means),
            # Percentage improvements: (AI_mean - baseline_mean) / baseline_mean * 100
            # Uses mean-of-per-prompt-means; paper tables cite these values directly.
            'ai_vs_random_pct': float((safe_mean(ai_means) - safe_mean(rand_means)) / max(safe_mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((safe_mean(ai_means) - safe_mean(greedy_means)) / max(safe_mean(greedy_means), 1e-10) * 100),
            'ai_vs_bandit_pct': float((safe_mean(ai_means) - safe_mean(bandit_means)) / max(safe_mean(bandit_means), 1e-10) * 100),
            'ai_oracle_efficiency': float(ai_cum / max(oracle_cum, 1e-10) * 100),
            'bandit_oracle_efficiency': float(bandit_cum / max(oracle_cum, 1e-10) * 100),
            'greedy_oracle_efficiency': float(greedy_cum / max(oracle_cum, 1e-10) * 100),
            'random_oracle_efficiency': float(rand_cum / max(oracle_cum, 1e-10) * 100),
        }
    }


# ======================================================================
# Steering experiment
# ======================================================================

def run_steering_experiment(
    model: ReplacementModel,
    prompts: List[str],
    multipliers: List[float] = [0.0, 2.0, 5.0, 10.0],
    n_features: int = 10,
) -> Dict[str, Any]:
    """Feature steering experiment."""
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
        per_mult_kls: Dict[float, List[float]] = {m: [] for m in multipliers}

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
                per_mult_kls[mult].append(kl)
            steer_results.append(feat_results)

        mean_kl_per_multiplier = [
            float(np.mean(per_mult_kls[m])) if per_mult_kls[m] else 0.0
            for m in multipliers
        ]

        all_results.append({
            'prompt': prompt,
            'clean_prediction': clean_top,
            'n_features_tested': len(candidates),
            'features': steer_results,
            'mean_kl_per_multiplier': mean_kl_per_multiplier,
        })

    return {
        'task': 'Steering',
        'multipliers': multipliers,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
    }


# ======================================================================
# Multi-step reasoning experiment
# ======================================================================

def run_multistep_experiment(
    model: ReplacementModel,
    prompts: List[str],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """Multi-step reasoning experiment with pymdp agent."""
    n_layers = model.cfg.n_layers
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
        clean_top = model.tokenizer.decode([int(clean_probs.argmax().item())])

        gt: Dict[str, Dict[str, Any]] = {}
        for feat in candidates:
            kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
            gt[feat['fid']] = {'kl': kl, 'ld': ld, 'layer': feat['layer']}
        gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)

        # pymdp agent
        agent = ActiveInferencePOMDPAgent(n_layers=n_layers)
        agent.initialize()
        observed_fids: set = set()
        ai_kls = []
        for step in range(min(budget, len(candidates))):
            unobserved = [c for c in candidates if c['fid'] not in observed_fids]
            if not unobserved:
                break
            feat, action, efe = agent.select_intervention(unobserved)
            kl = gt[feat['fid']]['kl']
            agent.update_beliefs(
                feat, kl_divergence=kl,
                activation_value=feat['act'],
                graph_connectivity=feat.get('in_degree', 0) + feat.get('out_degree', 0),
            )
            observed_fids.add(feat['fid'])
            ai_kls.append(kl)

        # bandit
        bandit = BanditSelector(candidates, exploration_weight=2.0)
        bandit_kls = []
        for step in range(min(budget, len(candidates))):
            idx, feat = bandit.select_next()
            kl = gt[feat['fid']]['kl']
            bandit.update(idx, kl)
            bandit_kls.append(kl)

        greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(min(budget, len(candidates)))]

        rand_trials = []
        for _ in range(10):
            sh = list(range(len(candidates)))
            np.random.shuffle(sh)
            rand_trials.append([gt[candidates[i]['fid']]['kl'] for i in sh[:budget]])

        oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

        early = sum(1 for _, v in gt_sorted[:10] if v['layer'] < n_layers // 3)
        mid   = sum(1 for _, v in gt_sorted[:10] if n_layers // 3 <= v['layer'] < 2 * n_layers // 3)
        late  = sum(1 for _, v in gt_sorted[:10] if v['layer'] >= 2 * n_layers // 3)

        all_results.append({
            'prompt': prompt,
            'clean_prediction': clean_top,
            'n_candidates': len(candidates),
            'top10_features': [(fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]],
            'ai_kls': ai_kls,
            'bandit_kls': bandit_kls,
            'greedy_kls': greedy_kls,
            'oracle_kls': oracle_kls,
            'ai_mean': float(np.mean(ai_kls)) if ai_kls else 0,
            'bandit_mean': float(np.mean(bandit_kls)) if bandit_kls else 0,
            'greedy_mean': float(np.mean(greedy_kls)),
            'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
            'ai_cumkl': float(np.sum(ai_kls)),
            'bandit_cumkl': float(np.sum(bandit_kls)),
            'greedy_cumkl': float(np.sum(greedy_kls)),
            'random_cumkl': float(np.mean([np.sum(t) for t in rand_trials])),
            'oracle_cumkl': float(np.sum(oracle_kls)),
            'layer_distribution': {'early': early, 'mid': mid, 'late': late},
            'agent_entropy_history': agent.get_belief_entropy_history(),
            'agent_efe_history': agent.get_efe_history(),
        })

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):
        return float(np.std(lst)) if lst else 0.0

    ai_means     = [r['ai_mean'] for r in all_results]
    bandit_means = [r['bandit_mean'] for r in all_results]
    greedy_means = [r['greedy_mean'] for r in all_results]
    rand_means   = [r['random_mean'] for r in all_results]

    ai_cum     = safe_mean([r['ai_cumkl'] for r in all_results])
    bandit_cum = safe_mean([r['bandit_cumkl'] for r in all_results])
    oracle_cum = safe_mean([r['oracle_cumkl'] for r in all_results])

    return {
        'task': 'MultiStep',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': {
            'ai_mean_kl': safe_mean(ai_means),
            'ai_std_kl': safe_std(ai_means),
            'bandit_mean_kl': safe_mean(bandit_means),
            'bandit_std_kl': safe_std(bandit_means),
            'greedy_mean_kl': safe_mean(greedy_means),
            'greedy_std_kl': safe_std(greedy_means),
            'random_mean_kl': safe_mean(rand_means),
            'random_std_kl': safe_std(rand_means),
            'ai_vs_random_pct': float((safe_mean(ai_means) - safe_mean(rand_means)) / max(safe_mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((safe_mean(ai_means) - safe_mean(greedy_means)) / max(safe_mean(greedy_means), 1e-10) * 100),
            'ai_vs_bandit_pct': float((safe_mean(ai_means) - safe_mean(bandit_means)) / max(safe_mean(bandit_means), 1e-10) * 100),
            'ai_oracle_efficiency': float(ai_cum / max(oracle_cum, 1e-10) * 100),
            'bandit_oracle_efficiency': float(bandit_cum / max(oracle_cum, 1e-10) * 100),
        }
    }


# ======================================================================
# Multi-domain experiment
# ======================================================================

def run_domain_experiment(
    model: ReplacementModel,
    domain_prompts: Dict[str, List[str]],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
) -> Dict[str, Any]:
    """Multi-domain experiment across 5 cognitive domains."""
    n_layers = model.cfg.n_layers
    by_domain: Dict[str, Dict[str, Any]] = {}

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

            gt: Dict[str, Dict[str, Any]] = {}
            for feat in candidates:
                kl, ld = ablate_feature(model, prompt, feat, clean_probs, clean_last)
                gt[feat['fid']] = {'kl': kl, 'ld': ld, 'layer': feat['layer']}
            gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)

            # pymdp agent
            agent = ActiveInferencePOMDPAgent(n_layers=n_layers)
            agent.initialize()
            observed_fids: set = set()
            ai_kls = []
            for step in range(min(budget, len(candidates))):
                unobserved = [c for c in candidates if c['fid'] not in observed_fids]
                if not unobserved:
                    break
                feat_sel, action, efe = agent.select_intervention(unobserved)
                kl = gt[feat_sel['fid']]['kl']
                agent.update_beliefs(
                    feat_sel, kl_divergence=kl,
                    activation_value=feat_sel['act'],
                    graph_connectivity=feat_sel.get('in_degree', 0) + feat_sel.get('out_degree', 0),
                )
                observed_fids.add(feat_sel['fid'])
                ai_kls.append(kl)

            greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(min(budget, len(candidates)))]

            rand_trials = []
            for _ in range(10):
                sh = list(range(len(candidates)))
                np.random.shuffle(sh)
                rand_trials.append([gt[candidates[i]['fid']]['kl'] for i in sh[:budget]])

            oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

            early = sum(1 for _, v in gt_sorted[:10] if v['layer'] < n_layers // 3)
            mid   = sum(1 for _, v in gt_sorted[:10] if n_layers // 3 <= v['layer'] < 2 * n_layers // 3)
            late  = sum(1 for _, v in gt_sorted[:10] if v['layer'] >= 2 * n_layers // 3)
            domain_early += early
            domain_mid   += mid
            domain_late  += late

            domain_results.append({
                'prompt': prompt,
                'clean_prediction': clean_top,
                'n_candidates': len(candidates),
                'top10_features': [(fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]],
                'ai_kls': ai_kls,
                'greedy_kls': greedy_kls,
                'ai_mean': float(np.mean(ai_kls)) if ai_kls else 0,
                'greedy_mean': float(np.mean(greedy_kls)),
                'random_mean': float(np.mean([np.mean(t) for t in rand_trials])),
                'ai_cumkl': float(np.sum(ai_kls)),
                'greedy_cumkl': float(np.sum(greedy_kls)),
                'random_cumkl': float(np.mean([np.sum(t) for t in rand_trials])),
                'oracle_cumkl': float(np.sum(oracle_kls)),
                'layer_distribution': {'early': early, 'mid': mid, 'late': late},
            })

        ai_means     = [r['ai_mean'] for r in domain_results]
        greedy_means = [r['greedy_mean'] for r in domain_results]
        rand_means   = [r['random_mean'] for r in domain_results]

        by_domain[domain] = {
            'per_prompt': domain_results,
            'layer_distribution': {'early': domain_early, 'mid': domain_mid, 'late': domain_late},
            'ai_mean_kl': float(np.mean(ai_means)),
            'greedy_mean_kl': float(np.mean(greedy_means)),
            'random_mean_kl': float(np.mean(rand_means)),
            'ai_vs_random_pct': float((np.mean(ai_means) - np.mean(rand_means)) / max(np.mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((np.mean(ai_means) - np.mean(greedy_means)) / max(np.mean(greedy_means), 1e-10) * 100),
        }

    all_ai     = [d['ai_mean_kl'] for d in by_domain.values()]
    all_greedy = [d['greedy_mean_kl'] for d in by_domain.values()]
    all_rand   = [d['random_mean_kl'] for d in by_domain.values()]

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


# ======================================================================
# Graph export
# ======================================================================

def _save_graph_for_prompt(
    model: ReplacementModel, prompt: str, slug: str, output_path: Path,
) -> None:
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


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Active Circuit Discovery experiments."
    )
    parser.add_argument(
        "--model", choices=["gemma", "llama", "both"], default="both",
        help="Model to run (default: both)",
    )
    parser.add_argument(
        "--experiment", choices=["ioi", "steering", "multistep", "domain", "all"],
        default="all", help="Experiment type (default: all)",
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

        extra_kwargs: Dict[str, Any] = {}
        local_path = cfg.get("local_path")
        if local_path and os.path.isdir(local_path):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading model weights from local path: {local_path}")
            extra_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(local_path)
            extra_kwargs["hf_model"] = AutoModelForCausalLM.from_pretrained(
                local_path, torch_dtype=torch.float32
            )

        model = ReplacementModel.from_pretrained(
            model_name=cfg["model_name"],
            transcoder_set=cfg["transcoder_set"],
            backend="transformerlens",
            device=device,
            dtype=torch.float32,
            **extra_kwargs,
        )
        logger.info("Model loaded.")

        if "ioi" in experiments_to_run:
            logger.info(f"Running IOI experiment [{model_key}]...")
            _save_graph_for_prompt(model, IOI_PROMPTS[0], f"ioi_{model_key}", graphs_dir)
            t0 = time.time()
            ioi_results = run_ioi_experiment(model, IOI_PROMPTS, budget=20)
            ioi_results["elapsed_seconds"] = time.time() - t0
            ioi_results["model"] = cfg["model_name"]
            out_path = results_dir / f"ioi_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(ioi_results, f, indent=2)
            logger.info(f"IOI results -> {out_path}")

        if "steering" in experiments_to_run:
            logger.info(f"Running steering experiment [{model_key}]...")
            _save_graph_for_prompt(model, STEERING_PROMPTS[0], f"steering_{model_key}", graphs_dir)
            t0 = time.time()
            steer_results = run_steering_experiment(model, STEERING_PROMPTS)
            steer_results["elapsed_seconds"] = time.time() - t0
            steer_results["model"] = cfg["model_name"]
            out_path = results_dir / f"steering_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(steer_results, f, indent=2)
            logger.info(f"Steering results -> {out_path}")

        if "multistep" in experiments_to_run:
            logger.info(f"Running multi-step experiment [{model_key}]...")
            _save_graph_for_prompt(model, MULTISTEP_PROMPTS[0], f"multistep_{model_key}", graphs_dir)
            t0 = time.time()
            ms_results = run_multistep_experiment(model, MULTISTEP_PROMPTS, budget=20)
            ms_results["elapsed_seconds"] = time.time() - t0
            ms_results["model"] = cfg["model_name"]
            out_path = results_dir / f"multistep_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(ms_results, f, indent=2)
            logger.info(f"Multi-step results -> {out_path}")

        if "domain" in experiments_to_run:
            logger.info(f"Running domain experiment [{model_key}]...")
            _save_graph_for_prompt(model, DOMAIN_PROMPTS["geography"][0], f"domain_{model_key}", graphs_dir)
            t0 = time.time()
            domain_results = run_domain_experiment(model, DOMAIN_PROMPTS, budget=20)
            domain_results["elapsed_seconds"] = time.time() - t0
            domain_results["model"] = cfg["model_name"]
            out_path = results_dir / f"domain_results_{model_key}.json"
            with open(out_path, "w") as f:
                json.dump(domain_results, f, indent=2)
            logger.info(f"Domain results -> {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for model_key in models_to_run:
        cfg = MODEL_CONFIG[model_key]
        print(f"\n--- {cfg['model_name']} ---")
        for exp_name in experiments_to_run:
            p = results_dir / f"{exp_name}_results_{model_key}.json"
            if not p.exists():
                continue
            with open(p) as f:
                r = json.load(f)
            agg = r.get("aggregate", {})
            if exp_name == "ioi":
                print(f"\n  IOI ({r['n_prompts']} prompts, budget={r['budget']}):")
                print(f"    pymdp agent mean KL: {agg.get('ai_mean_kl', 0):.6f}")
                print(f"    Bandit mean KL:      {agg.get('bandit_mean_kl', 0):.6f}")
                print(f"    Greedy mean KL:      {agg.get('greedy_mean_kl', 0):.6f}")
                print(f"    Random mean KL:      {agg.get('random_mean_kl', 0):.6f}")
                print(f"    Oracle eff: pymdp={agg.get('ai_oracle_efficiency', 0):.1f}%"
                      f"  bandit={agg.get('bandit_oracle_efficiency', 0):.1f}%"
                      f"  greedy={agg.get('greedy_oracle_efficiency', 0):.1f}%"
                      f"  random={agg.get('random_oracle_efficiency', 0):.1f}%")
            elif exp_name == "multistep":
                print(f"\n  Multi-step ({r['n_prompts']} prompts, budget={r['budget']}):")
                print(f"    pymdp agent mean KL: {agg.get('ai_mean_kl', 0):.6f}")
                print(f"    Oracle eff: pymdp={agg.get('ai_oracle_efficiency', 0):.1f}%"
                      f"  bandit={agg.get('bandit_oracle_efficiency', 0):.1f}%")
            elif exp_name == "steering":
                n_changed = n_total = 0
                for pr in r["per_prompt"]:
                    for feat in pr["features"]:
                        for mult in [5.0, 10.0]:
                            key = f"mult_{mult}"
                            if key in feat:
                                n_total += 1
                                if feat[key]["prediction_changed"]:
                                    n_changed += 1
                print(f"\n  Steering ({r['n_prompts']} prompts):")
                print(f"    Prediction changes: {n_changed}/{n_total}")
            elif exp_name == "domain":
                print(f"\n  Domain (5 domains):")
                for domain, d in r.get("by_domain", {}).items():
                    ld = d.get("layer_distribution", {})
                    print(f"    {domain}: [early={ld.get('early',0)}, mid={ld.get('mid',0)}, late={ld.get('late',0)}]")


if __name__ == "__main__":
    main()
