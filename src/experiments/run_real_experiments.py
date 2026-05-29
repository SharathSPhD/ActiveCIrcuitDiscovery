"""
Real Experiment Runner for Active Circuit Discovery
=====================================================
Runs experiments on Gemma-2-2B and Llama-3.2-1B using circuit-tracer
attribution graphs and active inference (pymdp POMDP) for feature
selection.

Selectors compared (feature selection x intervention type):
  - ActiveInferencePOMDPAgent   (pymdp, EFE-based, multi-action)
  - ActiveInferencePOMDPAgent   (pymdp, EFE-based, ablation-only: POMDP-abl)
  - BanditSelector              (UCB-style heuristic, ablation-only)
  - Greedy                      (descending graph connectivity, ablation-only)
  - EAP                         (descending direct logit attribution, ablation-only)
  - Random                      (uniform shuffle, 10 trials, ablation-only)
  - Oracle                      (true descending ablation KL -- upper bound)

Action-matched baselines (same multi-action space as the POMDP agent), used
to decompose whether the multi-action POMDP advantage comes from EFE-guided
*selection* or simply from *steering producing higher KL*:
  - Greedy + steering           (greedy selection, steering intervention)
  - Bandit + steering           (bandit selection, steering intervention)
  - Random + steering           (random selection, steering intervention)
  - Random-action               (random selection, random intervention type)
  - Oracle (multi-action)       (best features by max(ablation, steering) KL)

All KL values come from real `feature_intervention` calls; nothing is
synthesised.  Per-prompt arrays are saved so that downstream statistics
(bootstrap CIs, medians, RCK) are computed transparently from raw data.
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
from src.active_inference import pomdp_agent as pomdp_mod
from src.active_inference.pomdp_agent import ActiveInferencePOMDPAgent, ACTION_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# Multiplier used by the multi-action steering intervention.  Kept as a module
# constant so the agent's executed steering KL matches the precomputed
# steering ground truth used for the action-matched baselines.
STEER_MULT = 5.0

# Number of random-restart trials for the stochastic baselines.
N_RANDOM_TRIALS = 10
RANDOM_SEED = 42


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


class BanditUCBSelector:
    """Plain UCB1 bandit over layers (no hand-tuned inductive bias).

    Unlike :class:`BanditSelector`, which folds in graph-importance priors and
    positional/layer locality heuristics, this baseline treats each layer as an
    arm and applies the textbook UCB1 rule using the observed KL as reward.
    Within the chosen arm it picks the highest-importance unobserved candidate
    purely as a tie-break.  It isolates how much of the heuristic bandit's
    performance comes from its engineered priors rather than from
    upper-confidence exploration alone.
    """

    def __init__(self, candidates: List[Dict], exploration_weight: float = 2.0):
        self.candidates = candidates
        self.exploration_weight = exploration_weight
        self.observed: set = set()
        self.selection_order: List[int] = []
        self.layers = sorted({c['layer'] for c in candidates})
        self.layer_reward_sum: Dict[int, float] = defaultdict(float)
        self.layer_pulls: Dict[int, int] = defaultdict(int)
        self._by_layer: Dict[int, List[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            self._by_layer[c['layer']].append(i)
        for l in self._by_layer:
            self._by_layer[l].sort(key=lambda i: candidates[i]['imp'], reverse=True)
        self._t = 0

    def _layer_ucb(self, layer: int) -> float:
        n = self.layer_pulls[layer]
        if n == 0:
            return float('inf')  # optimistic initialisation: try every arm once
        mean = self.layer_reward_sum[layer] / n
        bonus = self.exploration_weight * np.sqrt(np.log(self._t + 1) / n)
        return mean + bonus

    def _next_in_layer(self, layer: int) -> Optional[int]:
        for i in self._by_layer[layer]:
            if i not in self.observed:
                return i
        return None

    def select_next(self) -> Tuple[int, Dict]:
        self._t += 1
        order = sorted(self.layers, key=self._layer_ucb, reverse=True)
        for layer in order:
            idx = self._next_in_layer(layer)
            if idx is not None:
                self.selection_order.append(idx)
                return idx, self.candidates[idx]
        # all candidates observed; fall back to first unobserved overall
        for i in range(len(self.candidates)):
            if i not in self.observed:
                self.selection_order.append(i)
                return i, self.candidates[i]
        return 0, self.candidates[0]

    def update(self, idx: int, kl_value: float) -> None:
        self.observed.add(idx)
        layer = self.candidates[idx]['layer']
        self.layer_reward_sum[layer] += kl_value
        self.layer_pulls[layer] += 1


# ======================================================================
# Candidate extraction
# ======================================================================

def extract_candidates(
    raw_graph, max_per_layer: int = 5, max_total: int = 60
) -> List[Dict[str, Any]]:
    """Extract diverse candidate features across all layers.

    Each candidate carries two attribution scores:
      - ``imp``: normalised bidirectional graph connectivity (sum of absolute
        in/out adjacency weights), used by the greedy baseline.
      - ``eap``: normalised *direct* attribution to the logit nodes (the
        feature's column summed over the logit rows of the adjacency matrix),
        used by the EAP baseline.  This isolates the classic Edge Attribution
        Patching "direct effect on the output" signal, distinct from total
        connectivity.
    """
    n_sel = len(raw_graph.selected_features)
    adj = raw_graph.adjacency_matrix
    infl = adj.abs().sum(0)[:n_sel] + adj.abs().sum(1)[:n_sel]
    mi = infl.max().item() or 1.0

    # Direct attribution to logit nodes (EAP-style output effect).
    # Node layout: [features, error(n_layers*n_pos), embed(n_pos), logits].
    n_tokens = int(raw_graph.n_pos)
    n_layers_g = int(raw_graph.cfg.n_layers)
    n_logits = int(len(raw_graph.logit_targets))
    logit_start = n_sel + n_layers_g * n_tokens + n_tokens
    total_nodes = adj.shape[0]
    logit_end = min(logit_start + n_logits, total_nodes)
    if logit_end > logit_start:
        logit_attr = adj[logit_start:logit_end, :n_sel].abs().sum(0)
        me = logit_attr.max().item() or 1.0
    else:
        logit_attr = None
        me = 1.0

    by_layer: Dict[int, list] = {}
    for i in range(n_sel):
        ft = raw_graph.selected_features[i]
        layer = int(raw_graph.active_features[ft, 0].item())
        pos   = int(raw_graph.active_features[ft, 1].item())
        fidx  = int(raw_graph.active_features[ft, 2].item())
        act   = float(raw_graph.activation_values[i].item())
        imp   = float(infl[i].item()) / mi
        eap   = float(logit_attr[i].item()) / me if logit_attr is not None else imp

        in_deg  = int((adj[:, i].abs() > 0).sum().item())
        out_deg = int((adj[i, :].abs() > 0).sum().item())

        entry = dict(
            layer=layer, pos=pos, fidx=fidx, act=act, imp=imp, eap=eap,
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


def steer_feature_topk(
    model: ReplacementModel, prompt: str, feat: Dict,
    multiplier: float, clean_probs: torch.Tensor, clean_last: torch.Tensor,
    k: int = 5,
) -> Tuple[float, float, str, List[Tuple[str, float]]]:
    """Steer a feature and return (KL, logit_diff, new_top, top-k tokens).

    The top-k decoded tokens with their probabilities let us check whether a
    large steering factor amplifies an interpretable concept rather than merely
    degrading the distribution.
    """
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
    topv, topi = torch.topk(iv_probs, k)
    topk = [
        (model.tokenizer.decode([int(i)]), float(p))
        for p, i in zip(topv.tolist(), topi.tolist())
    ]
    new_top = topk[0][0] if topk else ""
    return max(0, kl), ld, new_top, topk


def patch_feature(
    model: ReplacementModel, prompt: str, feat: Dict,
    clean_probs: torch.Tensor, clean_last: torch.Tensor,
    ref_value: float = 0.0,
) -> Tuple[float, float]:
    """Patch a feature to a reference value and return (KL, logit_diff)."""
    iv, _ = model.feature_intervention(
        prompt, [(feat['layer'], feat['pos'], feat['fidx'], ref_value)],
        return_activations=False
    )
    iv_last = iv[0, -1, :]
    iv_probs = torch.softmax(iv_last, -1)
    kl = float(torch.nn.functional.kl_div(
        torch.log(iv_probs + 1e-10), clean_probs, reduction='sum'
    ).item())
    ld = float(torch.norm(iv_last - clean_last).item())
    return max(0, kl), ld


def execute_intervention(
    model: ReplacementModel, prompt: str, feat: Dict,
    action_name: str, clean_probs: torch.Tensor, clean_last: torch.Tensor,
) -> Tuple[float, float]:
    """Dispatch to the appropriate intervention based on agent's selected action."""
    if action_name == "activation_patching":
        return patch_feature(model, prompt, feat, clean_probs, clean_last)
    elif action_name == "feature_steering":
        kl, ld, _ = steer_feature(model, prompt, feat, STEER_MULT, clean_probs, clean_last)
        return kl, ld
    else:
        return ablate_feature(model, prompt, feat, clean_probs, clean_last)


# ======================================================================
# Shared per-prompt evaluation
# ======================================================================

def _run_agent_multi(
    model, prompt, candidates, clean_probs, clean_last,
    budget, n_layers, agent_kwargs,
):
    """Run the online multi-action POMDP agent over one prompt."""
    agent = ActiveInferencePOMDPAgent(n_layers=n_layers, **(agent_kwargs or {}))
    agent.initialize()
    observed: set = set()
    ai_kls, ai_actions = [], []
    for _ in range(min(budget, len(candidates))):
        unobserved = [c for c in candidates if c['fid'] not in observed]
        if not unobserved:
            break
        feat, action_name, _ = agent.select_intervention(unobserved)
        kl, _ = execute_intervention(model, prompt, feat, action_name, clean_probs, clean_last)
        agent.update_beliefs(
            feat, action_name=action_name, kl_divergence=kl,
            activation_value=feat['act'],
            graph_connectivity=feat.get('in_degree', 0) + feat.get('out_degree', 0),
        )
        observed.add(feat['fid'])
        ai_kls.append(kl)
        ai_actions.append(action_name)
    return ai_kls, ai_actions, agent


def _run_agent_abl(
    model, prompt, candidates, clean_probs, clean_last,
    budget, n_layers, agent_kwargs,
):
    """Run the EFE agent but force every intervention to be an ablation."""
    agent = ActiveInferencePOMDPAgent(n_layers=n_layers, **(agent_kwargs or {}))
    agent.initialize()
    observed: set = set()
    ai_abl_kls = []
    for _ in range(min(budget, len(candidates))):
        unobserved = [c for c in candidates if c['fid'] not in observed]
        if not unobserved:
            break
        feat, _, _ = agent.select_intervention(unobserved)
        kl, _ = ablate_feature(model, prompt, feat, clean_probs, clean_last)
        agent.update_beliefs(
            feat, action_name="ablation", kl_divergence=kl,
            activation_value=feat['act'],
            graph_connectivity=feat.get('in_degree', 0) + feat.get('out_degree', 0),
        )
        observed.add(feat['fid'])
        ai_abl_kls.append(kl)
    return ai_abl_kls, agent


def _precompute_ground_truth(model, prompt, candidates, clean_probs, clean_last):
    """Compute ablation and steering KL for every candidate (real interventions)."""
    gt: Dict[str, Dict[str, Any]] = {}
    for feat in candidates:
        kl_a, ld_a = ablate_feature(model, prompt, feat, clean_probs, clean_last)
        kl_s, _, _ = steer_feature(model, prompt, feat, STEER_MULT, clean_probs, clean_last)
        gt[feat['fid']] = {
            'kl': kl_a, 'ld': ld_a, 'layer': feat['layer'], 'kl_steer': kl_s,
        }
    gt_sorted = sorted(gt.items(), key=lambda x: x[1]['kl'], reverse=True)
    return gt, gt_sorted


def _run_baselines(candidates, gt, gt_sorted, budget, baseline_kwargs=None):
    """Run all ablation-only and action-matched baselines from precomputed KL."""
    nb = min(budget, len(candidates))
    bk = baseline_kwargs or {}
    expl_w = bk.get("exploration_weight", 2.0)
    seed = bk.get("seed", RANDOM_SEED)

    # --- ablation-only baselines ---
    bandit = BanditSelector(candidates, exploration_weight=expl_w)
    bandit_kls = []
    for _ in range(nb):
        idx, feat = bandit.select_next()
        kl = gt[feat['fid']]['kl']
        bandit.update(idx, kl)
        bandit_kls.append(kl)
    bandit_order = list(bandit.selection_order)

    # plain UCB1-over-layers bandit (no engineered priors)
    ucb = BanditUCBSelector(candidates, exploration_weight=expl_w)
    ucb_kls = []
    for _ in range(nb):
        idx, feat = ucb.select_next()
        kl = gt[feat['fid']]['kl']
        ucb.update(idx, kl)
        ucb_kls.append(kl)

    greedy_kls = [gt[candidates[i]['fid']]['kl'] for i in range(nb)]

    eap_order = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i].get('eap', 0.0), reverse=True,
    )
    eap_kls = [gt[candidates[i]['fid']]['kl'] for i in eap_order[:budget]]

    oracle_kls = [v['kl'] for _, v in gt_sorted[:budget]]

    # --- multi-action oracle (best by max(ablation, steering)) ---
    oracle_steer_kls = sorted(
        (max(v['kl'], v['kl_steer']) for v in gt.values()), reverse=True
    )[:budget]

    # --- action-matched baselines (same selection policy, steering action) ---
    greedy_steer_kls = [gt[candidates[i]['fid']]['kl_steer'] for i in range(nb)]
    bandit_steer_kls = [gt[candidates[idx]['fid']]['kl_steer'] for idx in bandit_order[:budget]]

    # --- stochastic baselines (10 trials) ---
    rng = np.random.default_rng(seed)
    rand_abl, rand_steer, rand_act = [], [], []
    for _ in range(N_RANDOM_TRIALS):
        sh = list(range(len(candidates)))
        rng.shuffle(sh)
        idxs = sh[:budget]
        rand_abl.append([gt[candidates[i]['fid']]['kl'] for i in idxs])
        rand_steer.append([gt[candidates[i]['fid']]['kl_steer'] for i in idxs])
        acts = rng.integers(0, len(ACTION_NAMES), size=len(idxs))
        ra = []
        for i, a in zip(idxs, acts):
            v = gt[candidates[i]['fid']]
            # action 2 == feature_steering; 0/1 (ablation/patching@0) == ablation KL
            ra.append(v['kl_steer'] if a == 2 else v['kl'])
        rand_act.append(ra)

    def _mean_cum(trials):
        return float(np.mean([np.sum(t) for t in trials]))

    def _mean_mean(trials):
        return float(np.mean([np.mean(t) for t in trials]))

    return {
        # ablation-only
        'bandit_kls': bandit_kls,
        'ucb_kls': ucb_kls,
        'greedy_kls': greedy_kls,
        'eap_kls': eap_kls,
        'oracle_kls': oracle_kls,
        'oracle_steer_kls': oracle_steer_kls,
        'bandit_cumkl': float(np.sum(bandit_kls)),
        'ucb_cumkl': float(np.sum(ucb_kls)),
        'greedy_cumkl': float(np.sum(greedy_kls)),
        'eap_cumkl': float(np.sum(eap_kls)),
        'oracle_cumkl': float(np.sum(oracle_kls)),
        'oracle_steer_cumkl': float(np.sum(oracle_steer_kls)),
        'bandit_mean': float(np.mean(bandit_kls)) if bandit_kls else 0.0,
        'ucb_mean': float(np.mean(ucb_kls)) if ucb_kls else 0.0,
        'greedy_mean': float(np.mean(greedy_kls)),
        'eap_mean': float(np.mean(eap_kls)) if eap_kls else 0.0,
        # action-matched
        'greedy_steer_kls': greedy_steer_kls,
        'bandit_steer_kls': bandit_steer_kls,
        'greedy_steer_cumkl': float(np.sum(greedy_steer_kls)),
        'bandit_steer_cumkl': float(np.sum(bandit_steer_kls)),
        # random variants (averaged over trials)
        'random_mean': _mean_mean(rand_abl),
        'random_cumkl': _mean_cum(rand_abl),
        'random_steer_cumkl': _mean_cum(rand_steer),
        'random_action_cumkl': _mean_cum(rand_act),
        'random_action_mean': _mean_mean(rand_act),
    }


def _layer_dist(gt_sorted, n_layers, top=10):
    early = sum(1 for _, v in gt_sorted[:top] if v['layer'] < n_layers // 3)
    mid   = sum(1 for _, v in gt_sorted[:top] if n_layers // 3 <= v['layer'] < 2 * n_layers // 3)
    late  = sum(1 for _, v in gt_sorted[:top] if v['layer'] >= 2 * n_layers // 3)
    return {'early': early, 'mid': mid, 'late': late}


def _evaluate_prompt(
    model, prompt, candidates, clean_probs, clean_last,
    budget, n_layers, agent_kwargs=None, baseline_kwargs=None,
) -> Tuple[Dict[str, Any], Any]:
    """Run every selector on a single prompt and return the per-prompt record.

    Returns (record_dict, multi_action_agent) so callers can attach
    task-specific fields and inspect agent diagnostics.
    """
    ai_kls, ai_actions, agent = _run_agent_multi(
        model, prompt, candidates, clean_probs, clean_last, budget, n_layers, agent_kwargs
    )
    ai_abl_kls, abl_agent = _run_agent_abl(
        model, prompt, candidates, clean_probs, clean_last, budget, n_layers, agent_kwargs
    )
    gt, gt_sorted = _precompute_ground_truth(model, prompt, candidates, clean_probs, clean_last)
    bl = _run_baselines(candidates, gt, gt_sorted, budget, baseline_kwargs)

    record = {
        'prompt': prompt,
        'n_candidates': len(candidates),
        'ai_kls': ai_kls,
        'ai_actions': ai_actions,
        'ai_abl_kls': ai_abl_kls,
        'ai_mean': float(np.mean(ai_kls)) if ai_kls else 0.0,
        'ai_abl_mean': float(np.mean(ai_abl_kls)) if ai_abl_kls else 0.0,
        'ai_cumkl': float(np.sum(ai_kls)),
        'ai_abl_cumkl': float(np.sum(ai_abl_kls)),
        'layer_distribution': _layer_dist(gt_sorted, n_layers),
        'agent_entropy_history': agent.get_belief_entropy_history(),
        'agent_efe_history': agent.get_efe_history(),
        'agent_a_convergence': agent.get_a_drift_history(),
        'agent_abl_a_convergence': abl_agent.get_a_drift_history(),
        'agent_converged': agent.is_converged,
    }
    record.update(bl)
    return record, gt_sorted


def _aggregate(all_results: List[Dict[str, Any]], task: str, budget: int) -> Dict[str, Any]:
    """Aggregate per-prompt records into mean/std and oracle-efficiency stats."""
    def sm(key):
        vals = [r[key] for r in all_results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    def ss(key):
        vals = [r[key] for r in all_results if key in r]
        return float(np.std(vals)) if vals else 0.0

    ai_mean = sm('ai_mean')
    ai_abl_mean = sm('ai_abl_mean')
    bandit_mean = sm('bandit_mean')
    ucb_mean = sm('ucb_mean')
    greedy_mean = sm('greedy_mean')
    eap_mean = sm('eap_mean')
    rand_mean = sm('random_mean')

    ai_cum = sm('ai_cumkl')
    ai_abl_cum = sm('ai_abl_cumkl')
    bandit_cum = sm('bandit_cumkl')
    ucb_cum = sm('ucb_cumkl')
    greedy_cum = sm('greedy_cumkl')
    eap_cum = sm('eap_cumkl')
    rand_cum = sm('random_cumkl')
    oracle_cum = sm('oracle_cumkl')
    oracle_steer_cum = sm('oracle_steer_cumkl')
    greedy_steer_cum = sm('greedy_steer_cumkl')
    bandit_steer_cum = sm('bandit_steer_cumkl')
    rand_steer_cum = sm('random_steer_cumkl')
    rand_act_cum = sm('random_action_cumkl')

    od = max(oracle_cum, 1e-10)

    def pct(a, b):
        return float((a - b) / max(b, 1e-10) * 100)

    return {
        'ai_mean_kl': ai_mean, 'ai_std_kl': ss('ai_mean'),
        'ai_abl_mean_kl': ai_abl_mean, 'ai_abl_std_kl': ss('ai_abl_mean'),
        'bandit_mean_kl': bandit_mean, 'bandit_std_kl': ss('bandit_mean'),
        'ucb_mean_kl': ucb_mean, 'ucb_std_kl': ss('ucb_mean'),
        'greedy_mean_kl': greedy_mean, 'greedy_std_kl': ss('greedy_mean'),
        'eap_mean_kl': eap_mean, 'eap_std_kl': ss('eap_mean'),
        'random_mean_kl': rand_mean, 'random_std_kl': ss('random_mean'),
        'ai_vs_random_pct': pct(ai_mean, rand_mean),
        'ai_vs_greedy_pct': pct(ai_mean, greedy_mean),
        'ai_vs_bandit_pct': pct(ai_mean, bandit_mean),
        # ablation-comparable oracle efficiency (bounded by the ablation oracle)
        'ai_abl_oracle_efficiency': float(ai_abl_cum / od * 100),
        'bandit_oracle_efficiency': float(bandit_cum / od * 100),
        'ucb_oracle_efficiency': float(ucb_cum / od * 100),
        'greedy_oracle_efficiency': float(greedy_cum / od * 100),
        'eap_oracle_efficiency': float(eap_cum / od * 100),
        'random_oracle_efficiency': float(rand_cum / od * 100),
        # multi-action relative cumulative KL (RCK) vs the ablation oracle
        'ai_oracle_efficiency': float(ai_cum / od * 100),   # kept for back-compat
        'ai_rck': float(ai_cum / od * 100),
        'greedy_steer_rck': float(greedy_steer_cum / od * 100),
        'bandit_steer_rck': float(bandit_steer_cum / od * 100),
        'random_steer_rck': float(rand_steer_cum / od * 100),
        'random_action_rck': float(rand_act_cum / od * 100),
        # multi-action oracle (best by max(ablation, steering))
        'oracle_steer_cumkl': oracle_steer_cum,
        'oracle_cumkl': oracle_cum,
    }


# ======================================================================
# IOI experiment
# ======================================================================

def run_ioi_experiment(
    model: ReplacementModel,
    prompts: List[str],
    budget: int = 20,
    max_per_layer: int = 3,
    max_candidates: int = 40,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    baseline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """IOI experiment comparing all selectors."""
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

        record, gt_sorted = _evaluate_prompt(
            model, prompt, candidates, clean_probs, clean_last,
            budget, n_layers, agent_kwargs, baseline_kwargs,
        )
        record['n_features'] = int(raw.active_features.shape[0])
        record['ground_truth_top5'] = [(fid, v['kl']) for fid, v in gt_sorted[:5]]
        all_results.append(record)

    return {
        'task': 'IOI',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': _aggregate(all_results, 'IOI', budget),
    }


# ======================================================================
# Steering experiment
# ======================================================================

def run_steering_experiment(
    model: ReplacementModel,
    prompts: List[str],
    multipliers: List[float] = [0.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    n_features: int = 10,
) -> Dict[str, Any]:
    """Feature steering experiment.

    Steers the top-``n_features`` features by graph importance across a sweep of
    multipliers (including smaller factors 1.5, 3 as OOD controls), and -- as a
    control for out-of-distribution effects -- also steers a matched set of
    *randomly chosen* features.  This lets us separate genuine causal control
    from generic activation-scaling perturbations.
    """
    all_results = []
    rng = np.random.default_rng(RANDOM_SEED)

    for pi, prompt in enumerate(prompts):
        logger.info(f"Steering prompt {pi+1}/{len(prompts)}: '{prompt[:50]}...'")

        raw = attribute(
            prompt=prompt, model=model, max_n_logits=5,
            desired_logit_prob=0.9, batch_size=256, verbose=False
        )
        # all candidates (for sampling a random control set)
        all_cands = extract_candidates(raw, max_per_layer=5, max_total=120)
        candidates = sorted(all_cands, key=lambda x: x['imp'], reverse=True)[:n_features]

        # random control features: sample from the lower-importance pool
        pool = [c for c in all_cands if c['fid'] not in {x['fid'] for x in candidates}]
        if len(pool) >= n_features:
            ctrl_idx = rng.choice(len(pool), size=n_features, replace=False)
            control = [pool[int(i)] for i in ctrl_idx]
        else:
            control = pool

        clean_logits, _ = model.feature_intervention(prompt, [], return_activations=False)
        clean_last = clean_logits[0, -1, :]
        clean_probs = torch.softmax(clean_last, -1)
        clean_top = model.tokenizer.decode([int(clean_probs.argmax().item())])

        def _steer_set(feats):
            results = []
            per_mult = {m: [] for m in multipliers}
            for feat in feats:
                fr = {'fid': feat['fid'], 'layer': feat['layer'], 'act': feat['act']}
                for mult in multipliers:
                    kl, ld, new_top, topk = steer_feature_topk(
                        model, prompt, feat, mult, clean_probs, clean_last, k=5
                    )
                    fr[f'mult_{mult}'] = {
                        'kl': kl, 'logit_diff': ld, 'new_top': new_top,
                        'prediction_changed': new_top != clean_top,
                        'top_tokens': topk,
                    }
                    per_mult[mult].append(kl)
                results.append(fr)
            mean_kl = [float(np.mean(per_mult[m])) if per_mult[m] else 0.0 for m in multipliers]
            return results, mean_kl

        steer_results, mean_kl = _steer_set(candidates)
        control_results, control_mean_kl = _steer_set(control)

        all_results.append({
            'prompt': prompt,
            'clean_prediction': clean_top,
            'n_features_tested': len(candidates),
            'features': steer_results,
            'mean_kl_per_multiplier': mean_kl,
            'control_features': control_results,
            'control_mean_kl_per_multiplier': control_mean_kl,
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
    agent_kwargs: Optional[Dict[str, Any]] = None,
    baseline_kwargs: Optional[Dict[str, Any]] = None,
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

        record, gt_sorted = _evaluate_prompt(
            model, prompt, candidates, clean_probs, clean_last,
            budget, n_layers, agent_kwargs, baseline_kwargs,
        )
        record['clean_prediction'] = clean_top
        record['top10_features'] = [
            (fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]
        ]
        all_results.append(record)

    return {
        'task': 'MultiStep',
        'budget': budget,
        'n_prompts': len(prompts),
        'per_prompt': all_results,
        'aggregate': _aggregate(all_results, 'MultiStep', budget),
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
    agent_kwargs: Optional[Dict[str, Any]] = None,
    baseline_kwargs: Optional[Dict[str, Any]] = None,
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

            record, gt_sorted = _evaluate_prompt(
                model, prompt, candidates, clean_probs, clean_last,
                budget, n_layers, agent_kwargs, baseline_kwargs,
            )
            record['clean_prediction'] = clean_top
            record['top10_features'] = [
                (fid, v['kl'], v['layer']) for fid, v in gt_sorted[:10]
            ]
            ld = record['layer_distribution']
            domain_early += ld['early']; domain_mid += ld['mid']; domain_late += ld['late']
            domain_results.append(record)

        ai_means     = [r['ai_mean'] for r in domain_results]
        ai_abl_means = [r['ai_abl_mean'] for r in domain_results]
        bandit_means = [r['bandit_mean'] for r in domain_results]
        ucb_means    = [r['ucb_mean'] for r in domain_results]
        greedy_means = [r['greedy_mean'] for r in domain_results]
        eap_means    = [r['eap_mean'] for r in domain_results]
        rand_means   = [r['random_mean'] for r in domain_results]

        # cumulative-KL means (for bounded oracle efficiency / RCK in the
        # head-to-head table)
        def _cm(key):
            return float(np.mean([r[key] for r in domain_results if key in r]))
        oracle_cum = _cm('oracle_cumkl') or 1e-10

        by_domain[domain] = {
            'per_prompt': domain_results,
            'layer_distribution': {'early': domain_early, 'mid': domain_mid, 'late': domain_late},
            'ai_mean_kl': float(np.mean(ai_means)),
            'ai_abl_mean_kl': float(np.mean(ai_abl_means)),
            'bandit_mean_kl': float(np.mean(bandit_means)),
            'ucb_mean_kl': float(np.mean(ucb_means)),
            'greedy_mean_kl': float(np.mean(greedy_means)),
            'eap_mean_kl': float(np.mean(eap_means)),
            'random_mean_kl': float(np.mean(rand_means)),
            'ai_abl_oracle_efficiency': float(_cm('ai_abl_cumkl') / oracle_cum * 100),
            'bandit_oracle_efficiency': float(_cm('bandit_cumkl') / oracle_cum * 100),
            'ucb_oracle_efficiency': float(_cm('ucb_cumkl') / oracle_cum * 100),
            'eap_oracle_efficiency': float(_cm('eap_cumkl') / oracle_cum * 100),
            'ai_rck': float(_cm('ai_cumkl') / oracle_cum * 100),
            'ai_vs_random_pct': float((np.mean(ai_means) - np.mean(rand_means)) / max(np.mean(rand_means), 1e-10) * 100),
            'ai_vs_greedy_pct': float((np.mean(ai_means) - np.mean(greedy_means)) / max(np.mean(greedy_means), 1e-10) * 100),
        }

    all_ai     = [d['ai_mean_kl'] for d in by_domain.values()]
    all_ai_abl = [d['ai_abl_mean_kl'] for d in by_domain.values()]
    all_bandit = [d['bandit_mean_kl'] for d in by_domain.values()]
    all_ucb    = [d['ucb_mean_kl'] for d in by_domain.values()]
    all_greedy = [d['greedy_mean_kl'] for d in by_domain.values()]
    all_eap    = [d['eap_mean_kl'] for d in by_domain.values()]
    all_rand   = [d['random_mean_kl'] for d in by_domain.values()]
    all_ai_abl_eff = [d['ai_abl_oracle_efficiency'] for d in by_domain.values()]
    all_bandit_eff = [d['bandit_oracle_efficiency'] for d in by_domain.values()]
    all_ucb_eff    = [d['ucb_oracle_efficiency'] for d in by_domain.values()]
    all_eap_eff    = [d['eap_oracle_efficiency'] for d in by_domain.values()]
    all_ai_rck     = [d['ai_rck'] for d in by_domain.values()]

    return {
        'task': 'Domain',
        'budget': budget,
        'by_domain': by_domain,
        'aggregate': {
            'ai_mean_kl': float(np.mean(all_ai)),
            'ai_abl_mean_kl': float(np.mean(all_ai_abl)),
            'bandit_mean_kl': float(np.mean(all_bandit)),
            'ucb_mean_kl': float(np.mean(all_ucb)),
            'greedy_mean_kl': float(np.mean(all_greedy)),
            'eap_mean_kl': float(np.mean(all_eap)),
            'random_mean_kl': float(np.mean(all_rand)),
            'ai_abl_oracle_efficiency': float(np.mean(all_ai_abl_eff)),
            'bandit_oracle_efficiency': float(np.mean(all_bandit_eff)),
            'ucb_oracle_efficiency': float(np.mean(all_ucb_eff)),
            'eap_oracle_efficiency': float(np.mean(all_eap_eff)),
            'ai_rck': float(np.mean(all_ai_rck)),
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
    parser.add_argument(
        "--n-layer-role", type=int, default=None,
        help="Override the number of layer-role bins in the POMDP agent "
             "(default: agent default of 3). Used for the finer-layer-bin remedy.",
    )
    parser.add_argument(
        "--kl-threshold-scale", type=float, default=1.0,
        help="Multiply the KL discretisation thresholds by this factor "
             "(sensitivity analysis; e.g. 0.8 or 1.2).",
    )
    parser.add_argument(
        "--act-threshold-scale", type=float, default=1.0,
        help="Multiply the activation discretisation thresholds by this factor.",
    )
    parser.add_argument(
        "--imp-scale", type=float, default=None,
        help="Override the graph-importance -> KL prior scaling factor "
             "(default 0.01; sensitivity analysis).",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Override the POMDP action-precision gamma "
             "(agent default 16.0; sensitivity analysis).",
    )
    parser.add_argument(
        "--lr-pa", type=float, default=None,
        help="Override the Dirichlet likelihood learning rate eta=lr_pA "
             "(agent default 1.0; sensitivity analysis).",
    )
    parser.add_argument(
        "--pragmatic-weight", type=float, default=None,
        help="Override the pragmatic (preference) weight on C "
             "(agent default 1.0; sensitivity analysis).",
    )
    parser.add_argument(
        "--exploration-weight", type=float, default=2.0,
        help="UCB exploration weight for the heuristic bandit baseline "
             "(default 2.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for stochastic baselines (default {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Optional filename suffix for result JSON (e.g. '_finebins').",
    )
    return parser.parse_args()


def _apply_discretisation_overrides(kl_scale: float, act_scale: float) -> None:
    """Scale the module-level discretisation thresholds in-place (sensitivity)."""
    if kl_scale != 1.0:
        pomdp_mod.KL_THRESHOLDS = tuple(t * kl_scale for t in pomdp_mod.KL_THRESHOLDS)
        logger.info(f"KL_THRESHOLDS scaled by {kl_scale}: {pomdp_mod.KL_THRESHOLDS}")
    if act_scale != 1.0:
        pomdp_mod.ACT_THRESHOLDS = tuple(t * act_scale for t in pomdp_mod.ACT_THRESHOLDS)
        logger.info(f"ACT_THRESHOLDS scaled by {act_scale}: {pomdp_mod.ACT_THRESHOLDS}")


def main():
    args = _parse_args()

    models_to_run = ["gemma", "llama"] if args.model == "both" else [args.model]
    experiments_to_run = (
        ["ioi", "steering", "multistep", "domain"]
        if args.experiment == "all"
        else [args.experiment]
    )

    # Build agent kwargs for the finer-layer-bin remedy / importance scaling /
    # hyperparameter sensitivity sweeps.
    agent_kwargs: Dict[str, Any] = {}
    if args.n_layer_role is not None:
        agent_kwargs["n_layer_role"] = args.n_layer_role
    if args.imp_scale is not None:
        agent_kwargs["imp_scale"] = args.imp_scale
    if args.gamma is not None:
        agent_kwargs["gamma"] = args.gamma
    if args.lr_pa is not None:
        agent_kwargs["lr_pA"] = args.lr_pa
    if args.pragmatic_weight is not None:
        agent_kwargs["pragmatic_weight"] = args.pragmatic_weight

    # Baseline knobs (bandit exploration weight + global seed).
    global RANDOM_SEED
    RANDOM_SEED = args.seed
    baseline_kwargs: Dict[str, Any] = {
        "exploration_weight": args.exploration_weight,
        "seed": args.seed,
    }

    _apply_discretisation_overrides(args.kl_threshold_scale, args.act_threshold_scale)

    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    graphs_dir = results_dir / "graphs"
    results_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    suffix = args.suffix

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
            ioi_results = run_ioi_experiment(model, IOI_PROMPTS, budget=20, agent_kwargs=agent_kwargs, baseline_kwargs=baseline_kwargs)
            ioi_results["elapsed_seconds"] = time.time() - t0
            ioi_results["model"] = cfg["model_name"]
            ioi_results["agent_kwargs"] = agent_kwargs
            out_path = results_dir / f"ioi_results_{model_key}{suffix}.json"
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
            out_path = results_dir / f"steering_results_{model_key}{suffix}.json"
            with open(out_path, "w") as f:
                json.dump(steer_results, f, indent=2)
            logger.info(f"Steering results -> {out_path}")

        if "multistep" in experiments_to_run:
            logger.info(f"Running multi-step experiment [{model_key}]...")
            _save_graph_for_prompt(model, MULTISTEP_PROMPTS[0], f"multistep_{model_key}", graphs_dir)
            t0 = time.time()
            ms_results = run_multistep_experiment(model, MULTISTEP_PROMPTS, budget=20, agent_kwargs=agent_kwargs, baseline_kwargs=baseline_kwargs)
            ms_results["elapsed_seconds"] = time.time() - t0
            ms_results["model"] = cfg["model_name"]
            ms_results["agent_kwargs"] = agent_kwargs
            out_path = results_dir / f"multistep_results_{model_key}{suffix}.json"
            with open(out_path, "w") as f:
                json.dump(ms_results, f, indent=2)
            logger.info(f"Multi-step results -> {out_path}")

        if "domain" in experiments_to_run:
            logger.info(f"Running domain experiment [{model_key}]...")
            _save_graph_for_prompt(model, DOMAIN_PROMPTS["geography"][0], f"domain_{model_key}", graphs_dir)
            t0 = time.time()
            domain_results = run_domain_experiment(model, DOMAIN_PROMPTS, budget=20, agent_kwargs=agent_kwargs, baseline_kwargs=baseline_kwargs)
            domain_results["elapsed_seconds"] = time.time() - t0
            domain_results["model"] = cfg["model_name"]
            domain_results["agent_kwargs"] = agent_kwargs
            out_path = results_dir / f"domain_results_{model_key}{suffix}.json"
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
            p = results_dir / f"{exp_name}_results_{model_key}{suffix}.json"
            if not p.exists():
                continue
            with open(p) as f:
                r = json.load(f)
            agg = r.get("aggregate", {})
            if exp_name == "ioi":
                print(f"\n  IOI ({r['n_prompts']} prompts, budget={r['budget']}):")
                print(f"    POMDP-abl mean KL:   {agg.get('ai_abl_mean_kl', 0):.6f}")
                print(f"    Bandit mean KL:      {agg.get('bandit_mean_kl', 0):.6f}")
                print(f"    UCB mean KL:         {agg.get('ucb_mean_kl', 0):.6f}")
                print(f"    EAP mean KL:         {agg.get('eap_mean_kl', 0):.6f}")
                print(f"    Greedy mean KL:      {agg.get('greedy_mean_kl', 0):.6f}")
                print(f"    Random mean KL:      {agg.get('random_mean_kl', 0):.6f}")
                print(f"    Oracle eff: POMDP-abl={agg.get('ai_abl_oracle_efficiency', 0):.1f}%"
                      f"  bandit={agg.get('bandit_oracle_efficiency', 0):.1f}%"
                      f"  eap={agg.get('eap_oracle_efficiency', 0):.1f}%")
                print(f"    RCK (multi-action): POMDP={agg.get('ai_rck', 0):.1f}%"
                      f"  greedy+steer={agg.get('greedy_steer_rck', 0):.1f}%"
                      f"  bandit+steer={agg.get('bandit_steer_rck', 0):.1f}%")
            elif exp_name == "multistep":
                print(f"\n  Multi-step ({r['n_prompts']} prompts, budget={r['budget']}):")
                print(f"    POMDP-abl oracle eff: {agg.get('ai_abl_oracle_efficiency', 0):.1f}%"
                      f"  bandit={agg.get('bandit_oracle_efficiency', 0):.1f}%")
                print(f"    RCK (multi-action): POMDP={agg.get('ai_rck', 0):.1f}%"
                      f"  greedy+steer={agg.get('greedy_steer_rck', 0):.1f}%")
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
                print(f"    Prediction changes (top features): {n_changed}/{n_total}")
            elif exp_name == "domain":
                print(f"\n  Domain (5 domains):")
                for domain, d in r.get("by_domain", {}).items():
                    ld = d.get("layer_distribution", {})
                    print(f"    {domain}: [early={ld.get('early',0)}, mid={ld.get('mid',0)}, late={ld.get('late',0)}]")


if __name__ == "__main__":
    main()
