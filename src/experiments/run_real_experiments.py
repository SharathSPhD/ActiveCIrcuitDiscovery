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
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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


def main():
    logger.info("Loading Gemma-2-2B with transcoders...")
    model = ReplacementModel.from_pretrained(
        model_name='google/gemma-2-2b',
        transcoder_set='gemma',
        backend='transformerlens',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32,
    )
    logger.info("Model loaded.")

    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # IOI experiment
    logger.info("Running IOI experiment...")
    t0 = time.time()
    ioi_results = run_ioi_experiment(model, IOI_PROMPTS, budget=20)
    ioi_results['elapsed_seconds'] = time.time() - t0
    ioi_results['model'] = 'google/gemma-2-2b'

    with open(results_dir / 'ioi_results.json', 'w') as f:
        json.dump(ioi_results, f, indent=2)
    logger.info(f"IOI results saved. AI vs Random: {ioi_results['aggregate']['ai_vs_random_pct']:+.1f}%")

    # Steering experiment
    logger.info("Running steering experiment...")
    t0 = time.time()
    steer_results = run_steering_experiment(model, STEERING_PROMPTS)
    steer_results['elapsed_seconds'] = time.time() - t0
    steer_results['model'] = 'google/gemma-2-2b'

    with open(results_dir / 'steering_results.json', 'w') as f:
        json.dump(steer_results, f, indent=2)
    logger.info("Steering results saved.")

    # Multi-step reasoning experiment
    logger.info("Running multi-step reasoning experiment...")
    t0 = time.time()
    ms_results = run_multistep_experiment(model, MULTISTEP_PROMPTS, budget=20)
    ms_results['elapsed_seconds'] = time.time() - t0
    ms_results['model'] = 'google/gemma-2-2b'

    with open(results_dir / 'multistep_results.json', 'w') as f:
        json.dump(ms_results, f, indent=2)
    logger.info("Multi-step results saved.")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    agg = ioi_results['aggregate']
    print(f"\nIOI (5 prompts, budget=20):")
    print(f"  AI mean KL:      {agg['ai_mean_kl']:.6f} +/- {agg['ai_std_kl']:.6f}")
    print(f"  Greedy mean KL:  {agg['greedy_mean_kl']:.6f} +/- {agg['greedy_std_kl']:.6f}")
    print(f"  Random mean KL:  {agg['random_mean_kl']:.6f} +/- {agg['random_std_kl']:.6f}")
    print(f"  AI vs Random:    {agg['ai_vs_random_pct']:+.1f}%")
    print(f"  AI vs Greedy:    {agg['ai_vs_greedy_pct']:+.1f}%")
    print(f"  Oracle efficiency: AI={agg['ai_oracle_efficiency']:.1f}% "
          f"Greedy={agg['greedy_oracle_efficiency']:.1f}% "
          f"Random={agg['random_oracle_efficiency']:.1f}%")

    n_changed = 0
    n_total = 0
    for pr in steer_results['per_prompt']:
        for feat in pr['features']:
            for mult in [5.0, 10.0]:
                key = f'mult_{mult}'
                if key in feat:
                    n_total += 1
                    if feat[key]['prediction_changed']:
                        n_changed += 1
    print(f"\nSteering ({len(STEERING_PROMPTS)} prompts):")
    print(f"  Prediction changes: {n_changed}/{n_total} ({n_changed/n_total*100:.0f}%)")

    ms_agg = ms_results['aggregate']
    print(f"\nMulti-step Reasoning ({len(MULTISTEP_PROMPTS)} prompts, budget=20):")
    print(f"  AI mean KL:      {ms_agg['ai_mean_kl']:.6f} +/- {ms_agg['ai_std_kl']:.6f}")
    print(f"  Greedy mean KL:  {ms_agg['greedy_mean_kl']:.6f} +/- {ms_agg['greedy_std_kl']:.6f}")
    print(f"  Random mean KL:  {ms_agg['random_mean_kl']:.6f} +/- {ms_agg['random_std_kl']:.6f}")
    print(f"  AI vs Random:    {ms_agg['ai_vs_random_pct']:+.1f}%")
    print(f"  AI vs Greedy:    {ms_agg['ai_vs_greedy_pct']:+.1f}%")
    print(f"  Oracle efficiency: AI={ms_agg['ai_oracle_efficiency']:.1f}%")
    for r in ms_results['per_prompt']:
        ld = r['layer_distribution']
        print(f"    '{r['prompt'][:50]}...' -> '{r['clean_prediction']}' "
              f"[early={ld['early']}, mid={ld['mid']}, late={ld['late']}]")


if __name__ == '__main__':
    main()
